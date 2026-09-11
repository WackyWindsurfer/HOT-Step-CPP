import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { setTimeout as delay } from 'node:timers/promises';
import test from 'node:test';
import Database from 'better-sqlite3';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { getDefaultEnvironment, StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { DiscussionStore } from '../src/collaboration.js';
import { CHANNEL_NOTIFICATION } from '../src/discussion-wake.js';

const packageDir = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const HUMAN = '11111111-1111-4111-8111-111111111111';
const EVIDENCE = ['engine/src/minimax/mm3-depth-graph.h:687'];

test('channel wake: one notification per unseen event, only for an idle participant', { timeout: 40000 }, async () => {
  const temp = mkdtempSync(join(tmpdir(), 'hotstep-wake-test-'));
  const dbPath = join(temp, 'wake.db');
  const client = new Client({ name: 'wake-test', version: '1.0.0' });
  const received: { content: string; meta: Record<string, string> }[] = [];
  client.fallbackNotificationHandler = async notification => {
    if (notification.method === CHANNEL_NOTIFICATION) received.push(notification.params as { content: string; meta: Record<string, string> });
  };
  const store = new DiscussionStore(dbPath);
  const raw = new Database(dbPath);
  const room = 'wake-room';
  const claude = store.join(room, 'Claude', 'Wake test brief.', { blind_positions: true, base_commit: null });
  store.joinViewer(room, HUMAN);
  const settle = () => delay(450);
  // The chat's turn ended without leaving: its presence lease lapses. Simulated by expiring the row.
  const idle = (id: string) => { raw.prepare('DELETE FROM discussion_presence WHERE participant_id = ?').run(id); };
  const call = async (name: string, args: Record<string, unknown>) => {
    const response = await client.callTool({ name, arguments: args });
    const content = response.content as { type: string; text: string }[];
    if (response.isError) throw new Error(content[0].text);
    return JSON.parse(content[0].text);
  };
  try {
    await client.connect(new StdioClientTransport({
      command: process.execPath,
      args: [...process.execArgv.filter(arg => arg.startsWith('--preserve-symlinks')), '--import', 'tsx', 'src/collaboration-server.ts'],
      cwd: packageDir,
      env: { ...getDefaultEnvironment(), HOTSTEP_COLLAB_DB: dbPath, HOTSTEP_COLLAB_CHANNEL: '1', HOTSTEP_COLLAB_WAKE_POLL_MS: '100' },
      stderr: 'inherit',
    }));
    assert.deepEqual(client.getServerCapabilities()?.experimental, { 'claude/channel': {} });
    const joined = await call('collab_join_discussion', { room, name: 'Codex', role: 'app/integration lead' });
    const codex = joined.participant_id as string;

    // Events before the join never wake. A self-triggered reveal never wakes.
    idle(codex);
    await settle();
    assert.equal(received.length, 0);
    store.submitPosition(room, claude.participant_id, 'pos', 'Claude position.', EVIDENCE);
    await call('collab_submit_position', { room, participant_id: codex, request_id: 'pos', body: 'Codex position.', evidence: EVIDENCE });
    idle(codex);
    await settle();
    assert.deepEqual(received.map(r => r.meta.event), []);

    // A user direction wakes the idle participant once, with room and message id attached.
    const direction = store.post(room, HUMAN, 'dir-1', 'user_direction', 'All of you: focus on VRAM.');
    await settle();
    assert.deepEqual(received.map(r => [r.meta.event, r.meta.message_id, r.meta.room, r.meta.participant_id]), [['user_direction', String(direction.id), room, codex]]);
    assert.match(received[0].content, /Resume participation in room "wake-room"/);
    await settle();
    assert.equal(received.length, 1);

    // A present participant (between two waits) is never woken; the event waits for it to go idle.
    const page = await call('collab_read_discussion', { room, participant_id: codex, after_id: 0, limit: 100 });
    const nudge = store.post(room, HUMAN, 'dir-2', 'user_direction', 'Codex, while you are composing.');
    await settle();
    assert.equal(received.length, 1, 'present participant: the next wait delivers it');
    idle(codex);
    await settle();
    assert.deepEqual(received.slice(1).map(r => [r.meta.event, r.meta.message_id]), [['user_direction', String(nudge.id)]]);

    // Reading moves the cursor: nothing already seen wakes again; a ping wakes once.
    const cursor1 = (await call('collab_read_discussion', { room, participant_id: codex, after_id: page.next_after_id, limit: 100 })).next_after_id;
    const ping = store.post(room, claude.participant_id, 'ping', 'reply', '@codex what about the depth head?', undefined, cursor1);
    idle(codex);
    await settle();
    assert.deepEqual(received.slice(2).map(r => [r.meta.event, r.meta.message_id]), [['reply_requested', String(ping.id)]]);
    assert.match(received[2].content, /Claude requested your reply at message #/);

    // While this connection waits, the wait delivers the message and no wake fires afterwards.
    const cursor2 = (await call('collab_read_discussion', { room, participant_id: codex, after_id: 0, limit: 100 })).next_after_id;
    const waiting = call('collab_wait_for_message', { room, participant_id: codex, after_id: cursor2, timeout_ms: 3000 });
    await delay(150);
    await call('collab_post_message', { room, participant_id: codex, request_id: 'answer', body: 'Depth head stays.', read_after_id: cursor2 });
    const second = store.post(room, HUMAN, 'dir-3', 'user_direction', 'Thanks. Codex, also cover VRAM.');
    const delivered = await waiting;
    assert.ok(delivered.messages.some((m: { id: number }) => m.id === second.id));
    idle(codex);
    await settle();
    assert.equal(received.length, 3, 'the wait delivered the direction; no wake should duplicate it');

    // Paused rooms are silent; resuming wakes once.
    store.status(room, HUMAN, 'pause', 'paused', 'Pause for the ruling.');
    await settle();
    assert.equal(received.length, 3);
    const resumed = store.status(room, HUMAN, 'resume', 'active', 'Ruling posted; carry on.');
    await settle();
    assert.deepEqual(received.slice(3).map(r => [r.meta.event, r.meta.message_id]), [['resumed', String(resumed.event.id)]]);

    // Leaving is the idle case: wakes continue, and a backlog past 100 messages is still scanned.
    await call('collab_leave_discussion', { room, participant_id: codex });
    const gemini = store.join(room, 'Gemini');
    let last = store.read(room, 0, 1000).next_after_id;
    for (let i = 0; i < 120; i++) {
      const author = i % 2 ? gemini : claude;
      last = store.post(room, author.participant_id, `filler-${i}`, 'reply', `Filler ${i}`, undefined, last).id;
    }
    const late = store.post(room, HUMAN, 'dir-4', 'user_direction', 'Anyone there?');
    await settle();
    assert.deepEqual(received.slice(4).map(r => [r.meta.event, r.meta.message_id]), [['user_direction', String(late.id)]]);
    await settle();
    assert.equal(received.length, 5);
  } finally {
    await client.close();
    raw.close();
    store.close();
    const target = resolve(temp);
    assert.ok(target.startsWith(resolve(tmpdir()) + sep) && target.includes('hotstep-wake-test-'));
    rmSync(target, { recursive: true, force: true });
  }
});

test('without HOTSTEP_COLLAB_CHANNEL the server declares no channel capability', { timeout: 20000 }, async () => {
  const temp = mkdtempSync(join(tmpdir(), 'hotstep-wake-test-'));
  const client = new Client({ name: 'wake-off', version: '1.0.0' });
  try {
    await client.connect(new StdioClientTransport({
      command: process.execPath,
      args: [...process.execArgv.filter(arg => arg.startsWith('--preserve-symlinks')), '--import', 'tsx', 'src/collaboration-server.ts'],
      cwd: packageDir,
      env: { ...getDefaultEnvironment(), HOTSTEP_COLLAB_DB: join(temp, 'off.db'), HOTSTEP_COLLAB_CHANNEL: '' },
      stderr: 'inherit',
    }));
    assert.equal(client.getServerCapabilities()?.experimental, undefined);
  } finally {
    await client.close();
    const target = resolve(temp);
    assert.ok(target.startsWith(resolve(tmpdir()) + sep) && target.includes('hotstep-wake-test-'));
    rmSync(target, { recursive: true, force: true });
  }
});
