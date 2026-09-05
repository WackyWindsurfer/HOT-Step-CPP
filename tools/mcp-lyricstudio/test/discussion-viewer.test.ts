import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, sep } from 'node:path';
import { once } from 'node:events';
import { get as httpGet } from 'node:http';
import test from 'node:test';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport, getDefaultEnvironment } from '@modelcontextprotocol/sdk/client/stdio.js';
import { fileURLToPath } from 'node:url';
import { DiscussionStore } from '../src/collaboration.js';
import { createDiscussionViewer } from '../src/discussion-viewer.js';

test('group chat HTTP and MCP share the transcript without app access', { timeout: 15000 }, async t => {
  const temp = mkdtempSync(join(tmpdir(), 'hotstep-viewer-test-'));
  const dbPath = join(temp, 'collaboration.db');
  const server = createDiscussionViewer(dbPath);
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const address = server.address();
  assert.ok(address && typeof address !== 'string');
  const base = `http://127.0.0.1:${address.port}`;
  let store: DiscussionStore | undefined;
  const client = new Client({ name: 'viewer-test-agent', version: '1.0.0' });
  const participant = randomUUID();
  const write = (endpoint: string, value: Record<string, unknown>, origin = base) => fetch(`${base}/api/discussions/review/${endpoint}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json', Origin: origin },
    body: JSON.stringify({ participant_id: participant, request_id: randomUUID(), ...value }),
  });
  try {
    await t.test('an empty viewer serves its UI and does not create a database', async () => {
      const index = await fetch(base);
      assert.equal(index.status, 200);
      assert.match(await index.text(), /Your message to both agents/);
      assert.match(index.headers.get('content-security-policy')!, /default-src 'self'/);
      for (const asset of ['/viewer.js', '/viewer.css']) assert.equal((await fetch(base + asset)).status, 200);
      assert.deepEqual(await (await fetch(base + '/api/discussions')).json(), { discussions: [] });
      assert.equal(existsSync(dbPath), false);
    });
    store = new DiscussionStore(dbPath);
    const agent = store.join('review', 'Codex', 'Review the design');
    await client.connect(new StdioClientTransport({
      command: process.execPath,
      args: ['--import', 'tsx', 'src/collaboration-server.ts'],
      cwd: fileURLToPath(new URL('..', import.meta.url)),
      env: { ...getDefaultEnvironment(), HOTSTEP_COLLAB_DB: dbPath },
      stderr: 'inherit',
    }));

    await t.test('a human post reaches a waiting MCP agent and retry does not duplicate it', async () => {
      const waiting = client.callTool({ name: 'collab_wait_for_message', arguments: { room: 'review', after_id: 0, timeout_ms: 2000 } });
      const request_id = randomUUID();
      const body = 'Can we keep this backwards compatible? <script>alert(1)</script>';
      const response = await write('messages', { body, request_id });
      assert.equal(response.status, 200);
      const sent = await response.json() as { id: number; body: string; author: string; kind: string };
      assert.equal(sent.author, 'You');
      assert.equal(sent.kind, 'user_direction');
      assert.equal(sent.body, body);
      const result = await waiting;
      const received = JSON.parse((result.content as { text: string }[])[0].text);
      assert.equal(received.messages[0].id, sent.id);
      const retry = await write('messages', { body, request_id });
      assert.equal((await retry.json() as { id: number }).id, sent.id);
      assert.equal((await write('messages', { body: 'different', request_id })).status, 409);
      assert.equal(store!.read('review', 0, 100).messages.length, 1);
    });
    await t.test('agent replies appear through HTTP, including pagination and proposed decisions', async () => {
      const first = store!.read('review', 0, 100).messages[0];
      for (let i = 0; i < 105; i++) store!.post('review', agent.participant_id, `reply-${i}`, 'reply', `Reply ${i}`, first.id);
      store!.decide('review', agent.participant_id, 'plan', 0, 'Keep compatibility', 'Check the old format');
      const page = await (await fetch(base + '/api/discussions/review')).json() as any;
      assert.equal(page.messages.length, 100);
      assert.equal(page.has_more, true);
      assert.equal(page.decision.plan, 'Keep compatibility');
      const rest = await (await fetch(`${base}/api/discussions/review?after_id=${page.next_after_id}`)).json() as any;
      assert.equal(rest.messages.length, 7);
      assert.equal(rest.has_more, false);
      const list = await (await fetch(base + '/api/discussions')).json() as any;
      assert.equal(list.discussions[0].id, 'review');
      const readonly = new DiscussionStore(dbPath, { readonly: true });
      try { assert.throws(() => readonly.post('review', agent.participant_id, 'readonly', 'reply', 'No write'), /readonly/i); }
      finally { readonly.close(); }
    });
    await t.test('human pause wakes agents, rejects further messages, and resume restores posting', async () => {
      const cursor = store!.read('review', 0, 1000).next_after_id;
      const waiting = client.callTool({ name: 'collab_wait_for_message', arguments: { room: 'review', after_id: cursor, timeout_ms: 2000 } });
      assert.equal((await write('status', { status: 'paused', body: 'Hold on, I want to rethink this.' })).status, 200);
      const received = JSON.parse(((await waiting).content as { text: string }[])[0].text);
      assert.equal(received.discussion.status, 'paused');
      assert.equal((await write('messages', { body: 'Paused post' })).status, 409);
      assert.equal((await write('status', { status: 'active', body: 'Continue.' })).status, 200);
      assert.equal((await write('messages', { body: 'Consider the simpler option.' })).status, 200);
    });
    await t.test('invalid input, other origins, and agent impersonation are rejected', async () => {
      assert.equal((await write('messages', { body: 'Wrong origin' }, 'https://example.com')).status, 403);
      assert.equal((await write('messages', { body: ' ' })).status, 400);
      assert.equal((await write('messages', { body: 'x'.repeat(24001) })).status, 400);
      assert.equal((await write('messages', { participant_id: agent.participant_id, body: 'Pretend to be Codex' })).status, 409);
      assert.equal((await fetch(base + '/api/discussions/review?after_id=-1')).status, 400);
      assert.equal((await fetch(base + '/api/discussions/missing')).status, 404);
      assert.equal((await fetch(base + '/api/discussions', { method: 'DELETE' })).status, 405);
      const foreignHostStatus = await new Promise<number | undefined>((resolve, reject) => {
        httpGet(base + '/api/discussions', { headers: { Host: 'example.com' } }, response => {
          response.resume();
          resolve(response.statusCode);
        }).on('error', reject);
      });
      assert.equal(foreignHostStatus, 403);
    });
  } finally {
    await client.close();
    server.closeAllConnections();
    await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
    store?.close();
    const target = resolve(temp);
    assert.ok(target.startsWith(resolve(tmpdir()) + sep) && target.includes('hotstep-viewer-test-'));
    rmSync(target, { recursive: true, force: true });
  }
});
