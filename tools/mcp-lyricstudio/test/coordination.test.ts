import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, sep } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import test from 'node:test';
import Database from 'better-sqlite3';
import { DiscussionStore } from '../src/collaboration.js';

test('mentions and research holds coordinate agents without model invocations', async t => {
  const temp = mkdtempSync(join(tmpdir(), 'hotstep-coordination-'));
  const dbPath = join(temp, 'room.db');
  const store = new DiscussionStore(dbPath);
  const peer = new DiscussionStore(dbPath);
  const human = randomUUID();
  const codex = store.join('room', 'Codex', 'Review generation performance').participant_id;
  const claude = store.join('room', 'Claude').participant_id;
  store.joinViewer('room', human);
  const say = (id: string, body: string, cursor?: number) => store.post('room', id, randomUUID(), id === human ? 'user_direction' : 'reply', body, undefined, cursor);
  const page = () => store.read('room', 0, 1000);
  try {
    await t.test('mentions resolve known handles, ignore emails/code, and combine pending pings', () => {
      const m = say(human, '@CLAUDE, please check. @claude another detail. email@codex.com `@codex`');
      const next = say(human, '@claude use the current code.');
      assert.equal(page().coordination.requests.length, 1);
      assert.equal(page().coordination.requests[0].message_id, next.id);
      assert.deepEqual(page().messages.find(x => x.id === m.id)!.mentions, [{ participant_id: claude, name: 'Claude' }]);
      assert.throws(() => say(codex, 'I will answer instead.'), /requested from Claude/);
      assert.throws(() => store.activity('room', codex, 'researching', 'Unrequested', 120), /requested from Claude/);
    });
    await t.test('a research hold is visible across connections without consuming a reply', () => {
      const count = page().messages.length;
      store.activity('room', claude, 'researching', 'Checking the depth graph.', 120);
      assert.equal(peer.read('room', 0, 100).coordination.research?.name, 'Claude');
      assert.equal(page().messages.length, count);
      assert.throws(() => say(codex, 'New proposal'), /researching/);
      assert.throws(() => store.decide('room', codex, 'blocked-plan', 0, 'Plan', ''), /researching/);
      assert.throws(() => peer.activity('room', codex, 'researching', 'Competing claim', 120), /researching/);
    });
    await t.test('human directions stay open and a stale research answer is rejected', () => {
      const stale = page().next_after_id;
      say(human, 'Check the new quantisation setting too.');
      assert.throws(() => say(claude, 'Answer', stale), /Read all new room messages/);
      assert.throws(() => say(claude, 'Answer', page().next_after_id + 100), /Read all new room messages/);
      assert.throws(() => say(claude, 'Answer'), /read_after_id/);
      const answer = say(claude, 'Both settings use the same graph.', page().next_after_id);
      assert.equal(page().coordination.research, null);
      assert.equal(page().coordination.requests.length, 0);
      assert.equal(page().messages.at(-1)!.id, answer.id);
      assert.throws(() => say(claude, 'One more thing'), /Wait for another/);
    });
    await t.test('other agents cannot release a hold; renewal and human release work', () => {
      say(human, '@codex investigate next.');
      const old = store.activity('room', codex, 'researching', 'Checking kernels.', 30).research!;
      store.activity('room', claude, 'idle', '', 120);
      assert.equal(page().coordination.research?.participant_id, codex);
      const renewed = store.activity('room', codex, 'researching', 'Checking kernels and shapes.', 120).research!;
      assert.ok(renewed.expires_at > old.expires_at);
      assert.throws(() => store.clearCoordination('room', claude, 'fake-human', 'release_research'), /Only the human/);
      const request = randomUUID();
      store.clearCoordination('room', human, request, 'release_research');
      store.clearCoordination('room', human, request, 'release_research');
      assert.equal(page().coordination.research, null);
      assert.equal(page().coordination.requests.length, 1);
    });
    await t.test('declining resolves a request without manufacturing another discussion turn', () => {
      const request = page().coordination.requests[0];
      assert.throws(() => store.declineRequest('room', codex, 'decline', 'Nothing to add.', request.message_id - 1), /Read the latest ping/);
      const a = store.declineRequest('room', codex, 'decline', 'Nothing to add.', page().next_after_id);
      assert.deepEqual(store.declineRequest('room', codex, 'decline', 'Nothing to add.', page().next_after_id), a);
      assert.equal(page().coordination.requests.length, 0);
    });
    await t.test('a hold expires visibly and a waiting reader returns on activity change', async () => {
      store.activity('room', codex, 'researching', 'Short investigation.', 30);
      const raw = new Database(dbPath);
      try { raw.prepare('UPDATE discussion_research SET expires_at = ?').run(Date.now() + 80); }
      finally { raw.close(); }
      const result = await peer.wait('room', page().next_after_id, 1000, 50);
      assert.equal(result.timed_out, false);
      assert.equal(result.coordination.research, null);
      assert.equal(result.messages.length, 0);
      const waiting = peer.wait('room', page().next_after_id, 1000, 50);
      await delay(30);
      store.activity('room', codex, 'researching', 'Started another investigation.', 30);
      assert.equal((await waiting).coordination.research?.name, 'Codex');
    });
    await t.test('pause cancels research and prevents new holds; requests survive until human clears', () => {
      say(human, '@claude review later.');
      store.status('room', human, 'pause', 'paused', 'Hold the whole discussion.');
      assert.equal(page().coordination.research, null);
      assert.throws(() => store.activity('room', claude, 'researching', 'Cannot start', 30), /paused/);
      assert.equal(page().coordination.requests.length, 1);
      store.clearCoordination('room', human, 'clear-pings', 'clear_requests');
      assert.equal(page().coordination.requests.length, 0);
      store.status('room', human, 'resume', 'active', 'Continue');
    });
    await t.test('research and pending requests survive reopening; a decision also finishes research', () => {
      say(human, '@claude record the plan.');
      store.activity('room', claude, 'researching', 'Checking the final detail.', 120);
      const reopened = new DiscussionStore(dbPath, { readonly: true });
      try { assert.equal(reopened.read('room', 0, 100).coordination.research?.name, 'Claude'); }
      finally { reopened.close(); }
      store.decide('room', claude, 'final-plan', 0, 'Measure first', '', page().next_after_id);
      assert.equal(page().coordination.research, null);
      assert.equal(page().coordination.requests.length, 0);
    });
  } finally {
    peer.close(); store.close();
    const target = resolve(temp);
    assert.ok(target.startsWith(resolve(tmpdir()) + sep) && target.includes('hotstep-coordination-'));
    rmSync(target, { recursive: true, force: true });
  }
});

test('legacy discussion databases remain readable and migrate without losing history', () => {
  const temp = mkdtempSync(join(tmpdir(), 'hotstep-coordination-'));
  const dbPath = join(temp, 'legacy.db');
  try {
    const original = new DiscussionStore(dbPath);
    let participant: string;
    let messageId: number;
    try {
      participant = original.join('legacy', 'Codex', 'Existing discussion').participant_id;
      messageId = original.post('legacy', participant, 'old-message', 'reply', 'Existing evidence').id;
    } finally { original.close(); }
    // Reproduce the previous schema in this disposable test database only.
    const raw = new Database(dbPath);
    try {
      raw.exec('DROP TABLE discussion_mentions; DROP TABLE discussion_requests; DROP TABLE discussion_research;');
    } finally { raw.close(); }
    const readonly = new DiscussionStore(dbPath, { readonly: true });
    try {
      const page = readonly.read('legacy', 0, 100);
      assert.equal(page.messages[0].id, messageId);
      assert.deepEqual(page.messages[0].mentions, []);
      assert.deepEqual(page.coordination, { research: null, requests: [] });
    } finally { readonly.close(); }
    const migrated = new DiscussionStore(dbPath);
    try {
      const human = randomUUID();
      migrated.joinViewer('legacy', human);
      migrated.post('legacy', human, 'new-ping', 'user_direction', '@codex check this');
      const page = migrated.read('legacy', 0, 100);
      assert.equal(page.messages[0].body, 'Existing evidence');
      assert.equal(page.coordination.requests[0].participant_id, participant);
    } finally { migrated.close(); }
  } finally {
    const target = resolve(temp);
    assert.ok(target.startsWith(resolve(tmpdir()) + sep) && target.includes('hotstep-coordination-'));
    rmSync(target, { recursive: true, force: true });
  }
});
