import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, sep } from 'node:path';
import test from 'node:test';
import Database from 'better-sqlite3';
import { DiscussionStore } from '../src/collaboration.js';

test('plan agreement closes only after all agents review the current discussion', async t => {
  const temp = mkdtempSync(join(tmpdir(), 'hotstep-consensus-'));
  const dbPath = join(temp, 'room.db');
  const store = new DiscussionStore(dbPath);
  const peer = new DiscussionStore(dbPath);
  const human = randomUUID();
  const codex = store.join('room', 'Codex', 'Agree on the plan').participant_id;
  const duplicate = store.join('room', ' codex ').participant_id;
  store.joinViewer('room', human);
  const page = () => store.read('room', 0, 1000);
  const agree = (id: string, request: string = randomUUID()) => store.agreePlan('room', id, request, page().discussion.revision, page().next_after_id);
  const steer = () => store.post('room', human, randomUUID(), 'user_direction', 'Please consider this concern.');
  try {
    await t.test('requires a recorded plan and at least two distinct agent names', () => {
      assert.throws(() => agree(codex), /recorded plan/);
      store.decide('room', codex, 'plan1', 0, 'First plan', '');
      assert.equal(page().consensus.agents[0].agreed, false);
      agree(codex);
      agree(duplicate);
      assert.equal(page().consensus.agents.length, 1);
      assert.equal(page().consensus.reached, false);
      assert.equal(page().discussion.status, 'active');
      assert.throws(() => store.post('room', codex, 'extra-reply', 'reply', 'Another turn'), /Wait for another/);
    });
    const claude = store.join('room', 'Claude').participant_id;
    await t.test('new steering clears votes and rejects stale or foreign agreements', () => {
      const stale = page().next_after_id;
      steer();
      assert.equal(page().consensus.agents.some(a => a.agreed), false);
      assert.throws(() => store.agreePlan('room', claude, 'stale', 1, stale), /Read all new/);
      assert.throws(() => store.agreePlan('room', claude, 'future', 2, page().next_after_id), /revision must match/);
      assert.throws(() => agree(human), /belong to agents/);
      assert.throws(() => agree(randomUUID()), /Unknown participant/);
    });
    await t.test('agreement respects research holds and requests without consuming a turn', () => {
      store.activity('room', claude, 'researching', 'Reviewing the final detail', 120);
      assert.throws(() => agree(codex), /researching/);
      agree(claude);
      assert.equal(page().coordination.research, null);
      store.post('room', human, 'ping', 'user_direction', '@codex check this');
      assert.throws(() => agree(claude), /requested from\s+codex/i);
      agree(duplicate);
      assert.equal(page().coordination.requests.length, 0);
    });
    await t.test('a new revision requires fresh agreement from both agents', () => {
      store.decide('room', claude, 'plan2', 1, 'Revised plan', '');
      assert.equal(page().consensus.agents.some(a => a.agreed), false);
      assert.throws(() => store.agreePlan('room', codex, 'old-plan', 1, page().next_after_id), /revision must match/);
      agree(claude);
      const readonly = new DiscussionStore(dbPath, { readonly: true });
      try { assert.deepEqual(readonly.read('room', 0, 1000).consensus, page().consensus); }
      finally { readonly.close(); }
    });
    await t.test('final agreement atomically closes the room, releases all waiters and retries safely', async () => {
      const cursor = page().next_after_id;
      const waits = [peer.wait('room', cursor, 2000, 50), store.wait('room', cursor, 2000, 50)];
      const last = agree(codex, 'final-agreement');
      assert.equal(last.discussion.status, 'closed');
      assert.equal(last.consensus.reached, true);
      for (const result of await Promise.all(waits)) {
        assert.equal(result.discussion.status, 'closed');
        assert.equal(result.timed_out, false);
        assert.equal(result.consensus.reached, true);
      }
      assert.equal(page().messages.filter(m => m.kind === 'status').length, 1);
      assert.deepEqual(agree(codex, 'final-agreement'), last);
      assert.throws(() => store.agreePlan('room', codex, 'final-agreement', 1, cursor), /different content/);
      assert.throws(() => store.post('room', claude, 'closed-post', 'reply', 'More'), /closed/);
      assert.throws(() => agree(claude), /closed/);
    });
    await t.test('human reopening resets votes and ending clears outstanding research and requests', async () => {
      store.status('room', human, 'reopen', 'active', 'Revisit this');
      store.monitor('room', codex);
      store.monitor('room', claude);
      assert.equal(page().consensus.reached, false);
      assert.equal(page().consensus.agents.some(a => a.agreed), false);
      store.post('room', human, 'new-ping', 'user_direction', '@claude review');
      store.activity('room', claude, 'researching', 'Reviewing', 120);
      const waiting = peer.wait('room', page().next_after_id, 2000, 50);
      store.status('room', human, 'end', 'closed', 'End Discussion');
      const result = await waiting;
      assert.equal(result.discussion.status, 'closed');
      assert.deepEqual(result.coordination, { research: null, requests: [] });
    });
    await t.test('older databases remain readable and gain agreement storage without losing history', () => {
      const before = page().messages;
      const raw = new Database(dbPath);
      try { raw.exec('DROP TABLE discussion_agreements'); } finally { raw.close(); }
      const readonly = new DiscussionStore(dbPath, { readonly: true });
      try { assert.equal(readonly.read('room', 0, 1000).consensus.reached, false); }
      finally { readonly.close(); }
      const migrated = new DiscussionStore(dbPath);
      try { assert.deepEqual(migrated.read('room', 0, 1000).messages, before); }
      finally { migrated.close(); }
    });
    await t.test('two agreements do not close a room with a third named agent', () => {
      const a = store.join('three', 'A', 'Three reviewers').participant_id;
      const b = store.join('three', 'B').participant_id;
      const c = store.join('three', 'C').participant_id;
      store.decide('three', a, 'plan', 0, 'Plan', '');
      const vote = (id: string) => store.agreePlan('three', id, 'agree', 1, store.read('three', 0, 100).next_after_id);
      vote(a);
      assert.equal(vote(b).discussion.status, 'active');
      assert.equal(vote(c).discussion.status, 'closed');
    });
  } finally {
    peer.close(); store.close();
    const target = resolve(temp);
    assert.ok(target.startsWith(resolve(tmpdir()) + sep) && target.includes('hotstep-consensus-'));
    rmSync(target, { recursive: true, force: true });
  }
});
