import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, sep } from 'node:path';
import test from 'node:test';
import Database from 'better-sqlite3';
import { DiscussionStore } from '../src/collaboration.js';

test('presence tracks active room monitoring without deleting transcript identities', async t => {
  const temp = mkdtempSync(join(tmpdir(), 'hotstep-presence-'));
  const path = join(temp, 'room.db');
  const store = new DiscussionStore(path);
  const peer = new DiscussionStore(path);
  const raw = new Database(path);
  const human = randomUUID();
  const codex = store.join('room', 'Codex', 'Live presence').participant_id;
  const claude = peer.join('room', 'Claude').participant_id;
  store.joinViewer('room', human);
  const page = () => store.read('room', 0, 100);
  const expire = (id: string) => raw.prepare('UPDATE discussion_presence SET expires_at = ? WHERE participant_id = ?').run(Date.now() - 1, id);
  const names = () => page().participants.map(p => p.name);
  try {
    await t.test('repeat joins reuse one identity and normalize whitespace and case', () => {
      assert.equal(store.join('room', ' codex ').participant_id, codex);
      assert.equal(store.join('room', 'CODEX').participant_id, codex);
      assert.deepEqual(names(), ['You', 'Claude', 'Codex']);
      assert.equal((raw.prepare("SELECT COUNT(*) AS n FROM participants WHERE room = 'room' AND name != 'You'").get() as { n: number }).n, 2);
    });
    const message = store.post('room', codex, 'evidence', 'reply', 'Preserve this evidence');
    await t.test('old duplicate identities remain attributable but are not present', () => {
      raw.prepare('INSERT INTO participants VALUES (?, ?, ?, ?)').run(randomUUID(), 'room', 'Codex', '2000-01-01');
      assert.deepEqual(names(), ['You', 'Claude', 'Codex']);
    });
    await t.test('expired agents disappear on read and viewer polling never renews them', () => {
      expire(codex);
      const viewer = new DiscussionStore(path, { readonly: true });
      try {
        assert.deepEqual(viewer.read('room', 0, 100).participants.map(p => p.name), ['You', 'Claude']);
        assert.deepEqual(viewer.read('room', 0, 100).participants.map(p => p.name), ['You', 'Claude']);
      } finally { viewer.close(); }
      assert.equal(page().messages.find(m => m.id === message.id)?.author, 'Codex');
      store.monitor('room', codex);
      assert.ok(names().includes('Codex'));
      assert.throws(() => store.monitor('room', randomUUID()), /Unknown participant/);
    });
    await t.test('research extends presence through the hold, and explicit leave releases it', () => {
      peer.activity('room', claude, 'researching', 'Long investigation', 300);
      const hold = page().coordination.research!;
      const live = raw.prepare('SELECT expires_at FROM discussion_presence WHERE participant_id = ?').get(claude) as { expires_at: number };
      assert.ok(live.expires_at >= hold.expires_at);
      peer.leave('room', claude);
      assert.equal(page().coordination.research, null);
      assert.deepEqual(names(), ['You', 'Codex']);
      peer.leave('room', claude);
      assert.equal(peer.join('room', 'Claude').participant_id, claude);
    });
    await t.test('presence changes wake waiters and cancelled waits remove monitoring presence', async () => {
      const waiting = store.wait('room', page().next_after_id, 1000, 50);
      peer.leave('room', claude);
      assert.equal((await waiting).timed_out, false);
      const controller = new AbortController();
      const cancelled = peer.wait('room', page().next_after_id, 1000, 50, controller.signal, claude);
      const rejection = assert.rejects(cancelled, /abort/i);
      controller.abort();
      await rejection;
      assert.deepEqual(names(), ['You', 'Codex']);
      await peer.wait('room', page().next_after_id, 0, 50, undefined, claude);
      assert.ok(names().includes('Claude'));
    });
    await t.test('disconnect only removes leases owned by that connection', () => {
      const oldConnection = new DiscussionStore(path);
      oldConnection.join('room', 'Codex');
      store.monitor('room', codex);
      oldConnection.close();
      assert.ok(names().includes('Codex'));
      const departing = new DiscussionStore(path);
      departing.join('room', 'Reviewer');
      assert.ok(names().includes('Reviewer'));
      departing.close();
      assert.ok(!names().includes('Reviewer'));
    });
    await t.test('consensus uses present agents and remains recorded after they disappear', () => {
      peer.decide('room', claude, 'plan', 0, 'A solid plan', '');
      const absent = store.join('room', 'Offline reviewer').participant_id;
      expire(absent);
      store.agreePlan('room', codex, 'agree', 1, page().next_after_id);
      const result = peer.agreePlan('room', claude, 'agree', 1, page().next_after_id);
      assert.equal(result.discussion.status, 'closed');
      assert.equal(result.consensus.reached, true);
      assert.deepEqual(names(), ['You']);
      store.join('room', 'Late reader');
      assert.equal(page().consensus.reached, true);
      assert.deepEqual(names(), ['You']);
      store.status('room', human, 'reopen', 'active', 'Continue');
      assert.deepEqual(names(), ['You']);
      assert.equal(page().consensus.reached, false);
    });
    await t.test('legacy databases show no historical agents as online and migrate safely', () => {
      raw.exec('DROP TABLE discussion_presence');
      const viewer = new DiscussionStore(path, { readonly: true });
      try { assert.deepEqual(viewer.read('room', 0, 100).participants.map(p => p.name), ['You']); }
      finally { viewer.close(); }
      const migrated = new DiscussionStore(path);
      try {
        assert.equal(migrated.join('room', 'Codex').participant_id, codex);
        assert.equal(migrated.read('room', 0, 100).messages[0].id, message.id);
      } finally { migrated.close(); }
    });
  } finally {
    raw.close(); peer.close(); store.close();
    const target = resolve(temp);
    assert.ok(target.startsWith(resolve(tmpdir()) + sep) && target.includes('hotstep-presence-'));
    rmSync(target, { recursive: true, force: true });
  }
});
