import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, sep } from 'node:path';
import test from 'node:test';
import Database from 'better-sqlite3';
import { DEFAULT_MAX_ROUNDS, DiscussionStore } from '../src/collaboration.js';

const HUMAN = '11111111-1111-4111-8111-111111111111';

function scratch() {
  const temp = mkdtempSync(join(tmpdir(), 'hotstep-positions-test-'));
  return {
    db: join(temp, 'positions.db'),
    cleanup() {
      const target = resolve(temp);
      assert.ok(target.startsWith(resolve(tmpdir()) + sep) && target.includes('hotstep-positions-test-'));
      rmSync(target, { recursive: true, force: true });
    },
  };
}

test('sealed positions stay hidden until every present agent has submitted', t => {
  const { db, cleanup } = scratch();
  const store = new DiscussionStore(db);
  t.after(() => { store.close(); cleanup(); });
  const room = 'blind';
  const a = store.join(room, 'Claude', 'Which crop regime?', { blind_positions: true, base_commit: 'abc123def456', role: 'engine/logic lead' });
  const b = store.join(room, 'Codex', undefined, { role: 'app/integration lead' });
  assert.equal(a.discussion.phase, 'positions');
  assert.equal(a.discussion.base_commit, 'abc123def456');
  assert.equal(a.discussion.max_rounds, DEFAULT_MAX_ROUNDS);
  assert.equal(a.role, 'engine/logic lead');
  assert.match(a.next_step, /submit your independent position/);
  assert.deepEqual(a.positions, { phase: 'positions', submitted: [], awaiting: ['Claude'], revealed: false });

  // Nothing but positions may enter the room while sealed. Research holds are allowed.
  assert.throws(() => store.post(room, a.participant_id, 'p1', 'proposal', 'Crop 800'), /Sealed positions phase/);
  assert.throws(() => store.decide(room, a.participant_id, 'd1', 0, 'Crop 800', ''), /Sealed positions phase/);
  store.activity(room, a.participant_id, 'researching', 'Reading the crop study', 60);
  store.activity(room, a.participant_id, 'idle', '', 60);
  assert.throws(() => store.submitPosition(room, HUMAN, 'h', 'Human position'), /Unknown participant/);

  const first = store.submitPosition(room, a.participant_id, 'pos-1', 'Crop 800, target 0.3. Evidence: blind study 2026-09-05.');
  assert.deepEqual(first, { phase: 'positions', submitted: ['Claude'], awaiting: ['Codex'], revealed: false, submitted_now: true });
  // The peer sees who has submitted but never the body.
  const sealed = store.read(room, 0, 100);
  assert.deepEqual(sealed.messages, []);
  assert.deepEqual(sealed.positions.submitted, ['Claude']);
  assert.deepEqual(sealed.positions.awaiting, ['Codex']);
  assert.equal(JSON.stringify(sealed).includes('blind study'), false);
  // One position per agent; an identical retry is idempotent.
  assert.throws(() => store.submitPosition(room, a.participant_id, 'pos-2', 'Changed my mind'), /already submitted/);
  assert.equal(store.submitPosition(room, a.participant_id, 'pos-1', 'Crop 800, target 0.3. Evidence: blind study 2026-09-05.').submitted_now, true);
  assert.throws(() => store.submitPosition(room, a.participant_id, 'pos-1', 'Different body'), /different content/);
  assert.throws(() => store.agreePlan(room, b.participant_id, 'agree', 1, sealed.next_after_id), /Read the current recorded plan/);

  const second = store.submitPosition(room, b.participant_id, 'pos-1', 'Crop 1500 for likeness; accept some bittiness.');
  assert.equal(second.revealed, true);
  assert.equal(second.phase, 'discussion');
  const revealed = store.read(room, 0, 100);
  assert.equal(revealed.discussion.phase, 'discussion');
  assert.deepEqual(revealed.messages.map(m => [m.kind, m.author]), [['coordination', 'Codex'], ['position', 'Claude'], ['position', 'Codex']]);
  assert.match(revealed.messages[0].body, /Positions revealed \(2: Claude, Codex\) because every present agent has submitted/);
  assert.equal(revealed.messages[1].body, 'Crop 800, target 0.3. Evidence: blind study 2026-09-05.');
  assert.equal(revealed.positions.revealed, true);
  assert.deepEqual(revealed.participants.map(p => [p.name, p.role]), [['Claude', 'engine/logic lead'], ['Codex', 'app/integration lead']]);
  // Revealed positions do not consume a turn: the last submitter may speak first.
  store.post(room, b.participant_id, 'c1', 'critique', 'Crop 800 loses the album arc; see the 09-03 verdict.');
  assert.throws(() => store.post(room, b.participant_id, 'c2', 'reply', 'And another thing'), /Wait for another/);
  assert.throws(() => store.submitPosition(room, b.participant_id, 'late', 'Late position'), /no sealed positions phase open/);
  // Retrying the original submission after the reveal still succeeds without a second copy.
  assert.equal(store.submitPosition(room, a.participant_id, 'pos-1', 'Crop 800, target 0.3. Evidence: blind study 2026-09-05.').revealed, true);
  assert.equal(store.read(room, 0, 100).messages.filter(m => m.kind === 'position').length, 2);
});

test('sealed phase closes the side channels and clears holds; a departed agent cannot keep the room sealed', t => {
  const { db, cleanup } = scratch();
  const store = new DiscussionStore(db);
  t.after(() => { store.close(); cleanup(); });
  const room = 'channels';
  const a = store.join(room, 'Claude', 'Test the side channels.', { blind_positions: true, base_commit: null });
  const b = store.join(room, 'Codex');
  // Status reasons and decline reasons are free text in the transcript: refused while sealed.
  assert.throws(() => store.status(room, a.participant_id, 's1', 'active', 'MY SEALED PLAN: crop 800'), /Sealed positions phase/);
  store.joinViewer(room, HUMAN);
  store.post(room, HUMAN, 'ping', 'user_direction', '@codex please answer first');
  assert.throws(() => store.declineRequest(room, b.participant_id, 'dec', 'my plan is crop 1500', store.read(room, 0, 100).next_after_id), /Sealed positions phase/);
  assert.deepEqual(store.read(room, 0, 100).messages.map(m => m.kind), ['user_direction']);
  store.clearCoordination(room, HUMAN, 'clear', 'clear_requests');
  // The human may still pause and resume while sealed.
  store.status(room, HUMAN, 'pause', 'paused', 'Hold on.');
  assert.throws(() => store.clearCoordination(room, HUMAN, 'reveal-paused', 'reveal_positions'), /paused/);
  store.status(room, HUMAN, 'resume', 'active', 'Go on.');
  // A research hold ends with the submission, and no hold survives the reveal.
  store.activity(room, a.participant_id, 'researching', 'Reading the study', 300);
  store.submitPosition(room, a.participant_id, 'pos', 'Crop 800.');
  assert.equal(store.read(room, 0, 100).coordination.research, null);
  store.activity(room, b.participant_id, 'researching', 'Reading too', 300);
  store.submitPosition(room, b.participant_id, 'pos', 'Crop 1500.');
  const revealed = store.read(room, 0, 100);
  assert.equal(revealed.discussion.phase, 'discussion');
  assert.equal(revealed.coordination.research, null);
  store.post(room, a.participant_id, 'first', 'critique', 'Speaking first after the reveal.', undefined, revealed.next_after_id);

  // Three agents: the third leaves without submitting, and the room opens on its departure.
  const x = store.join('three', 'Claude', 'Three agents.', { blind_positions: true, base_commit: null });
  const y = store.join('three', 'Codex');
  const z = store.join('three', 'Gemini');
  store.submitPosition('three', x.participant_id, 'pos', 'A');
  const waiting = store.submitPosition('three', y.participant_id, 'pos', 'B');
  assert.equal(waiting.revealed, false);
  assert.deepEqual(waiting.awaiting, ['Gemini']);
  store.leave('three', z.participant_id);
  const opened = store.read('three', 0, 100);
  assert.equal(opened.discussion.phase, 'discussion');
  assert.deepEqual(opened.messages.map(m => m.kind), ['coordination', 'position', 'position']);
});

test('the human can reveal early; rooms without the phase reject positions', t => {
  const { db, cleanup } = scratch();
  const store = new DiscussionStore(db);
  t.after(() => { store.close(); cleanup(); });
  store.createViewerDiscussion('early', 'Decide the outro fix.', HUMAN, 'create-1', { blind_positions: true, max_rounds: 3 });
  const a = store.join('early', 'Claude');
  store.join('early', 'Codex');
  assert.equal(store.room('early').max_rounds, 3);
  assert.throws(() => store.clearCoordination('early', HUMAN, 'reveal-0', 'reveal_positions'), /No positions have been submitted/);
  store.submitPosition('early', a.participant_id, 'pos', 'Train whole songs.');
  assert.throws(() => store.clearCoordination('early', a.participant_id, 'reveal-agent', 'reveal_positions'), /Only the human/);
  const event = store.clearCoordination('early', HUMAN, 'reveal-1', 'reveal_positions');
  assert.match(event.body, /Positions revealed \(1: Claude\) because the user opened them/);
  assert.equal(store.clearCoordination('early', HUMAN, 'reveal-1', 'reveal_positions').id, event.id);
  assert.throws(() => store.clearCoordination('early', HUMAN, 'reveal-2', 'reveal_positions'), /already revealed/);
  const page = store.read('early', 0, 100);
  assert.deepEqual(page.messages.map(m => m.kind), ['user_direction', 'coordination', 'position']);
  assert.equal(page.discussion.phase, 'discussion');

  const plain = store.join('plain', 'Claude', 'Ordinary room');
  assert.equal(plain.discussion.phase, 'discussion');
  assert.equal(plain.positions, undefined);
  assert.throws(() => store.submitPosition('plain', plain.participant_id, 'pos', 'Nope'), /no sealed positions phase open/);
  assert.throws(() => store.join('bad', 'Claude', 'Bad rounds', { max_rounds: 9 }), /max_rounds/);
});

test('open items block consensus and the round cap hands the room to the user', t => {
  const { db, cleanup } = scratch();
  const store = new DiscussionStore(db);
  t.after(() => { store.close(); cleanup(); });
  const room = 'rounds';
  const a = store.join(room, 'Claude', 'Settle the depth question.', { base_commit: null });
  const b = store.join(room, 'Codex');
  assert.equal(store.room(room).base_commit, null);
  store.post(room, a.participant_id, 'p1', 'proposal', 'Export at 0.1.');
  assert.throws(() => store.decide(room, b.participant_id, 'bad-owner', 0, 'Plan', '', undefined, [{ issue: 'x', owner: 'Nobody' }]), /not You or a joined agent name/);
  const r1 = store.decide(room, b.participant_id, 'd1', 0, 'Export at 0.1 with crop 800.', '', undefined, [
    { issue: 'Does crop 800 hold up on long albums?', owner: 'claude' },
  ]);
  assert.deepEqual(r1.open_items, [{ issue: 'Does crop 800 hold up on long albums?', owner: 'Claude' }]);
  assert.equal(r1.paused, false);
  let page = store.read(room, 0, 100);
  assert.deepEqual(page.decision?.open_items, r1.open_items);
  assert.throws(() => store.agreePlan(room, a.participant_id, 'agree-1', 1, page.next_after_id), /Consensus requires no open items/);
  assert.throws(() => store.agreePlan(room, b.participant_id, 'agree-1', 1, page.next_after_id), /Consensus requires no open items/);

  // Revision 2 (the cap) still has an item: the room pauses itself for the user.
  store.post(room, a.participant_id, 'r1', 'reply', 'Long albums untested; I still object.');
  const r2 = store.decide(room, b.participant_id, 'd2', 1, 'Same plan.', 'Long-album behaviour is unresolved.');
  assert.deepEqual(r2.open_items, [{ issue: 'Long-album behaviour is unresolved.', owner: 'You' }]);
  assert.equal(r2.paused, true);
  page = store.read(room, 0, 100);
  assert.equal(page.discussion.status, 'paused');
  assert.match(page.messages.at(-1)!.body, /Round cap: revision 2 of 2 still lists 1 open item/);
  assert.throws(() => store.post(room, a.participant_id, 'r2', 'reply', 'One more round'), /paused/);
  assert.equal(store.decide(room, b.participant_id, 'd2', 1, 'Same plan.', 'Long-album behaviour is unresolved.').paused, true);

  // The user rules, resumes, and the next revision closes the item.
  store.joinViewer(room, HUMAN);
  store.status(room, HUMAN, 'resume', 'active', 'Ruling: ship crop 800; long albums get their own test later.');
  store.post(room, HUMAN, 'ruling', 'user_direction', 'Ship crop 800.');
  const r3 = store.decide(room, a.participant_id, 'd3', 2, 'Export at 0.1 with crop 800. Long albums: separate test per user ruling #6.', 'None outstanding. Codex withdrew the long-album objection after the ruling.');
  assert.deepEqual(r3.open_items, []);
  // Legacy free text: a none-clause must lead; prose that merely starts with "None of" is a real disagreement.
  assert.deepEqual(store.decide('rounds-text', store.join('rounds-text', 'Claude', 'Text mapping', { base_commit: null }).participant_id, 'd', 0, 'Plan', 'None of the crop questions are settled.').open_items,
    [{ issue: 'None of the crop questions are settled.', owner: 'You' }]);
  assert.equal(r3.paused, false);
  page = store.read(room, 0, 100);
  store.agreePlan(room, a.participant_id, 'agree-3', 3, page.next_after_id);
  page = store.read(room, 0, 100);
  const closed = store.agreePlan(room, b.participant_id, 'agree-3', 3, page.next_after_id);
  assert.equal(closed.discussion.status, 'closed');
  assert.equal(closed.consensus.reached, true);

  const plan = store.exportPlan(room, { transcript: true });
  assert.match(plan.markdown, /Base commit: not recorded/);
  assert.match(plan.markdown, /Revision: 3 of at most 2/);
  assert.match(plan.markdown, /## Open items\n\nNone\./);
  assert.match(plan.markdown, /## Transcript/);
  assert.match(plan.markdown, /Recorded plan revision 3 with 0 open item\(s\)\./);
  assert.match(plan.markdown, /Ship crop 800\./);
  const bare = store.exportPlan('rounds');
  assert.doesNotMatch(bare.markdown, /## Transcript/);
});

test('an older database gains the new columns and reads with defaults', t => {
  const { db, cleanup } = scratch();
  const closers: (() => void)[] = [];
  // Hooks run in registration order: every connection must close before the directory goes.
  t.after(() => { for (const close of closers.reverse()) close(); cleanup(); });
  const legacy = new Database(db);
  // The exact tables and positional inserts the previous code shipped with.
  legacy.exec(`
    CREATE TABLE discussions (id TEXT PRIMARY KEY, brief TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active', revision INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
    CREATE TABLE participants (id TEXT PRIMARY KEY, room TEXT NOT NULL REFERENCES discussions(id), name TEXT NOT NULL, joined_at TEXT NOT NULL);
    CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, room TEXT NOT NULL, participant_id TEXT NOT NULL, author TEXT NOT NULL, kind TEXT NOT NULL, body TEXT NOT NULL, reply_to INTEGER, request_id TEXT NOT NULL, created_at TEXT NOT NULL, UNIQUE(participant_id, request_id));
    CREATE TABLE decisions (room TEXT NOT NULL, revision INTEGER NOT NULL, message_id INTEGER NOT NULL, plan TEXT NOT NULL, disagreements TEXT NOT NULL, PRIMARY KEY(room, revision));
    INSERT INTO discussions (id, brief, status, revision, created_at) VALUES ('old', 'Legacy brief', 'closed', 1, '2026-09-05T20:12:43.262Z');
    INSERT INTO participants VALUES ('${HUMAN}', 'old', 'Codex', '2026-09-05T20:12:44.000Z');
    INSERT INTO messages (room, participant_id, author, kind, body, reply_to, request_id, created_at) VALUES ('old', '${HUMAN}', 'Codex', 'decision', '{"plan":"Old plan","disagreements":"Old text","expected_revision":0}', NULL, 'd1', '2026-09-05T20:13:00.000Z');
    INSERT INTO decisions VALUES ('old', 1, 1, 'Old plan', 'Old text');
  `);
  legacy.close();
  const store = new DiscussionStore(db);
  closers.push(() => store.close());
  const room = store.room('old');
  assert.equal(room.phase, 'discussion');
  assert.equal(room.base_commit, null);
  assert.equal(room.max_rounds, DEFAULT_MAX_ROUNDS);
  assert.deepEqual(store.latestDecision('old')?.open_items, []);
  assert.equal(store.list(10)[0].phase, 'discussion');
  const page = store.read('old', 0, 100);
  assert.deepEqual(page.positions, { phase: 'discussion', submitted: [], awaiting: [], revealed: false });
  // A second writer on the migrated file is a no-op, and readers see the columns.
  new DiscussionStore(db).close();
  const reader = new DiscussionStore(db, { readonly: true });
  assert.equal(reader.room('old').max_rounds, DEFAULT_MAX_ROUNDS);
  reader.close();
  const created = store.join('fresh', 'Claude', 'New room after migration', { blind_positions: true });
  assert.equal(created.discussion.phase, 'positions');
  assert.ok(created.discussion.base_commit === null || /^[0-9a-f]{7,40}(-dirty)?$/.test(created.discussion.base_commit));
  // A process still running the previous code inserts positionally into participants and decisions.
  const old = new Database(db);
  closers.push(() => old.close());
  old.prepare('INSERT INTO participants VALUES (?, ?, ?, ?)').run('22222222-2222-4222-8222-222222222222', 'old', 'Claude', '2026-09-11T10:00:00.000Z');
  old.prepare('INSERT INTO decisions VALUES (?, ?, ?, ?, ?)').run('old', 2, 1, 'Plan from the old process', '');
  assert.equal(store.latestDecision('old')?.plan, 'Plan from the old process');
  assert.deepEqual(store.latestDecision('old')?.open_items, []);
});
