import type Database from 'better-sqlite3';

export const PRESENCE_TTL_MS = 90_000;
export const PRESENCE_SCHEMA = `
  CREATE TABLE IF NOT EXISTS discussion_presence (
    participant_id TEXT PRIMARY KEY REFERENCES participants(id),
    room TEXT NOT NULL REFERENCES discussions(id),
    owner TEXT NOT NULL, expires_at INTEGER NOT NULL
  );
`;

type Participant = { id: string; name: string; joined_at: string };

export class DiscussionPresence {
  constructor(private db: Database.Database, private owner = '') {}
  private available() {
    return Boolean(this.db.prepare("SELECT 1 FROM sqlite_master WHERE name = 'discussion_presence'").get());
  }
  list(room: string) {
    // The viewer represents the human. Old joins without a lease are history,
    // never evidence that an agent is still monitoring the discussion.
    const humans = this.db.prepare("SELECT id, name, joined_at FROM participants WHERE room = ? AND name = 'You' ORDER BY joined_at LIMIT 1")
      .all(room) as Participant[];
    const agents = this.available() ? this.db.prepare(`SELECT p.id, p.name, p.joined_at
      FROM participants p JOIN discussion_presence live ON live.participant_id = p.id
      JOIN discussions d ON d.id = p.room
      WHERE p.room = ? AND p.name != 'You' AND live.expires_at > ? AND d.status = 'active'
      ORDER BY live.expires_at DESC, p.rowid DESC`).all(room, Date.now()) as Participant[] : [];
    const unique = new Map<string, Participant>();
    for (const p of [...humans, ...agents]) {
      const key = p.name.trim().toLowerCase();
      if (!unique.has(key)) unique.set(key, { ...p, name: p.name.trim() });
    }
    return [...unique.values()].sort((a, b) => a.name === 'You' ? -1 : b.name === 'You' ? 1 : a.name.localeCompare(b.name));
  }
  touch(room: string, participant: string) {
    const p = this.db.prepare('SELECT name FROM participants WHERE room = ? AND id = ?').get(room, participant) as { name: string } | undefined;
    if (!p) throw new Error('Unknown participant for this discussion. Use the participant_id returned by join.');
    if (p.name === 'You') return;
    const d = this.db.prepare('SELECT status FROM discussions WHERE id = ?').get(room) as { status: string };
    if (d.status !== 'active') return;
    const hold = this.db.prepare('SELECT expires_at FROM discussion_research WHERE room = ? AND participant_id = ?')
      .get(room, participant) as { expires_at: number } | undefined;
    const expires = Math.max(Date.now() + PRESENCE_TTL_MS, hold?.expires_at ?? 0);
    this.db.prepare(`INSERT INTO discussion_presence VALUES (?, ?, ?, ?)
      ON CONFLICT(participant_id) DO UPDATE SET owner = excluded.owner, expires_at = excluded.expires_at`)
      .run(participant, room, this.owner, expires);
  }
  leave(room: string, participant: string) {
    this.db.prepare('DELETE FROM discussion_presence WHERE room = ? AND participant_id = ?').run(room, participant);
  }
  clear(room: string) {
    this.db.prepare('DELETE FROM discussion_presence WHERE room = ?').run(room);
  }
  disconnect() {
    if (!this.available()) return;
    // An older connection closing must not remove a newer connection's lease.
    this.db.prepare(`DELETE FROM discussion_research WHERE participant_id IN
      (SELECT participant_id FROM discussion_presence WHERE owner = ?)`).run(this.owner);
    this.db.prepare('DELETE FROM discussion_presence WHERE owner = ?').run(this.owner);
  }
}
