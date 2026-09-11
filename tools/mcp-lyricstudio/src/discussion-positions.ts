import type Database from 'better-sqlite3';

// Sealed positions: each agent's independent plan, written before it can see
// the other agent's. Rows live here, not in messages, until the reveal copies
// them into the transcript in submission order. Nothing hidden is ever paged.
export const POSITIONS_SCHEMA = `
  CREATE TABLE IF NOT EXISTS discussion_positions (
    room TEXT NOT NULL REFERENCES discussions(id), agent TEXT NOT NULL,
    participant_id TEXT NOT NULL REFERENCES participants(id), name TEXT NOT NULL,
    body TEXT NOT NULL, request_id TEXT NOT NULL, created_at TEXT NOT NULL,
    message_id INTEGER REFERENCES messages(id),
    PRIMARY KEY(room, agent)
  );
`;

export const MAX_POSITION_CHARS = 24000;

export type Position = { room: string; agent: string; participant_id: string; name: string; body: string; request_id: string; created_at: string; message_id: number | null };

// Called within the store's read or IMMEDIATE write transaction.
export class DiscussionPositions {
  constructor(private db: Database.Database) {}
  private available() {
    return Boolean(this.db.prepare("SELECT 1 FROM sqlite_master WHERE name = 'discussion_positions'").get());
  }
  list(room: string): Position[] {
    if (!this.available()) return [];
    return this.db.prepare('SELECT * FROM discussion_positions WHERE room = ? ORDER BY created_at, rowid').all(room) as Position[];
  }
  get(room: string, agent: string): Position | undefined {
    if (!this.available()) return undefined;
    return this.db.prepare('SELECT * FROM discussion_positions WHERE room = ? AND agent = ?').get(room, agent) as Position | undefined;
  }
  previous(participant: string, request: string): Position | undefined {
    if (!this.available()) return undefined;
    return this.db.prepare('SELECT * FROM discussion_positions WHERE participant_id = ? AND request_id = ?').get(participant, request) as Position | undefined;
  }
  submit(room: string, agent: string, participant: string, name: string, body: string, request: string) {
    this.db.prepare('INSERT INTO discussion_positions (room, agent, participant_id, name, body, request_id, created_at, message_id) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)')
      .run(room, agent, participant, name, body, request, new Date().toISOString());
  }
  markRevealed(room: string, agent: string, message: number) {
    this.db.prepare('UPDATE discussion_positions SET message_id = ? WHERE room = ? AND agent = ?').run(message, room, agent);
  }
  // Who has submitted and which present agents are still awaited. Bodies are
  // never part of the snapshot: they reach the transcript only at the reveal.
  snapshot(room: string, phase: 'positions' | 'discussion', roster: { name: string }[]) {
    const rows = this.list(room);
    const submitted = new Set(rows.map(r => r.agent));
    return {
      phase,
      submitted: rows.map(r => r.name),
      awaiting: phase === 'positions' ? roster.map(p => p.name).filter(name => !submitted.has(name.trim().toLowerCase())) : [],
      revealed: phase === 'discussion' && rows.length > 0,
    };
  }
}
