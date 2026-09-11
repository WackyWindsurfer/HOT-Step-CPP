import type Database from 'better-sqlite3';
import { DiscussionPresence } from './discussion-presence.js';

export const CONSENSUS_SCHEMA = `
  CREATE TABLE IF NOT EXISTS discussion_agreements (
    room TEXT NOT NULL REFERENCES discussions(id), revision INTEGER NOT NULL,
    agent TEXT NOT NULL, participant_id TEXT NOT NULL REFERENCES participants(id),
    message_id INTEGER NOT NULL REFERENCES messages(id),
    PRIMARY KEY(room, agent)
  );
  CREATE TABLE IF NOT EXISTS discussion_consensus_results (
    room TEXT PRIMARY KEY REFERENCES discussions(id), revision INTEGER NOT NULL, agents TEXT NOT NULL
  );
`;

// Called within the store's read or IMMEDIATE write transaction.
export class DiscussionConsensus {
  constructor(private db: Database.Database) {}
  private available() {
    return Boolean(this.db.prepare("SELECT 1 FROM sqlite_master WHERE name = 'discussion_agreements'").get());
  }
  snapshot(room: string, revision: number, excluded: Set<string> = new Set()) {
    const status = (this.db.prepare('SELECT status FROM discussions WHERE id = ?').get(room) as { status: string }).status;
    const hasResults = this.db.prepare("SELECT 1 FROM sqlite_master WHERE name = 'discussion_consensus_results'").get();
    const completed = hasResults ? this.db.prepare('SELECT agents FROM discussion_consensus_results WHERE room = ? AND revision = ?')
      .get(room, revision) as { agents: string } | undefined : undefined;
    if (status === 'closed' && completed) return { revision, reached: true, minimum_agents: 2, agents: JSON.parse(completed.agents) as { name: string; agreed: boolean }[] };
    // Keep consensus on old closed rooms readable after presence expires.
    const roster = (status === 'closed'
      ? this.db.prepare("SELECT id, name FROM participants WHERE room = ? AND name != 'You' ORDER BY joined_at, rowid").all(room) as { id: string; name: string }[]
      : new DiscussionPresence(this.db).list(room).filter(p => p.name !== 'You')).filter(p => !excluded.has(p.id));
    const agents = new Map(roster.map(p => [p.name.trim().toLowerCase(), p.name]));
    const votes = this.available() ? this.db.prepare('SELECT agent FROM discussion_agreements WHERE room = ? AND revision = ?')
      .all(room, revision) as { agent: string }[] : [];
    const agreed = new Set(votes.map(v => v.agent));
    return {
      revision,
      reached: status === 'closed' && revision > 0 && agents.size >= 2 && [...agents.keys()].every(name => agreed.has(name)),
      minimum_agents: 2,
      agents: [...agents].map(([key, name]) => ({ name, agreed: agreed.has(key) })),
    };
  }
  agree(room: string, revision: number, name: string, participant: string, message: number) {
    this.db.prepare(`INSERT INTO discussion_agreements VALUES (?, ?, ?, ?, ?)
      ON CONFLICT(room, agent) DO UPDATE SET revision = excluded.revision,
      participant_id = excluded.participant_id, message_id = excluded.message_id`)
      .run(room, revision, name.trim().toLowerCase(), participant, message);
  }
  clear(room: string) {
    this.db.prepare('DELETE FROM discussion_agreements WHERE room = ?').run(room);
    this.db.prepare('DELETE FROM discussion_consensus_results WHERE room = ?').run(room);
  }
  complete(room: string, revision: number, agents: { name: string; agreed: boolean }[]) {
    this.db.prepare('INSERT OR REPLACE INTO discussion_consensus_results VALUES (?, ?, ?)').run(room, revision, JSON.stringify(agents));
  }
}
