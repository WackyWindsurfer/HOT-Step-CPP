import type Database from 'better-sqlite3';

export const COORDINATION_SCHEMA = `
  CREATE TABLE IF NOT EXISTS discussion_research (
    room TEXT PRIMARY KEY REFERENCES discussions(id),
    participant_id TEXT NOT NULL REFERENCES participants(id),
    reason TEXT NOT NULL, expires_at INTEGER NOT NULL
  );
  CREATE TABLE IF NOT EXISTS discussion_requests (
    room TEXT NOT NULL REFERENCES discussions(id),
    participant_id TEXT NOT NULL REFERENCES participants(id),
    message_id INTEGER NOT NULL REFERENCES messages(id),
    PRIMARY KEY(room, participant_id)
  );
  CREATE TABLE IF NOT EXISTS discussion_mentions (
    message_id INTEGER NOT NULL REFERENCES messages(id),
    participant_id TEXT NOT NULL REFERENCES participants(id),
    PRIMARY KEY(message_id, participant_id)
  );
`;

export function mentionHandle(name: string) {
  return name.trim().toLowerCase().replace(/\s+/g, '-');
}

type Research = { participant_id: string; name: string; reason: string; expires_at: number };
type ResponseRequest = { participant_id: string; name: string; message_id: number };

// All mutations are called inside the store's IMMEDIATE write transaction.
export class DiscussionCoordination {
  constructor(private db: Database.Database) {}
  private available() {
    // An old database can still be viewed read-only before its first new writer.
    return Boolean(this.db.prepare("SELECT 1 FROM sqlite_master WHERE name = 'discussion_research'").get());
  }
  snapshot(room: string) {
    if (!this.available()) return { research: null, requests: [] as ResponseRequest[] };
    const research = (this.db.prepare(`SELECT r.participant_id, p.name, r.reason, r.expires_at
      FROM discussion_research r JOIN participants p ON p.id = r.participant_id
      WHERE r.room = ? AND r.expires_at > ?`).get(room, Date.now()) as Research | undefined) ?? null;
    const requests = this.db.prepare(`SELECT r.participant_id, p.name, r.message_id
      FROM discussion_requests r JOIN participants p ON p.id = r.participant_id
      WHERE r.room = ? ORDER BY r.message_id, r.participant_id`).all(room) as ResponseRequest[];
    return { research, requests };
  }
  mentions(message: number) {
    if (!this.available()) return [];
    return this.db.prepare(`SELECT p.id AS participant_id, p.name FROM discussion_mentions m
      JOIN participants p ON p.id = m.participant_id WHERE m.message_id = ? ORDER BY p.name`).all(message) as { participant_id: string; name: string }[];
  }
  enqueue(room: string, sender: string, message: number, body: string) {
    const roster = this.db.prepare("SELECT id, name FROM participants WHERE room = ? AND name != 'You' ORDER BY joined_at DESC, rowid DESC").all(room) as { id: string; name: string }[];
    const handles = new Map<string, string>();
    for (const p of roster) if (!handles.has(mentionHandle(p.name))) handles.set(mentionHandle(p.name), p.id);
    // Ignore fenced/inline code and email addresses. Unknown names stay plain text.
    const prose = body.replace(/```[\s\S]*?```|`[^`\n]*`/g, '');
    const targets = new Set<string>();
    for (const match of prose.matchAll(/(?:^|[\s(])@([a-zA-Z0-9][a-zA-Z0-9._-]*)/g)) {
      const target = handles.get(match[1].toLowerCase().replace(/[.]+$/, ''));
      if (target && target !== sender) targets.add(target);
    }
    for (const target of targets) {
      this.db.prepare('INSERT INTO discussion_mentions VALUES (?, ?)').run(message, target);
      this.db.prepare(`INSERT INTO discussion_requests VALUES (?, ?, ?)
        ON CONFLICT(room, participant_id) DO UPDATE SET message_id = excluded.message_id`).run(room, target, message);
    }
  }
  beforeContribution(room: string, participant: string, readAfter?: number) {
    const { research, requests } = this.snapshot(room);
    if (research && research.participant_id !== participant) {
      throw new Error(`${research.name} is researching: ${research.reason}. Wait until the hold is released or expires.`);
    }
    if (research) {
      const latest = this.db.prepare('SELECT MAX(id) AS id FROM messages WHERE room = ?').get(room) as { id: number | null };
      if (readAfter === undefined || readAfter !== (latest.id ?? 0)) {
        throw new Error('Read all new room messages before answering from research; pass the returned next_after_id as read_after_id.');
      }
    } else if (requests.length && !requests.some(r => r.participant_id === participant)) {
      throw new Error(`A reply is requested from ${requests.map(r => r.name).join(', ')}. Wait for them or ask the human to clear the request.`);
    }
  }
  research(room: string, participant: string, reason: string, seconds: number) {
    const current = this.snapshot(room).research;
    if (current && current.participant_id !== participant) throw new Error(`${current.name} already holds the research turn.`);
    this.db.prepare(`INSERT INTO discussion_research VALUES (?, ?, ?, ?)
      ON CONFLICT(room) DO UPDATE SET participant_id = excluded.participant_id, reason = excluded.reason, expires_at = excluded.expires_at`)
      .run(room, participant, reason, Date.now() + seconds * 1000);
    return this.snapshot(room);
  }
  release(room: string, participant: string, human = false) {
    if (human) this.db.prepare('DELETE FROM discussion_research WHERE room = ?').run(room);
    else this.db.prepare('DELETE FROM discussion_research WHERE room = ? AND participant_id = ?').run(room, participant);
  }
  answered(room: string, participant: string) {
    this.release(room, participant);
    this.db.prepare('DELETE FROM discussion_requests WHERE room = ? AND participant_id = ?').run(room, participant);
  }
  decline(room: string, participant: string, readAfter: number) {
    const request = this.snapshot(room).requests.find(r => r.participant_id === participant);
    if (!request) throw new Error('No pending reply request for this participant.');
    const latest = this.db.prepare('SELECT MAX(id) AS id FROM messages WHERE room = ?').get(room) as { id: number };
    if (readAfter !== latest.id) throw new Error('Read the latest ping and all subsequent messages before declining it.');
    this.answered(room, participant);
  }
  clearRequests(room: string) {
    this.db.prepare('DELETE FROM discussion_requests WHERE room = ?').run(room);
  }
}
