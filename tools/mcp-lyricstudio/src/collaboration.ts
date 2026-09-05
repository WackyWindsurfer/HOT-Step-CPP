// Shared discussion storage for independent stdio MCP processes.
// No imports from the app and no connection to hotstep.db.
import Database from 'better-sqlite3';
import { mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';
import { setTimeout as delay } from 'node:timers/promises';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';

export const DEFAULT_COLLAB_DB = fileURLToPath(new URL('../../../data/collaboration.db', import.meta.url));

export const DISCUSSION_PROTOCOL = `You are participating as the current chat agent, not launching another model.
Read the brief and transcript before replying. Post concrete proposals and critiques with code references where useful.
Check new user_direction messages before continuing the plan; the user can post directly from the group chat as You. Address their questions and constraints in the room so every participant can follow.
Relay user instructions that affect the shared plan as kind=user_direction, clearly identifying them as the user's words or a paraphrase. Never invent user approval.
Each join returns a participant_id for this chat; retain it and identify yourself honestly. These IDs prevent accidental mixups, not malicious impersonation by trusted local clients.
After reading a page, retain next_after_id. If has_more is true, read the next page before replying. Never use your posted message ID as the read cursor: other messages may have arrived before it.
Use collab_wait_for_message with that cursor between responses. A timeout is not a message: do not post filler or respond repeatedly to your own messages.
Stop waiting after 3 consecutive timeouts, at the user's deadline, or after 8 substantive replies from you, whichever comes first. Summarize remaining questions in your chat.
Pause or close the room when asked; all participants must stop discussion work when its status is paused or closed. Resume only on user direction.
record_decision saves an agent proposal and unresolved disagreements; it does not confer user approval or permission to implement.
MCP does not automatically wake a chat after its turn ends. The user must start or resume participation in each chat.
Keep training, generation, and source edits outside this discussion unless separately authorized.`;

type Room = { id: string; brief: string; status: 'active' | 'paused' | 'closed'; revision: number; created_at: string };
type Message = { id: number; room: string; participant_id: string; author: string; kind: string; body: string; reply_to: number | null; request_id: string; created_at: string };

export class DiscussionStore {
  private db: Database.Database;

  constructor(dbPath: string, options: { readonly?: boolean } = {}) {
    if (!options.readonly) mkdirSync(dirname(resolve(dbPath)), { recursive: true });
    this.db = new Database(dbPath, { timeout: 5000, readonly: options.readonly ?? false, fileMustExist: options.readonly ?? false });
    if (options.readonly) return;
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('foreign_keys = ON');
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS discussions (
        id TEXT PRIMARY KEY, brief TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','paused','closed')),
        revision INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS participants (
        id TEXT PRIMARY KEY, room TEXT NOT NULL REFERENCES discussions(id),
        name TEXT NOT NULL, joined_at TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT, room TEXT NOT NULL REFERENCES discussions(id),
        participant_id TEXT NOT NULL REFERENCES participants(id), author TEXT NOT NULL,
        kind TEXT NOT NULL, body TEXT NOT NULL, reply_to INTEGER REFERENCES messages(id),
        request_id TEXT NOT NULL, created_at TEXT NOT NULL,
        UNIQUE(participant_id, request_id)
      );
      CREATE INDEX IF NOT EXISTS messages_room_id ON messages(room, id);
      CREATE TABLE IF NOT EXISTS decisions (
        room TEXT NOT NULL REFERENCES discussions(id), revision INTEGER NOT NULL,
        message_id INTEGER NOT NULL REFERENCES messages(id), plan TEXT NOT NULL,
        disagreements TEXT NOT NULL, PRIMARY KEY(room, revision)
      );
    `);
  }

  close() { this.db.close(); }
  room(id: string): Room {
    const room = this.db.prepare('SELECT * FROM discussions WHERE id = ?').get(id) as Room | undefined;
    if (!room) throw new Error(`Unknown discussion: ${id}. Join it first.`);
    return room;
  }
  private participant(room: string, id: string) {
    const participant = this.db.prepare('SELECT name FROM participants WHERE room = ? AND id = ?').get(room, id) as { name: string } | undefined;
    if (!participant) throw new Error('Unknown participant for this discussion. Use the participant_id returned by join.');
    return participant;
  }
  private active(room: string) {
    const current = this.room(room);
    if (current.status !== 'active') throw new Error(`Discussion is ${current.status}. Resume only on user direction.`);
    return current;
  }
  list(limit: number) {
    return this.db.prepare('SELECT * FROM discussions ORDER BY created_at DESC, id LIMIT ?').all(limit);
  }
  join(room: string, name: string, brief?: string) {
    return this.db.transaction(() => {
      const existing = this.db.prepare('SELECT id FROM discussions WHERE id = ?').get(room);
      if (!existing) {
        if (!brief) throw new Error('A brief is required to create a discussion.');
        this.db.prepare('INSERT INTO discussions (id, brief, created_at) VALUES (?, ?, ?)').run(room, brief, new Date().toISOString());
      }
      const id = randomUUID();
      this.db.prepare('INSERT INTO participants VALUES (?, ?, ?, ?)').run(id, room, name, new Date().toISOString());
      return { discussion: this.room(room), participant_id: id, protocol: DISCUSSION_PROTOCOL, next_step: 'Read from after_id=0 before replying. Joining never overwrites an existing brief or resumes a room.' };
    }).immediate();
  }
  joinViewer(room: string, participantId: string) {
    return this.db.transaction(() => {
      this.room(room);
      const existing = this.db.prepare('SELECT room, name FROM participants WHERE id = ?').get(participantId) as { room: string; name: string } | undefined;
      if (existing) {
        if (existing.room !== room || existing.name !== 'You') throw new Error('Viewer identity belongs to another participant. Reload the page with a fresh viewer identity.');
      } else {
        this.db.prepare('INSERT INTO participants VALUES (?, ?, ?, ?)').run(participantId, room, 'You', new Date().toISOString());
      }
      return participantId;
    }).immediate();
  }
  read(room: string, after: number, limit: number) {
    return this.db.transaction(() => {
      const discussion = this.room(room);
      const rows = this.db.prepare('SELECT * FROM messages WHERE room = ? AND id > ? ORDER BY id LIMIT ?').all(room, after, limit + 1) as Message[];
      const messages = rows.slice(0, limit);
      return {
        discussion, messages, has_more: rows.length > limit,
        next_after_id: messages.at(-1)?.id ?? after,
        participants: this.db.prepare('SELECT id, name, joined_at FROM participants WHERE room = ? ORDER BY joined_at').all(room),
        decision: this.db.prepare('SELECT * FROM decisions WHERE room = ? ORDER BY revision DESC LIMIT 1').get(room) ?? null,
      };
    })();
  }
  private previous(participant: string, request: string) {
    return this.db.prepare('SELECT * FROM messages WHERE participant_id = ? AND request_id = ?').get(participant, request) as Message | undefined;
  }
  private insert(room: string, participant: string, request: string, kind: string, body: string, replyTo?: number) {
    const { name } = this.participant(room, participant);
    if (replyTo !== undefined && !this.db.prepare('SELECT id FROM messages WHERE room = ? AND id = ?').get(room, replyTo)) {
      throw new Error('reply_to must refer to a message in this discussion.');
    }
    const result = this.db.prepare('INSERT INTO messages (room, participant_id, author, kind, body, reply_to, request_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)')
      .run(room, participant, name, kind, body, replyTo ?? null, request, new Date().toISOString());
    return this.db.prepare('SELECT * FROM messages WHERE id = ?').get(result.lastInsertRowid) as Message;
  }
  post(room: string, participant: string, request: string, kind: string, body: string, replyTo?: number) {
    return this.db.transaction(() => {
      this.participant(room, participant);
      const previous = this.previous(participant, request);
      if (previous) {
        if (previous.kind !== kind || previous.body !== body || previous.reply_to !== (replyTo ?? null)) throw new Error('request_id was already used for different content.');
        return previous;
      }
      this.active(room);
      return this.insert(room, participant, request, kind, body, replyTo);
    }).immediate();
  }
  status(room: string, participant: string, request: string, status: Room['status'], reason: string) {
    return this.db.transaction(() => {
      this.participant(room, participant);
      const body = JSON.stringify({ status, reason });
      const previous = this.previous(participant, request);
      if (previous) {
        if (previous.kind !== 'status' || previous.body !== body) throw new Error('request_id was already used for different content.');
        return { event: previous, discussion: this.room(room) };
      }
      this.db.prepare('UPDATE discussions SET status = ? WHERE id = ?').run(status, room);
      return { event: this.insert(room, participant, request, 'status', body), discussion: this.room(room) };
    }).immediate();
  }
  decide(room: string, participant: string, request: string, expected: number, plan: string, disagreements: string) {
    return this.db.transaction(() => {
      this.participant(room, participant);
      const body = JSON.stringify({ plan, disagreements, expected_revision: expected });
      const previous = this.previous(participant, request);
      if (previous) {
        if (previous.kind !== 'decision' || previous.body !== body) throw new Error('request_id was already used for different content.');
        return this.db.prepare('SELECT * FROM decisions WHERE message_id = ?').get(previous.id);
      }
      const current = this.active(room);
      if (current.revision !== expected) throw new Error(`Decision changed: expected revision ${expected}, current ${current.revision}. Read it before revising.`);
      const message = this.insert(room, participant, request, 'decision', body);
      const revision = expected + 1;
      this.db.prepare('INSERT INTO decisions VALUES (?, ?, ?, ?, ?)').run(room, revision, message.id, plan, disagreements);
      this.db.prepare('UPDATE discussions SET revision = ? WHERE id = ?').run(revision, room);
      return { room, revision, message_id: message.id, plan, disagreements };
    }).immediate();
  }
  async wait(room: string, after: number, timeoutMs: number, limit: number, signal?: AbortSignal) {
    const deadline = Date.now() + timeoutMs;
    while (true) {
      signal?.throwIfAborted();
      const page = this.read(room, after, limit);
      if (page.messages.length || page.discussion.status !== 'active') return { ...page, timed_out: false };
      const remaining = deadline - Date.now();
      if (remaining <= 0) return { ...page, timed_out: true };
      await delay(Math.min(250, remaining), undefined, { signal });
    }
  }
}

export function registerCollaborationTools(server: McpServer, dbPath = process.env.HOTSTEP_COLLAB_DB ?? DEFAULT_COLLAB_DB) {
  // Lazy opening keeps existing lyric-only clients independent of collaboration storage.
  let store: DiscussionStore | undefined;
  const get = () => store ??= new DiscussionStore(dbPath);
  const result = async (action: () => unknown | Promise<unknown>) => {
    try { return { content: [{ type: 'text' as const, text: JSON.stringify(await action()) }] }; }
    catch (error) { return { isError: true, content: [{ type: 'text' as const, text: error instanceof Error ? error.message : String(error) }] }; }
  };
  const room = z.string().min(1).max(100).regex(/^[a-zA-Z0-9][a-zA-Z0-9._-]*$/).describe('Shared discussion name, for example mm3-cache-design');
  const identity = { room, participant_id: z.string().uuid().describe('ID returned by your join call'), request_id: z.string().min(1).max(100).describe('Unique ID for this write. Reuse it only when retrying identical content.') };
  const cursor = { room, after_id: z.number().int().min(0).default(0).describe('Last message ID actually read, initially 0'), limit: z.number().int().min(1).max(100).default(50) };
  const body = z.string().trim().min(1).max(24000);

  server.tool('collab_list_discussions', 'List shared project discussions. Does not start another agent.', { limit: z.number().int().min(1).max(100).default(30) },
    async ({ limit }) => result(() => get().list(limit)));
  server.tool('collab_join_discussion', 'Join as this chat agent. Creates a room only if missing and a brief is supplied. Returns participation instructions; read and follow them.',
    { room, name: z.string().trim().min(1).max(100).describe('Honest chat identity, e.g. Codex or Claude'), brief: body.optional() },
    async ({ room, name, brief }) => result(() => get().join(room, name, brief)));
  server.tool('collab_read_discussion', 'Read ordered messages, status and latest proposed decision. Page until has_more=false before replying. Retain next_after_id.', cursor,
    async ({ room, after_id, limit }) => result(() => get().read(room, after_id, limit)));
  server.tool('collab_post_message', 'Post a discussion message. Relay relevant user steering as user_direction; distinguish quotes from paraphrases. This grants no permission to implement.',
    { ...identity, kind: z.enum(['proposal', 'critique', 'question', 'reply', 'user_direction', 'summary']).default('reply'), body, reply_to: z.number().int().positive().optional() },
    async ({ room, participant_id, request_id, kind, body, reply_to }) => result(() => get().post(room, participant_id, request_id, kind, body, reply_to)));
  server.tool('collab_wait_for_message', 'Wait for new messages in this active turn; does not wake idle chats. Retain next_after_id; stop after 3 consecutive timeouts or when paused/closed. Never post filler on timeout.',
    { ...cursor, timeout_ms: z.number().int().min(0).max(25000).default(20000) },
    async ({ room, after_id, limit, timeout_ms }, extra) => result(() => get().wait(room, after_id, timeout_ms, limit, extra.signal)));
  server.tool('collab_set_status', 'Pause/close on user request or completion; resume (active) only on user direction. Status changes are visible to all waiting participants.',
    { ...identity, status: z.enum(['active', 'paused', 'closed']), reason: body },
    async ({ room, participant_id, request_id, status, reason }) => result(() => get().status(room, participant_id, request_id, status, reason)));
  server.tool('collab_record_decision', 'Save a proposed plan and disagreements, not user approval. expected_revision must match the latest discussion revision to prevent overwrites.',
    { ...identity, expected_revision: z.number().int().min(0), plan: body, disagreements: z.string().max(24000).default('') },
    async ({ room, participant_id, request_id, expected_revision, plan, disagreements }) => result(() => get().decide(room, participant_id, request_id, expected_revision, plan, disagreements)));
  return { close: () => { store?.close(); store = undefined; } };
}
