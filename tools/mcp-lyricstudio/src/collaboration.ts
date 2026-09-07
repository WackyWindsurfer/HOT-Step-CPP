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
import { COORDINATION_SCHEMA, DiscussionCoordination, mentionHandle } from './discussion-coordination.js';

export const DEFAULT_COLLAB_DB = fileURLToPath(new URL('../../../data/collaboration.db', import.meta.url));

export const DISCUSSION_PROTOCOL = `You are participating as the current chat agent, not launching another model.
Read the brief and transcript before replying. Post concrete proposals and critiques with code references where useful.
Aim for 150 words per reply; agent messages are limited to 2400 characters. State only new evidence, disagreements, or the next decision. Do not repeat a peer's proposal or announce that you will reply later.
One agent contribution per turn, including a decision. After posting, wait for a different speaker (another agent or the human) before posting again. Do not send an acknowledgement followed by a proposal or decision. Combine them into one contribution. Rejoining or relaying user_direction does not bypass this rule.
Once agreement is reached, one agent records the plan as its contribution. Others need not repeat it. Stop when the requested discussion is complete.
An @mention requests a reply from that participant. Read coordination.requests and wait if another agent was asked. Requests are queued, not proof that an idle VSCode chat was woken.
Before investigating, use collab_set_activity(researching, reason). It reserves the room for 120 seconds, renewable up to 300 seconds per call, without consuming your reply. Other agents may read but must hold proposals and decisions. Human steering stays open. Do not post a separate "please hold" message.
Before answering from research, read all new messages and pass next_after_id as read_after_id with your reply or decision. The answer releases the hold and resolves your pending mentions. Use activity=idle to release without answering; the human can also clear a hold or unanswered mention. Renew before expiry if more time is needed.
If a ping needs no substantive answer, use collab_decline_request with a short reason after reading it. Activity changes and coordination events are not invitations to reply.
Check new user_direction messages before continuing the plan; the user can post directly from the group chat as You. Address their questions and constraints in the room so every participant can follow.
Relay user instructions that affect the shared plan as kind=user_direction, clearly identifying them as the user's words or a paraphrase. Never invent user approval.
Each join returns a participant_id for this chat; retain it and identify yourself honestly. These IDs prevent accidental mixups, not malicious impersonation by trusted local clients.
After reading a page, retain next_after_id. If has_more is true, read the next page before replying. Never use your posted message ID as the read cursor: other messages may have arrived before it.
Reads are compact by default: brief and participant list arrive on the initial read only; decision text arrives initially and with a new decision event. Retain earlier values. Older decision bodies are revision references. Use compact=false for full historical text or refreshed participant metadata. Write results acknowledge IDs without echoing your text.
Use collab_wait_for_message with that cursor between responses. A timeout is not a message: do not post filler or respond repeatedly to your own messages.
An empty wait ends only that tool call, not your participation. Keep calling collab_wait_for_message while the discussion is active, including while another participant researches. There is no automatic idle-time or reply-count cutoff. Stop when the requested discussion is complete, the room is paused or closed, or the user asks you to stop or sets a deadline that has arrived. Do not end your chat turn merely because repeated waits return no messages. Keep individual waits short so user steering stays responsive.
Pause or close the room when asked; all participants must stop discussion work when its status is paused or closed. Resume only on user direction.
record_decision saves an agent proposal and unresolved disagreements; it does not confer user approval or permission to implement.
MCP does not automatically wake a chat after its turn ends. The user must start or resume participation in each chat.
Keep training, generation, and source edits outside this discussion unless separately authorized.`;

type Room = { id: string; brief: string; status: 'active' | 'paused' | 'closed'; revision: number; created_at: string };
type Message = { id: number; room: string; participant_id: string; author: string; kind: string; body: string; reply_to: number | null; request_id: string; created_at: string };
type Decision = { room: string; revision: number; message_id: number; plan: string; disagreements: string };
export const MAX_AGENT_REPLY_CHARS = 2400;

function compactPage(page: ReturnType<DiscussionStore['read']>, after: number) {
  const decision = page.decision;
  const includeDecision = after === 0 || page.messages.some(m => m.id === decision?.message_id);
  return {
    discussion: { id: page.discussion.id, status: page.discussion.status, revision: page.discussion.revision,
      ...(after === 0 ? { brief: page.discussion.brief } : {}) },
    coordination: page.coordination,
    messages: page.messages.map(({ id, author, kind, body, reply_to, mentions }) => {
      if (kind === 'decision') {
        try { body = JSON.stringify({ revision: JSON.parse(body).expected_revision + 1, superseded: id !== decision?.message_id }); }
        catch { /* Preserve unrecognised historical events. */ }
      }
      return { id, author, kind, body, ...(reply_to !== null ? { reply_to } : {}), ...(mentions.length ? { mentions } : {}) };
    }),
    has_more: page.has_more, next_after_id: page.next_after_id,
    ...(after === 0 ? { participants: page.participants } : {}),
    ...(includeDecision ? { decision } : {}),
  };
}

export class DiscussionStore {
  private db: Database.Database;
  private coordination: DiscussionCoordination;

  constructor(dbPath: string, options: { readonly?: boolean } = {}) {
    if (!options.readonly) mkdirSync(dirname(resolve(dbPath)), { recursive: true });
    this.db = new Database(dbPath, { timeout: 5000, readonly: options.readonly ?? false, fileMustExist: options.readonly ?? false });
    this.coordination = new DiscussionCoordination(this.db);
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
    this.db.exec(COORDINATION_SCHEMA);
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
    if (name.trim().toLowerCase() === 'you') throw new Error('You is reserved for the human discussion viewer. Use your honest agent name.');
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
  createViewerDiscussion(room: string, brief: string, participantId: string, requestId: string) {
    return this.db.transaction(() => {
      const existing = this.db.prepare('SELECT id FROM discussions WHERE id = ?').get(room);
      if (existing) {
        const previous = this.previous(participantId, requestId);
        if (!previous || previous.room !== room || previous.kind !== 'user_direction' || previous.body !== brief || this.room(room).brief !== brief) {
          throw new Error(`Discussion ${room} already exists. Select it above or choose another name.`);
        }
        this.joinViewer(room, participantId);
        return { discussion: this.room(room), participant_id: participantId, message: previous };
      }
      this.db.prepare('INSERT INTO discussions (id, brief, created_at) VALUES (?, ?, ?)').run(room, brief, new Date().toISOString());
      this.joinViewer(room, participantId);
      const message = this.insert(room, participantId, requestId, 'user_direction', brief);
      return { discussion: this.room(room), participant_id: participantId, message };
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
      const messages = rows.slice(0, limit).map(m => ({ ...m, mentions: this.coordination.mentions(m.id) }));
      return {
        discussion, messages, has_more: rows.length > limit,
        next_after_id: messages.at(-1)?.id ?? after,
        participants: (this.db.prepare('SELECT id, name, joined_at FROM participants WHERE room = ? ORDER BY joined_at').all(room) as { id: string; name: string; joined_at: string }[]).map(p => ({ ...p, handle: mentionHandle(p.name) })),
        decision: this.latestDecision(room),
        coordination: this.coordination.snapshot(room),
      };
    })();
  }
  latestDecision(room: string) {
    return (this.db.prepare('SELECT * FROM decisions WHERE room = ? ORDER BY revision DESC LIMIT 1').get(room) as Decision | undefined) ?? null;
  }
  exportPlan(room: string) {
    return this.db.transaction(() => {
      const discussion = this.room(room);
      const decision = this.latestDecision(room);
      if (!decision) throw new Error('No proposed plan has been recorded for this discussion.');
      // Older clients sometimes sent literal backslash-n separators throughout.
      const prose = (value: string) => value.includes('\n') ? value : value.replaceAll('\\n', '\n');
      const markdown = `# ${room}: proposed plan\n\nRevision: ${decision.revision}\n\nRoom status: ${discussion.status}\n\nThis is an agent proposal, not user approval or permission to implement.\n\n## Plan\n\n${prose(decision.plan)}\n\n## Open disagreements\n\n${prose(decision.disagreements) || 'None recorded.'}\n`;
      return { filename: `${room}-r${decision.revision}.md`, markdown, revision: decision.revision };
    })();
  }
  private checkTurn(room: string, participant: string, body?: string, readAfter?: number) {
    const { name } = this.participant(room, participant);
    if (name === 'You') return; // Human messages can always steer an active room.
    this.coordination.beforeContribution(room, participant, readAfter);
    if (body !== undefined && body.length > MAX_AGENT_REPLY_CHARS) {
      throw new Error(`Keep agent replies within ${MAX_AGENT_REPLY_CHARS} characters. Combine only new evidence and your proposed next step.`);
    }
    // Agent control events must neither consume nor unlock a discussion turn.
    const last = this.db.prepare("SELECT author FROM messages WHERE room = ? AND (kind NOT IN ('status', 'coordination') OR author = 'You') ORDER BY id DESC LIMIT 1").get(room) as { author: string } | undefined;
    if (last?.author.trim().toLowerCase() === name.trim().toLowerCase()) {
      throw new Error('Wait for another agent or the human to reply before posting again. Do not retry, rejoin, or post user_direction to bypass the turn limit.');
    }
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
  post(room: string, participant: string, request: string, kind: string, body: string, replyTo?: number, readAfter?: number) {
    return this.db.transaction(() => {
      this.participant(room, participant);
      const previous = this.previous(participant, request);
      if (previous) {
        if (previous.kind !== kind || previous.body !== body || previous.reply_to !== (replyTo ?? null)) throw new Error('request_id was already used for different content.');
        return previous;
      }
      this.active(room);
      if (replyTo !== undefined && !this.db.prepare('SELECT id FROM messages WHERE room = ? AND id = ?').get(room, replyTo)) {
        throw new Error('reply_to must refer to a message in this discussion.');
      }
      this.checkTurn(room, participant, body, readAfter);
      const message = this.insert(room, participant, request, kind, body, replyTo);
      this.coordination.answered(room, participant);
      this.coordination.enqueue(room, participant, message.id, body);
      return message;
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
      if (status !== 'active') this.coordination.release(room, participant, true);
      return { event: this.insert(room, participant, request, 'status', body), discussion: this.room(room) };
    }).immediate();
  }
  decide(room: string, participant: string, request: string, expected: number, plan: string, disagreements: string, readAfter?: number) {
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
      this.checkTurn(room, participant, undefined, readAfter);
      const message = this.insert(room, participant, request, 'decision', body);
      const revision = expected + 1;
      this.db.prepare('INSERT INTO decisions VALUES (?, ?, ?, ?, ?)').run(room, revision, message.id, plan, disagreements);
      this.db.prepare('UPDATE discussions SET revision = ? WHERE id = ?').run(revision, room);
      this.coordination.answered(room, participant);
      return { room, revision, message_id: message.id, plan, disagreements };
    }).immediate();
  }
  activity(room: string, participant: string, activity: 'researching' | 'idle', reason: string, seconds: number) {
    return this.db.transaction(() => {
      if (this.participant(room, participant).name === 'You') throw new Error('Research holds belong to agents.');
      if (activity === 'idle') this.coordination.release(room, participant);
      else {
        this.active(room);
        const current = this.coordination.snapshot(room).research;
        if (!current || current.participant_id !== participant) {
          this.checkTurn(room, participant);
        }
        if (!reason.trim() || reason.length > 240) throw new Error('Give a research reason of 1 to 240 characters.');
        if (!Number.isInteger(seconds) || seconds < 30 || seconds > 300) throw new Error('Research holds must last 30 to 300 seconds.');
        this.coordination.research(room, participant, reason, seconds);
      }
      return this.coordination.snapshot(room);
    }).immediate();
  }
  clearCoordination(room: string, participant: string, request: string, action: 'release_research' | 'clear_requests') {
    return this.db.transaction(() => {
      if (this.participant(room, participant).name !== 'You') throw new Error('Only the human can clear room coordination.');
      const previous = this.previous(participant, request);
      if (previous) {
        if (previous.kind !== 'coordination' || previous.body !== action) throw new Error('request_id was already used for different content.');
        return previous;
      }
      if (action === 'release_research') this.coordination.release(room, participant, true);
      else this.coordination.clearRequests(room);
      return this.insert(room, participant, request, 'coordination', action);
    }).immediate();
  }
  declineRequest(room: string, participant: string, request: string, reason: string, readAfter: number) {
    return this.db.transaction(() => {
      this.participant(room, participant);
      const body = `Declined reply request: ${reason}`;
      const previous = this.previous(participant, request);
      if (previous) {
        if (previous.kind !== 'coordination' || previous.body !== body) throw new Error('request_id was already used for different content.');
        return { id: previous.id };
      }
      this.coordination.decline(room, participant, readAfter);
      return { id: this.insert(room, participant, request, 'coordination', body).id };
    }).immediate();
  }
  async wait(room: string, after: number, timeoutMs: number, limit: number, signal?: AbortSignal) {
    const deadline = Date.now() + timeoutMs;
    const initialCoordination = JSON.stringify(this.coordination.snapshot(room));
    while (true) {
      signal?.throwIfAborted();
      const page = this.read(room, after, limit);
      if (page.messages.length || page.discussion.status !== 'active' || JSON.stringify(page.coordination) !== initialCoordination) return { ...page, timed_out: false };
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
  const cursor = { room, after_id: z.number().int().min(0).default(0).describe('Last message ID actually read, initially 0'), limit: z.number().int().min(1).max(100).default(50), compact: z.boolean().default(true).describe('Omit repeated metadata and old decision bodies; false returns the full transcript format.') };
  const body = z.string().trim().min(1).max(24000);
  const readAfter = z.number().int().min(0).optional().describe('Last next_after_id read; required to answer while holding the research turn.');

  server.tool('collab_list_discussions', 'List shared project discussions. Does not start another agent.', { limit: z.number().int().min(1).max(100).default(30) },
    async ({ limit }) => result(() => get().list(limit)));
  server.tool('collab_join_discussion', 'Join as this chat agent. Creates a room only if missing and a brief is supplied. Returns participation instructions; read and follow them.',
    { room, name: z.string().trim().min(1).max(100).describe('Honest chat identity, e.g. Codex or Claude'), brief: body.optional() },
    async ({ room, name, brief }) => result(() => get().join(room, name, brief)));
  server.tool('collab_read_discussion', 'Read ordered messages, status and latest proposed decision. Page until has_more=false before replying. Retain next_after_id.', cursor,
    async ({ room, after_id, limit, compact }) => result(() => { const page = get().read(room, after_id, limit); return compact ? compactPage(page, after_id) : page; }));
  server.tool('collab_post_message', 'Post one reply (max 2400 characters), then wait for another speaker. Relay user steering accurately. No implementation permission.',
    { ...identity, kind: z.enum(['proposal', 'critique', 'question', 'reply', 'user_direction', 'summary']).default('reply'), body, reply_to: z.number().int().positive().optional(), read_after_id: readAfter },
    async ({ room, participant_id, request_id, kind, body, reply_to, read_after_id }) => result(() => { const message = get().post(room, participant_id, request_id, kind, body, reply_to, read_after_id); return { id: message.id, kind: message.kind }; }));
  server.tool('collab_set_activity', 'Reserve the room while researching (no reply consumed), renew the hold, or release it with idle. Human steering remains open. Read new messages before answering.',
    { room, participant_id: identity.participant_id, activity: z.enum(['researching', 'idle']), reason: z.string().trim().max(240).default(''), lease_seconds: z.number().int().min(30).max(300).default(120) },
    async ({ room, participant_id, activity, reason, lease_seconds }) => result(() => get().activity(room, participant_id, activity, reason, lease_seconds)));
  server.tool('collab_decline_request', 'Resolve your pending ping when no substantive reply is appropriate. Read it first. Does not consume or unlock a discussion turn.',
    { ...identity, reason: z.string().trim().min(1).max(240), read_after_id: z.number().int().min(0) },
    async ({ room, participant_id, request_id, reason, read_after_id }) => result(() => get().declineRequest(room, participant_id, request_id, reason, read_after_id)));
  server.tool('collab_wait_for_message', 'Wait for new messages in this active turn; does not wake idle chats. Retain next_after_id and repeat empty waits while active, including during peer research. No automatic idle or reply-count cutoff. Stop on completion, pause/close, or user stop/deadline. Never post filler on timeout.',
    { ...cursor, timeout_ms: z.number().int().min(0).max(25000).default(20000) },
    async ({ room, after_id, limit, timeout_ms, compact }, extra) => result(async () => { const page = await get().wait(room, after_id, timeout_ms, limit, extra.signal); return compact ? { ...compactPage(page, after_id), timed_out: page.timed_out } : page; }));
  server.tool('collab_set_status', 'Pause/close on user request or completion; resume (active) only on user direction. Status changes are visible to all waiting participants.',
    { ...identity, status: z.enum(['active', 'paused', 'closed']), reason: body },
    async ({ room, participant_id, request_id, status, reason }) => result(() => { const value = get().status(room, participant_id, request_id, status, reason); return { id: value.event.id, status: value.discussion.status }; }));
  server.tool('collab_record_decision', 'Save the plan as your one contribution this turn; do not post an announcement first. Not user approval. expected_revision prevents overwrites.',
    { ...identity, expected_revision: z.number().int().min(0), plan: body, disagreements: z.string().max(24000).default(''), read_after_id: readAfter },
    async ({ room, participant_id, request_id, expected_revision, plan, disagreements, read_after_id }) => result(() => { const value = get().decide(room, participant_id, request_id, expected_revision, plan, disagreements, read_after_id) as Decision; return { revision: value.revision, message_id: value.message_id }; }));
  return { close: () => { store?.close(); store = undefined; } };
}
