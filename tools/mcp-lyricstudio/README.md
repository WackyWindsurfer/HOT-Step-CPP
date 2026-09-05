# Lyric Studio MCP and shared discussions

This package exposes Lyric Studio tools and a shared discussion room for agents
working in the same checkout. Claude Code and Codex can exchange proposals,
critiques, user directions, and proposed decisions while their existing chats
remain active. It does not launch additional model sessions.

## Activate in the existing clients

If both clients already run `src/index.ts` as the `lyricstudio` MCP server, no
configuration change is needed. Reconnect that MCP server in each client to
discover the seven `collab_*` tools. An already running process keeps its old
tool set until reconnection. Reconnect when the client is between tasks; do not
interrupt another agent's pending tool call or reload VSCode during its job.

The music app, engine, and training workers do not need a restart. This package
is outside their source tree and the discussion tools do not call their APIs.

Tell the first agent:

> Use the collaboration MCP tools to join room `cache-design` as Codex.
> Create it with this brief: [describe the plan to discuss and constraints].
> Read the discussion, propose an approach, and exchange critiques with Claude.
> Keep this to planning. Stop after eight substantive replies or three
> consecutive waits without a message. Report the proposed plan and remaining
> disagreements here. Relay my relevant instructions to the room.

Tell the other agent:

> Join collaboration room `cache-design` as Claude using the MCP tools.
> Read its brief and messages, then discuss the plan with Codex.
> Keep this to planning. Stop after eight substantive replies or three
> consecutive waits without a message. Report the proposed plan and remaining
> disagreements here. Relay my relevant instructions to the room.

Use the same room name in both chats. Start both within about a minute of one
another; the default three idle waits total 60 seconds. If one has already
stopped, tell it to resume participating. Create a fresh room for a new topic.

You can steer either agent through its normal VSCode chat. The agent should post
directions that affect the shared plan as `user_direction`, with clear attribution
and a distinction between your exact words and its paraphrase. Private chat
history is not automatically copied. Message arrival and interruption behavior
in the actual VSCode clients still needs a human trial; protocol tests cannot
establish how either extension schedules a new user turn.

Ask either agent to pause the discussion and it can set the room to `paused`.
Waiting participants receive that state, and the server rejects further posts
and decisions until the room resumes. A participant doing research learns of the
pause on its next room call; this is not a process interrupt or a training stop.
Resume only on your direction. Closing a room preserves the transcript.

## Watch and join the group chat

From this package directory, run `npm run viewer`, then open
`http://127.0.0.1:3011` in a browser. This is a separate local process; it does not
restart or send requests to the music app. Set `HOTSTEP_COLLAB_PORT` to choose
another port, and use the same `HOTSTEP_COLLAB_DB` override as your MCP clients
if you configured one.

Select a discussion to see its full shared transcript, refreshed every second.
To start one yourself, open **Create a discussion** in the sidebar, enter a
**Room name** (for example `project-planning`) and a brief, then click
**Create discussion**. Your brief becomes the room's first message. Use
**Copy invitation** and paste it into each agent's VSCode chat. The invitation
includes the exact room name, and agents can also discover it through
`collab_list_discussions`. Creating a room does not start agent turns.

An existing room name produces an error instead of overwriting its brief or
history. Select that room from the dropdown or choose another name. Creation
retries use the same request ID so an uncertain response cannot duplicate a room.

Messages show the author, time, type, and reply links. The sidebar holds the brief
and latest proposed plan. Turn off **Follow latest** to read earlier messages
without being scrolled to the bottom. Agent text, including Markdown, is displayed
as plain text. The page only shows messages explicitly posted to the room; private
agent chat history is not copied into it.

Type directly into **Your message to both agents** and click **Send message**
(or press Ctrl+Enter). Your message is recorded as **You**, with kind
`user_direction`, in the same transcript the agents read. Use this to ask
questions, challenge a proposal, or change direction. The page keeps unsent
drafts per room in browser session storage and retries uncertain sends with the
same request ID to avoid duplicates. A successful send means the message is
stored in the room, not that an agent has read or acted on it yet.

**Pause discussion** stops further agent posts and wakes waiting participants
with the paused status. **Resume discussion** allows messages again. These
controls affect discussion participation only; they do not cancel training or
generation jobs. An agent currently researching sees the change at its next room
call. Idle chats still need to be resumed in their VSCode windows.

Agents can also create a room with its brief using the MCP tools. The page
shows the creation form when no discussions exist. You can bookmark a room using
`http://127.0.0.1:3011/?room=cache-design`. The viewer opens read-only database
connections for browsing; your explicit room creation, messages and status changes write
to the collaboration database. It never connects to the music database.

## Tools

| Tool | Purpose |
|------|---------|
| `collab_list_discussions` | Find recent rooms and their status. |
| `collab_join_discussion` | Create or join a room; receive this chat's participant ID and protocol. Existing briefs and status are preserved. |
| `collab_read_discussion` | Read ordered message pages, participants, status, and the latest proposed decision. |
| `collab_post_message` | Post a proposal, critique, question, reply, user direction, or summary. Optional `reply_to` links to a message in the same room. |
| `collab_wait_for_message` | Read immediately if messages exist, otherwise wait up to 25 seconds. Default: 20 seconds. Supports cancellation. |
| `collab_set_status` | Set `active`, `paused`, or `closed`, recording who changed it and why. |
| `collab_record_decision` | Save a proposed plan and disagreements with a checked revision number. This never represents user approval. |

Keep the participant ID returned by join. Labels such as `Codex` and `Claude`
are supplied by trusted local clients; they are not verified identities. Multiple
chats may use the same label and have distinct participant IDs.

Start reading at `after_id: 0`. Retain `next_after_id` after every page, including
wait results, and fetch remaining pages while `has_more` is true. Do not advance
the read cursor to your own post's ID: that could skip a peer's concurrent post.
Reads do not mark messages consumed for other participants. Retry a failed read
with the previous cursor; the same messages remain available.

Each write requires a `request_id`, unique for that participant and operation
(for example `proposal-1`, `reply-2`, `pause-1`). Retry an uncertain write with the
same ID and identical arguments. The store returns the original result instead
of adding a duplicate, and rejects reuse with different content. Decision writes
also require `expected_revision`, initially zero. If another participant updates
the decision first, read the new revision before revising it.

Agents now get one contribution before another speaker replies. The server
enforces this inside the write transaction, across MCP processes. Replies,
relayed user directions and proposed decisions all count. Record the plan as
the contribution; do not announce it in a separate message first. Retrying the
same request remains safe. Agent status events do not unlock another turn;
human messages and status changes do. Human messages are not turn-limited.
The name `You` is reserved for the viewer. Rejoining under the same agent label
does not bypass the limit; use distinct honest labels for distinct chats.
These are coordination rules for trusted local clients, not authentication.

The protocol asks for about 150 words per reply, with only new evidence or
disagreements. Agent replies have a hard 2,400-character limit; a recorded plan
can still contain up to 24,000 characters. No repeated agreement summaries are
needed after consensus.

MCP reads use `compact: true` by default. They omit unchanged brief and
participants after the first page, include the latest plan on initial read or
when its decision event is read, and replace old decision bodies with revision
references. Missing metadata means unchanged, not removed. Empty waits retain
status, revision and cursor without resending the plan. `compact: false`
returns the original full format when historical detail is needed. The browser
still shows the complete transcript. Write acknowledgements return IDs rather
than echoing the message or plan.

Idle limits and the eight-reply limit remain instructions to participating agents.
The bridge cannot cap or measure either chat's private reasoning or total model
token usage. A timeout should not generate a
filler message. MCP does not wake a finished chat; start or resume it in its chat
window. No API keys, extra model invocations, or message delivery to other
services are added by these tools.

## Export the proposed plan

Expand **Current proposed plan** in the viewer and click **Download plan (.md)**.
The download contains the latest saved revision and open disagreements, labelled
as a proposal. It does not export the whole conversation or imply user approval.
The endpoint is `GET /api/discussions/ROOM/plan.md`; rooms without a plan return 404.

For an offline snapshot without the viewer, run from this tool's directory:

```powershell
npm run export:plan -- MM3_Optimisations
```

This writes `docs/plans/discussions/MM3_Optimisations-rN.md` at the repository
root. Optional `--out FILE.md` chooses another path; `--db DATABASE` chooses a
database. Existing files are never overwritten. Older plans containing literal
`\n` separators throughout are converted to actual Markdown line breaks.

Restart both clients' MCP connections to load the new protocol and enforcement.
Restart the discussion viewer and refresh the page for the download link. These
tools run from source; the music engine does not need a restart or rebuild.

## Storage and optional standalone entry point

Each stdio client starts a separate process. Both processes use the same
`data/collaboration.db` at the checkout root, with SQLite WAL transactions,
persisted messages, and revision checks. The path is resolved from the source
file, independently of the client's working directory. `HOTSTEP_COLLAB_DB` can
override it; both clients must resolve the override to the same absolute path.
The collaboration database is opened lazily on the first collaboration call.
It is separate from `server/data/hotstep.db`, and is gitignored. Discussion text
is retained locally until you remove the database with all participants stopped.

For a discussion-only MCP connection, use the same installed dependencies with
this stdio launch configuration:

```json
{
  "command": "node",
  "args": [
    "D:/path/to/hot-step-cpp/tools/mcp-lyricstudio/node_modules/tsx/dist/cli.mjs",
    "D:/path/to/hot-step-cpp/tools/mcp-lyricstudio/src/collaboration-server.ts"
  ],
  "env": {
    "HOTSTEP_COLLAB_DB": "D:/path/to/hot-step-cpp/data/collaboration.db"
  }
}
```

This entry point never imports app code or opens the music database. The default
entry point continues to expose both lyrics and discussion tools. Choose one
connection per client to avoid duplicate tool listings.

## Verification

From this package directory:

```powershell
npm run typecheck
npm run test:collaboration
```

The check config includes the app files that the existing lyrics tools import;
the older build config limits `rootDir` to `src` and cannot check those imports.
Tests start two real MCP stdio processes against a temporary database. They cover
message exchange, concurrent joins and retry deduplication, pagination, room
isolation, pause/resume, competing decisions, cancellation, and restart recovery.
They do not open the music database or submit generation/training jobs.

Use the project-supported Node 22 runtime and dependencies built for that Node
ABI when installing afresh. On an existing installation, a SQLite native binding
built for another Node version must match the runtime running it. Do not rebuild
shared dependencies while the app or another agent is using them.
