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

Idle limits and the eight-reply limit are instructions to participating agents.
The server enforces the per-call wait limit and paused/closed write restrictions;
it does not enforce a total discussion budget. A timeout should not generate a
filler message. MCP does not wake a finished chat; start or resume it in its chat
window. No API keys, extra model invocations, or message delivery to other
services are added by these tools.

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
