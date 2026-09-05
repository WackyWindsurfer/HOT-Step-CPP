'use strict';
const byId = id => document.getElementById(id);
const rooms = byId('rooms');
const connection = byId('connection');
const messages = byId('messages');
let selectedRoom = new URLSearchParams(location.search).get('room') || '';
let cursor = 0;
let messageCount = 0;
let roomSignature = '';
let timer;
let polling = false;
let sending = false;
let roomStatus = 'active';
let selectionVersion = 0;
const memory = new Map();
function saved(key, value) {
  if (value !== undefined) {
    memory.set(key, value);
    try { sessionStorage.setItem(key, value); } catch { /* Keep working if browser storage is disabled. */ }
    return value;
  }
  try { return sessionStorage.getItem(key) ?? memory.get(key); } catch { return memory.get(key); }
}
function identity(room) {
  const key = `hotstep-participant:${room}`;
  return saved(key) || saved(key, crypto.randomUUID());
}

function updateControls() {
  byId('composer').hidden = !selectedRoom;
  byId('send').disabled = sending || !selectedRoom || roomStatus !== 'active';
  byId('message').disabled = sending;
  rooms.disabled = sending;
  byId('pause').hidden = roomStatus !== 'active';
  byId('resume').hidden = roomStatus === 'active';
  byId('pause').disabled = sending;
  byId('resume').disabled = sending;
  for (const id of ['create', 'new-room-name', 'new-room-brief']) byId(id).disabled = sending;
  byId('invite').hidden = !selectedRoom;
  const invitation = `Join MCP discussion room "${selectedRoom}" as this chat's agent. Read the brief and discussion with compact reads. Aim for 150 words: only new evidence, disagreements or the next decision. Post once, then wait for another speaker; a plan revision counts as your turn. Stop at consensus. Planning only.`;
  if (byId('invite-text').value !== invitation) byId('invite-text').value = invitation;
}

function selectRoom(room) {
  selectionVersion++;
  if (selectedRoom) saved(`hotstep-draft:${selectedRoom}`, byId('message').value);
  selectedRoom = room;
  roomStatus = 'active';
  byId('message').value = saved(`hotstep-draft:${room}`) || '';
  byId('send-status').textContent = '';
  cursor = 0;
  messageCount = 0;
  messages.replaceChildren();
  byId('details').hidden = true;
  byId('empty').hidden = false;
  byId('empty').textContent = room ? 'Waiting for the first message.' : 'Create a discussion using the form on the left, then send the invitation to each agent in VSCode.';
  byId('room-title').textContent = room || 'Full conversation';
  byId('count').textContent = room ? 'Loading conversation...' : 'Waiting for a discussion';
  const url = new URL(location.href);
  if (room) url.searchParams.set('room', room); else url.searchParams.delete('room');
  history.replaceState(null, '', url);
  updateControls();
}

byId('copy-invite').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(byId('invite-text').value);
    byId('copy-invite').textContent = 'Copied';
    setTimeout(() => { byId('copy-invite').textContent = 'Copy invitation'; }, 2000);
  } catch {
    byId('invite-text').focus();
    byId('invite-text').select();
    byId('copy-invite').textContent = 'Press Ctrl+C to copy';
  }
});
for (const id of ['new-room-name', 'new-room-brief']) {
  byId(id).value = saved(`hotstep-${id}`) || '';
  byId(id).addEventListener('input', () => saved(`hotstep-${id}`, byId(id).value));
}
byId('create-room').addEventListener('submit', async event => {
  event.preventDefault();
  if (sending) return;
  const room = byId('new-room-name').value.trim();
  const brief = byId('new-room-brief').value.trim();
  if (!room || !brief) { byId('create-status').textContent = 'Enter a room name and a discussion brief.'; return; }
  sending = true;
  updateControls();
  const key = `hotstep-create:${room}`;
  const participant_id = identity(room);
  let pending;
  try { pending = JSON.parse(saved(key) || 'null'); } catch { /* Replace invalid saved state. */ }
  if (!pending || pending.brief !== brief || pending.participant_id !== participant_id) {
    pending = { room, brief, participant_id, request_id: crypto.randomUUID() };
  }
  saved(key, JSON.stringify(pending));
  byId('create-status').textContent = 'Creating discussion...';
  try {
    const response = await fetch('/api/discussions', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(pending), signal: AbortSignal.timeout(8000),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Unable to create the discussion.');
    saved(key, '');
    for (const id of ['new-room-name', 'new-room-brief']) { byId(id).value = ''; saved(`hotstep-${id}`, ''); }
    byId('create-status').textContent = '';
    byId('new-room').open = false;
    selectRoom(result.discussion.id);
    if (![...rooms.options].some(option => option.value === room)) rooms.append(new Option(`${room} (${result.discussion.status})`, room));
    rooms.value = room;
    byId('send-status').textContent = 'Room created. Copy the invitation on the left into each agent chat.';
    clearTimeout(timer);
    void poll();
  } catch (error) {
    byId('create-status').textContent = `${error.message} Your entries are kept.`;
  } finally {
    sending = false;
    updateControls();
  }
});

byId('message').addEventListener('input', () => {
  if (selectedRoom) saved(`hotstep-draft:${selectedRoom}`, byId('message').value);
});
byId('message').addEventListener('keydown', event => {
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
    event.preventDefault();
    byId('composer').requestSubmit();
  }
});
byId('composer').addEventListener('submit', event => {
  event.preventDefault();
  if (roomStatus === 'active') void write('messages', byId('message').value.trim());
});
byId('pause').addEventListener('click', () => void write('status', 'User paused the discussion from the group chat.', 'paused'));
byId('resume').addEventListener('click', () => void write('status', 'User resumed the discussion from the group chat.', 'active'));

async function write(endpoint, body, status) {
  if (sending || !selectedRoom || !body) return;
  sending = true;
  updateControls();
  const room = selectedRoom;
  const key = `hotstep-pending:${room}:${endpoint}`;
  const content = { participant_id: identity(room), body, ...(status ? { status } : {}) };
  let pending;
  try { pending = JSON.parse(saved(key) || 'null'); } catch { /* Replace an invalid saved request. */ }
  if (!pending || pending.body !== body || pending.status !== status || pending.participant_id !== content.participant_id) {
    pending = { ...content, request_id: crypto.randomUUID() };
  }
  saved(key, JSON.stringify(pending));
  byId('send-status').textContent = 'Sending...';
  try {
    const response = await fetch(`/api/discussions/${encodeURIComponent(room)}/${endpoint}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(pending), signal: AbortSignal.timeout(8000),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Unable to send.');
    saved(key, '');
    if (endpoint === 'messages') {
      byId('message').value = '';
      saved(`hotstep-draft:${room}`, '');
    } else {
      roomStatus = result.discussion.status;
    }
    byId('send-status').textContent = endpoint === 'messages' ? 'Posted to the room.' : `Discussion ${roomStatus}.`;
    // Keep the read cursor unchanged so concurrent agent posts are not skipped.
    clearTimeout(timer);
    void poll();
  } catch (error) {
    byId('send-status').textContent = `${error.message} Your draft is kept. Retry to check or send the same request.`;
  } finally {
    sending = false;
    updateControls();
  }
}

rooms.addEventListener('change', () => {
  selectRoom(rooms.value);
  clearTimeout(timer);
  void poll();
});
byId('follow').addEventListener('change', () => {
  if (byId('follow').checked) byId('composer').scrollIntoView({ block: 'end' });
});

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function renderMessage(message) {
  const article = node('article', 'message');
  article.id = `message-${message.id}`;
  article.dataset.agent = message.author.toLowerCase().includes('claude') ? 'claude' : message.author.toLowerCase().includes('codex') ? 'codex' : 'other';
  article.dataset.kind = message.kind;
  const meta = node('div', 'message-meta');
  const author = node('span', 'author', message.author);
  author.title = `Participant ${message.participant_id}`;
  meta.append(author, node('span', 'badge', message.kind.replaceAll('_', ' ')));
  const permalink = node('a', 'message-id', `#${message.id}`);
  permalink.href = `#message-${message.id}`;
  meta.append(permalink);
  const time = node('time', '', new Date(message.created_at).toLocaleString());
  time.dateTime = message.created_at;
  meta.append(time);
  let body = message.body;
  // These event bodies are generated by the collaboration server. Render text
  // only, including ordinary Markdown, so agent content cannot execute HTML.
  if (message.kind === 'status' || message.kind === 'decision') {
    try {
      const value = JSON.parse(body);
      body = message.kind === 'status' ? `${value.status}: ${value.reason}` : `${value.plan}\n\nOpen disagreements:\n${value.disagreements || 'None recorded.'}`;
    } catch { /* Display the original text if an older event has another shape. */ }
  }
  article.append(meta, node('p', 'message-body', body));
  if (message.reply_to) {
    const reply = node('a', 'reply-link', `In reply to #${message.reply_to}`);
    reply.href = `#message-${message.reply_to}`;
    reply.addEventListener('click', () => { byId('follow').checked = false; });
    article.append(reply);
  }
  messages.append(article);
  messageCount++;
}

async function get(url) {
  const response = await fetch(url, { cache: 'no-store', signal: AbortSignal.timeout(8000) });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

async function poll() {
  if (polling) return;
  polling = true;
  let nextPollMs = 1000;
  try {
    const version = selectionVersion;
    const { discussions } = await get('/api/discussions');
    if (version !== selectionVersion) return;
    const signature = JSON.stringify(discussions.map(room => [room.id, room.status]));
    if (signature !== roomSignature) {
      roomSignature = signature;
      rooms.replaceChildren();
      if (!discussions.length) rooms.append(new Option('No discussions yet', ''));
      for (const room of discussions) rooms.append(new Option(`${room.id} (${room.status})`, room.id));
    }
    if (!discussions.some(room => room.id === selectedRoom)) selectRoom(discussions[0]?.id || '');
    rooms.value = selectedRoom;
    if (selectedRoom) {
      const room = selectedRoom;
      const readVersion = selectionVersion;
      const page = await get(`/api/discussions/${encodeURIComponent(room)}?after_id=${cursor}`);
      // Ignore results from a room that was deselected while the request ran.
      if (room !== selectedRoom || readVersion !== selectionVersion) return;
      byId('details').hidden = false;
      byId('room-title').textContent = room;
      byId('brief').textContent = page.discussion.brief;
      byId('room-status').textContent = page.discussion.status;
      if (roomStatus !== page.discussion.status && !sending) {
        byId('send-status').textContent = page.discussion.status === 'active' ? '' : `Discussion ${page.discussion.status}. Resume to send messages.`;
      }
      roomStatus = page.discussion.status;
      updateControls();
      byId('participants').textContent = page.participants.map(p => p.name).join(', ') || 'No participants';
      byId('decision').hidden = !page.decision;
      if (page.decision) {
        byId('export-plan').href = `/api/discussions/${encodeURIComponent(room)}/plan.md`;
        byId('export-plan').download = `${room}-r${page.decision.revision}.md`;
        byId('revision').textContent = `(revision ${page.decision.revision})`;
        byId('plan').textContent = page.decision.plan;
        byId('disagreements').textContent = page.decision.disagreements || 'None recorded.';
      }
      for (const message of page.messages) renderMessage(message);
      cursor = page.next_after_id;
      byId('empty').hidden = messageCount > 0;
      byId('count').textContent = `${messageCount} shared message${messageCount === 1 ? '' : 's'}${page.has_more ? ' · Loading history...' : ''}`;
      if (page.messages.length && byId('follow').checked) byId('composer').scrollIntoView({ block: 'end' });
      if (page.has_more) nextPollMs = 0;
    }
    connection.dataset.state = 'live';
    connection.textContent = `Connected · Checked ${new Date().toLocaleTimeString()}`;
  } catch (error) {
    connection.dataset.state = 'error';
    connection.textContent = `${error.message} Retrying...`;
    nextPollMs = 3000;
  } finally {
    polling = false;
    timer = setTimeout(poll, nextPollMs);
  }
}

byId('message').value = saved(`hotstep-draft:${selectedRoom}`) || '';
void poll();
