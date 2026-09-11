import { randomUUID, timingSafeEqual } from 'node:crypto';
import type { ServerResponse } from 'node:http';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { isInitializeRequest } from '@modelcontextprotocol/sdk/types.js';
import { registerCollaborationTools } from './collaboration.js';
import { createDiscussionViewer, readJson } from './discussion-viewer.js';

// Each HTTP session gets its own MCP server, identity defaults and DB connection.
// The DB itself is shared with the existing stdio clients and local viewer.
export function createDiscussionHttp(options: {
  dbPath?: string;
  token: string;
  allowedHostnames: string[];
  sessionIdleMs?: number;
}) {
  if (options.token.length < 32) throw new Error('Use a discussion token of at least 32 characters.');
  const expected = Buffer.from(options.token);
  const sessions = new Map<string, {
    mcp: McpServer; transport: StreamableHTTPServerTransport; lastAccess: number; pending: number;
  }>();
  const send = (res: ServerResponse, status: number, message: string) => {
    res.writeHead(status, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ jsonrpc: '2.0', error: { code: -32000, message }, id: null }));
  };
  const server = createDiscussionViewer(options.dbPath, {
    allowedHostnames: options.allowedHostnames,
    handleRequest: async (req, res) => {
      const isMcp = req.url?.split('?')[0] === '/mcp';
      const auth = req.headers.authorization ?? '';
      let supplied = auth.startsWith('Bearer ') ? auth.slice(7) : '';
      // Browsers can use their built-in password dialog. MCP uses a Bearer header.
      if (!isMcp && auth.startsWith('Basic ')) {
        const decoded = Buffer.from(auth.slice(6), 'base64').toString('utf8');
        supplied = decoded.includes(':') ? decoded.slice(decoded.indexOf(':') + 1) : '';
      }
      const actual = Buffer.from(supplied);
      if (actual.length !== expected.length || !timingSafeEqual(actual, expected)) {
        res.setHeader('WWW-Authenticate', isMcp ? 'Bearer realm="HOT-Step discussions"' : 'Basic realm="HOT-Step discussions", charset="UTF-8"');
        send(res, 401, 'A discussion access token is required.');
        return true;
      }
      if (!isMcp) return false;
      if (!['POST', 'GET', 'DELETE'].includes(req.method ?? '')) {
        res.setHeader('Allow', 'POST, GET, DELETE');
        send(res, 405, 'Method not allowed.');
        return true;
      }
      const id = req.headers['mcp-session-id'];
      if (id !== undefined && typeof id !== 'string') {
        send(res, 400, 'Invalid session header.'); return true;
      }
      let session = id ? sessions.get(id) : undefined;
      if (id && !session) { send(res, 404, 'Session expired. Initialize a new MCP session.'); return true; }
      let body: unknown;
      if (req.method === 'POST') {
        if (!req.headers['content-type']?.startsWith('application/json')) {
          send(res, 415, 'Use application/json.'); return true;
        }
        try { body = await readJson(req); }
        catch { send(res, 400, 'Invalid or oversized JSON request.'); return true; }
      }
      if (!session) {
        if (!isInitializeRequest(body)) { send(res, 400, 'Initialize an MCP session first.'); return true; }
        if (sessions.size >= 128) { send(res, 503, 'Too many MCP sessions.'); return true; }
        const mcp = new McpServer({ name: 'hotstep-collaboration', version: '1.0.0' });
        // Streamable HTTP sessions declare no channel capability, so no wake poller either.
        const collaboration = registerCollaborationTools(mcp, options.dbPath, { wake: false });
        const transport = new StreamableHTTPServerTransport({
          sessionIdGenerator: randomUUID,
          enableJsonResponse: true,
          onsessioninitialized: sid => { sessions.set(sid, session!); },
        });
        session = { mcp, transport, lastAccess: Date.now(), pending: 0 };
        mcp.server.onclose = () => {
          if (transport.sessionId) sessions.delete(transport.sessionId);
          collaboration.close();
        };
        await mcp.connect(transport);
      }
      session.lastAccess = Date.now();
      session.pending++;
      try { await session.transport.handleRequest(req, res, body); }
      finally {
        session.pending--;
        session.lastAccess = Date.now();
        if (!session.transport.sessionId) await session.mcp.close();
      }
      return true;
    },
  });
  // Room waits remain 25-second calls, repeated indefinitely. Only unused HTTP
  // sessions expire; room history/decisions are persistent and presence has its own lease.
  const sweep = setInterval(() => {
    for (const session of sessions.values()) {
      if (!session.pending && Date.now() - session.lastAccess > (options.sessionIdleMs ?? 30 * 60_000)) {
        void session.mcp.close();
      }
    }
  }, 30_000);
  sweep.unref();
  const closeSessions = async () => {
    clearInterval(sweep);
    await Promise.all([...sessions.values()].map(session => session.mcp.close()));
  };
  server.on('close', () => { void closeSessions(); });
  return { server, closeSessions };
}
