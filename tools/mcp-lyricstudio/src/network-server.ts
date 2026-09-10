import { readFileSync } from 'node:fs';
import { createDiscussionHttp } from './discussion-http.js';

const host = process.env.HOTSTEP_COLLAB_HOST ?? '127.0.0.1';
const port = Number(process.env.HOTSTEP_COLLAB_PORT ?? 3012);
if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Invalid HOTSTEP_COLLAB_PORT.');
const token = process.env.HOTSTEP_COLLAB_TOKEN_FILE
  ? readFileSync(process.env.HOTSTEP_COLLAB_TOKEN_FILE, 'utf8').trim()
  : process.env.HOTSTEP_COLLAB_TOKEN ?? '';
const allowedHostnames = (process.env.HOTSTEP_COLLAB_ALLOWED_HOSTS ?? host).split(',').map(value => value.trim());
if (allowedHostnames.some(value => !value || value === '0.0.0.0' || value === '::')) {
  throw new Error('Set HOTSTEP_COLLAB_ALLOWED_HOSTS to the actual IP addresses or hostnames clients use.');
}
const { server, closeSessions } = createDiscussionHttp({ token, allowedHostnames });
server.on('error', error => { console.error(`[discussion-network] ${error.message}`); process.exitCode = 1; });
server.listen(port, host, () => {
  console.error(`[discussion-network] Viewer http://${host}:${port}/ ; MCP http://${host}:${port}/mcp`);
});
let stopping = false;
const stop = async () => {
  if (stopping) return;
  stopping = true;
  await closeSessions();
  server.close();
  server.closeAllConnections();
};
process.on('SIGINT', stop);
process.on('SIGTERM', stop);
