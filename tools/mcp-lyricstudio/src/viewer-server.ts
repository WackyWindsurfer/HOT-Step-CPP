import { createDiscussionViewer } from './discussion-viewer.js';

const port = Number(process.env.HOTSTEP_COLLAB_PORT ?? 3011);
if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('HOTSTEP_COLLAB_PORT must be between 1 and 65535.');
const server = createDiscussionViewer();
server.on('error', error => {
  console.error(`[discussion-viewer] ${error.message}`);
  process.exitCode = 1;
});
server.listen(port, '127.0.0.1', () => console.error(`[discussion-viewer] http://127.0.0.1:${port}`));
