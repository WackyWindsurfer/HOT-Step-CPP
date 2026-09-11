// Optional discussion-only entry point. Never loads the app or music database.
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { registerCollaborationTools } from './collaboration.js';
import { channelServerOptions } from './discussion-wake.js';

// With HOTSTEP_COLLAB_CHANNEL=1 the server also declares the Claude Code
// channel capability so an idle `claude --channels` session is woken by room events.
const server = new McpServer({ name: 'hotstep-collaboration', version: '1.0.0' }, channelServerOptions());
const collaboration = registerCollaborationTools(server);
server.server.onclose = () => collaboration.close();
await server.connect(new StdioServerTransport());
