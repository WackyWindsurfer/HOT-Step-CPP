// Optional discussion-only entry point. Never loads the app or music database.
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { registerCollaborationTools } from './collaboration.js';

const server = new McpServer({ name: 'hotstep-collaboration', version: '1.0.0' });
const collaboration = registerCollaborationTools(server);
server.server.onclose = () => collaboration.close();
await server.connect(new StdioServerTransport());
