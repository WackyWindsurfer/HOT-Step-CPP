import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { DEFAULT_COLLAB_DB, DiscussionStore } from './collaboration.js';

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: { out: { type: 'string' }, db: { type: 'string' } },
});
const room = positionals[0];
if (positionals.length !== 1 || !/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}$/.test(room ?? '')) {
  throw new Error('Usage: npm run export:plan -- ROOM [--out FILE.md] [--db DATABASE]');
}
const store = new DiscussionStore(values.db ?? process.env.HOTSTEP_COLLAB_DB ?? DEFAULT_COLLAB_DB, { readonly: true });
try {
  const plan = store.exportPlan(room);
  const target = values.out ? resolve(values.out) : fileURLToPath(new URL(`../../../docs/plans/discussions/${plan.filename}`, import.meta.url));
  mkdirSync(dirname(target), { recursive: true });
  writeFileSync(target, plan.markdown, { encoding: 'utf8', flag: 'wx' });
  console.log(target);
} finally {
  store.close();
}
