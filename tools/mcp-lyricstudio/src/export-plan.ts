import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { DEFAULT_COLLAB_DB, DiscussionStore } from './collaboration.js';

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: { out: { type: 'string' }, db: { type: 'string' }, transcript: { type: 'boolean', default: false } },
});
const room = positionals[0];
if (positionals.length !== 1 || !/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}$/.test(room ?? '')) {
  throw new Error('Usage: npm run export:plan -- ROOM [--transcript] [--out FILE.md] [--db DATABASE]');
}
const dbPath = values.db ?? process.env.HOTSTEP_COLLAB_DB ?? DEFAULT_COLLAB_DB;
// An older database gains the sealed-position columns from one writer open.
if (existsSync(dbPath)) {
  try { new DiscussionStore(dbPath).close(); }
  catch { /* A read-only file still exports whatever columns it has. */ }
}
const store = new DiscussionStore(dbPath, { readonly: true });
try {
  const plan = store.exportPlan(room, { transcript: values.transcript });
  const filename = values.transcript ? plan.filename.replace(/\.md$/, '-transcript.md') : plan.filename;
  const target = values.out ? resolve(values.out) : fileURLToPath(new URL(`../../../docs/plans/discussions/${filename}`, import.meta.url));
  mkdirSync(dirname(target), { recursive: true });
  writeFileSync(target, plan.markdown, { encoding: 'utf8', flag: 'wx' });
  console.log(target);
} finally {
  store.close();
}
