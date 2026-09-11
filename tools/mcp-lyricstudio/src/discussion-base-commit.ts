import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('../../..', import.meta.url));

// The commit a discussion was created against, so both agents plan from the
// same frozen source and the record says which one. "-dirty" marks uncommitted
// tracked changes at creation time. Best effort: null when git is unavailable.
export function detectBaseCommit(root = process.env.HOTSTEP_COLLAB_REPO ?? REPO_ROOT): string | null {
  try {
    const run = (...args: string[]) => execFileSync('git', ['-C', root, ...args], {
      encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], timeout: 10_000,
    }).trim();
    const sha = run('rev-parse', '--short=12', 'HEAD');
    if (!/^[0-9a-f]{7,40}$/.test(sha)) return null;
    const dirty = run('status', '--porcelain', '--untracked-files=no') !== '';
    return dirty ? `${sha}-dirty` : sha;
  } catch {
    return null;
  }
}
