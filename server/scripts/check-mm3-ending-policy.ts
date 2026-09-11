// Run with: node --import tsx scripts/check-mm3-ending-policy.ts
import assert from 'node:assert/strict';
import { mapMinimaxParams } from '../src/services/backends/minimax/generate.js';

const prompt = {caption:'Acoustic folk with a quiet ending', lyrics:'[Verse]\nThe last train leaves'};
const single = mapMinimaxParams(prompt);
assert.equal(single.req.takes, 3, 'One requested song must retain the candidate batch');
assert.equal(single.req.require_eos, true);
assert.equal(single.req.stop_after_first_eos, true);
assert(single.notes.some(note => note.includes('renders that one song')));

for (const count of [2, 3, 4]) {
  const variations = mapMinimaxParams({...prompt, mm3Takes:count});
  assert.equal(variations.req.takes, Math.max(3, count));
  assert.equal(variations.req.stop_after_first_eos, undefined, 'Explicit variations must retain all ended candidates');
}

for (const count of [1, 3]) {
  const unrestricted = mapMinimaxParams({...prompt, mm3Takes:count, mm3RequireEnding:false});
  assert.equal(unrestricted.req.takes ?? 1, count);
  assert.equal(unrestricted.req.require_eos, undefined);
  assert.equal(unrestricted.req.stop_after_first_eos, undefined);
}
console.log('PASS: one-song candidate batch, explicit variations, and unrestricted generation');
