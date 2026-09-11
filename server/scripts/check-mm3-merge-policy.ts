// Run with: node --import tsx scripts/check-mm3-merge-policy.ts
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { config } from '../src/config.js';
import { mapMinimaxParams } from '../src/services/backends/minimax/generate.js';

// Mapping only: the fixture is never passed to an inference engine.
const fixtureRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'mm3-merge-policy-'));
const previousRoot = config.aceServer.adapters;
const adapterDir = path.join(fixtureRoot, 'mm3-lm-adapters');
fs.mkdirSync(adapterDir);
fs.writeFileSync(path.join(adapterDir, 'fixture.safetensors'), 'mapping fixture');
config.aceServer.adapters = fixtureRoot;
try {
  const params = {
    caption: 'Acoustic folk', lyrics: '[Verse]\nThe last train leaves',
    mm3LmAdapter: 'fixture.safetensors', mm3LmAdapterMode: 'merge',
  };
  assert.equal(mapMinimaxParams(params).req.lm_adapter_merge_gpu, true, 'GPU is the default merge policy');
  assert.equal(mapMinimaxParams({ ...params, mm3LmMergeGpu: true }).req.lm_adapter_merge_gpu, true);
  const cpu = mapMinimaxParams({ ...params, mm3LmMergeGpu: false });
  assert.equal(cpu.req.lm_adapter_merge_gpu, false, 'The unchecked toggle must survive request mapping');
  assert(cpu.notes.some(note => note.includes('CPU-assisted')));
  assert.equal(mapMinimaxParams({ ...params, mm3LmAdapterMode: 'runtime' }).req.lm_adapter_merge_gpu, true);
  assert.equal(mapMinimaxParams({ caption: params.caption }).req.lm_adapter_merge_gpu, undefined);
  console.log('PASS: GPU default, explicit GPU/CPU selection, runtime mode and no-adapter mapping');
} finally {
  config.aceServer.adapters = previousRoot;
  fs.unlinkSync(path.join(adapterDir, 'fixture.safetensors'));
  fs.rmdirSync(adapterDir);
  fs.rmdirSync(fixtureRoot);
}
