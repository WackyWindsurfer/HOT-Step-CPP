/**
 * param-ab.ts — parameterization A/B batch, driven THROUGH the running app.
 *
 * Why through the app and not ace-train directly: the app's training routes
 * are the form's defaults (an empty option bag is the batch-pipeline recipe),
 * its job queue serialises the GPU, it stops and respawns its engine around
 * every run, and the Monitor tab shows progress. What this script adds is the
 * matrix, the renders and the listening-hub staging.
 *
 *   npx tsx server/scripts/param-ab.ts --phase lm      # form-default LM run + eval
 *   npx tsx server/scripts/param-ab.ts --phase dit     # 3 datasets x 8 arms, target 0.5
 *   npx tsx server/scripts/param-ab.ts --phase all
 *     [--datasets mj_dangerous,dio_holydiver] [--target 0.5] [--epochs 200] [--rank 128]
 *
 * Needs the Node server on :3001 and its engine on config.aceServer.port.
 * Re-runnable: arms whose wav already exists are skipped.
 */
import fs from 'fs';
import path from 'path';
import { spawnSync } from 'child_process';
import crypto from 'crypto';
import { aceClient, type AceRequest } from '../src/services/aceClient.js';
import { config } from '../src/config.js';

const API   = 'http://127.0.0.1:3001/api/training';
const ROOT  = path.resolve(process.cwd(), '..');  // run from server/
const STAMP = '2026-09-04';
const LST   = path.join(ROOT, '_experiments', '_LISTENING', `${STAMP}-dit-param-ab`);
const SYNTH_MODEL = 'acestep-v15-merge-base-sft-turbo-xl-thirds-BF16.gguf';
const LM_MODEL    = 'acestep-5Hz-lm-4B-BF16.gguf';
const VAE_MODEL   = 'scragvae-BF16.gguf';
const EMB_MODEL   = 'Qwen3-Embedding-0.6B-BF16.gguf';
const P50 = {
  inference_steps: 50, guidance_scale: 20, shift: -1,
  infer_method: 'md_hamiltonian_v2', scheduler: 'linear_quadratic', guidance_mode: 'dynamic_cfg',
} as Partial<AceRequest>;

type Ds = { id: string; caption: string; lyrics: string };
const DATASETS: Record<string, Ds> = {
  mj_dangerous: {
    id: '732bfdcd-fcb9-4d6a-b5bd-c6c21dac3286',
    caption: 'a slick early-90s pop record, tight funk drums, slap bass, bright synth stabs, male lead vocal with layered harmonies',
    lyrics: '[Verse]\nHold the line and let the rhythm take it\n\n[Chorus]\nWe can turn it up tonight\n',
  },
  dio_holydiver: {
    id: '8203ce8f-92c3-480f-a073-0bfe463c0a58',
    caption: 'classic early-80s heavy metal, galloping bass, twin harmony guitars, powerful male vocal, big arena drums',
    lyrics: '[Verse]\nRide the night on wheels of thunder\n\n[Chorus]\nHold on, the sky is falling down\n',
  },
  adtr_whatseparates: {
    id: '7f18925e-ad7c-4ac2-a860-f6ba9a18b18f',
    caption: 'anthemic pop punk with metalcore breakdowns, chugging drop-tuned guitars, fast punchy drums, gang vocals, clean male lead switching to screams',
    lyrics: '[Verse]\nBurn the map and drive all night\n\n[Chorus]\nWe were never coming home\n',
  },
  carpenterbrut_trilogy: {
    id: '3da921d9-45e0-4a8b-8dd5-c4ffe262bf33',
    caption: 'dark synthwave, pounding analog synth bass, gated retro drum machine, distorted lead synth, instrumental, cinematic',
    lyrics: '[Instrumental]\n',
  },
};

// LoRA-shaped arms at the shipped rank; rsLoRA at matched strength
// (alpha/sqrt(r) == 256/128 -> alpha 2*sqrt(128) ~ 23); HRA at r=8 because its
// step cost scales with the reflection count (10x a LoRA already at 8).
type Arm = { key: string; body: Record<string, unknown>; note: string };
function arms(rank: number): Arm[] {
  const alpha = rank * 2;
  const base = { adapterType: 'lora', rank, alpha };
  return [
    { key: 'lora',     body: { ...base },                                          note: 'control' },
    { key: 'dora',     body: { ...base, dora: true },                              note: 'learned per-row magnitude' },
    { key: 'pissa',    body: { ...base, pissa: true },                             note: 'principal-direction init, exported as rank-2r' },
    { key: 'loraplus', body: { ...base, loraPlusRatio: 16 },                       note: 'B at 16x A LR' },
    { key: 'rslora',   body: { ...base, alpha: Math.round(2 * Math.sqrt(rank)), rslora: true }, note: 'alpha/sqrt(r), strength matched to control' },
    { key: 'hira',     body: { ...base, hira: true },                              note: 'W (.) BA, merge-only' },
    { key: 'loha',     body: { ...base, loha: true },                              note: 'Hadamard of two pairs, merge-only' },
    { key: 'hra',      body: { adapterType: 'lora', rank: 8, alpha: 8, hra: true }, note: '8 Householder reflections, orthogonal' },
    // LyCORIS LoKr at the form's own defaults (dim 512, alpha 512, factor 6,
    // decompose both) — the other shipped parameterization, for coverage.
    { key: 'lokr',     body: { adapterType: 'lokr' },                             note: 'LoKr dim 512 / factor 6 (form defaults)' },
    // PiSSA again after the export moved to F32 (the BF16 rank-2r file carried
    // ~10% cancellation noise); a new name so the route trains from scratch.
    { key: 'pissa32',  body: { adapterType: 'lora', rank: 128, alpha: 256, pissa: true }, note: 'PiSSA, F32 export' },
  ];
}

function log(m: string) { console.log(`[${new Date().toISOString().slice(11, 19)}] ${m}`); }
function args() {
  const a = new Map<string, string>();
  for (let i = 2; i < process.argv.length; i += 2) a.set(process.argv[i].replace(/^--/, ''), process.argv[i + 1] ?? '');
  return a;
}

async function post(p: string, body: unknown): Promise<string> {
  const r = await fetch(`${API}${p}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json() as { jobId?: string; error?: string };
  if (!r.ok || !j.jobId) throw new Error(`${p}: HTTP ${r.status} ${j.error ?? ''}`);
  return j.jobId;
}
async function waitJob(jobId: string, label: string): Promise<{ status: string; error: string | null; secs: number }> {
  const t0 = Date.now();
  let lastPhase = '';
  let unknown = 0;
  for (;;) {
    const res = await fetch(`${API}/jobs/${jobId}`).catch(() => null);
    const j = res ? await res.json().catch(() => ({})) as { status?: string; phase?: string; error?: string | null } : {};
    if (!res || res.status === 404 || !j.status) {
      // The job queue is in-memory: a server restart forgets every job. Three
      // misses in a row = the run is gone (its child died with the server).
      if (++unknown >= 3) return { status: 'failed', error: 'job vanished (server restarted?)', secs: (Date.now() - t0) / 1000 };
      await new Promise(r => setTimeout(r, 5000));
      continue;
    }
    unknown = 0;
    if (j.phase && j.phase !== lastPhase) { lastPhase = j.phase; log(`${label}: ${j.phase}`); }
    if (j.status === 'done' || j.status === 'failed' || j.status === 'cancelled') {
      return { status: j.status, error: j.error ?? null, secs: (Date.now() - t0) / 1000 };
    }
    await new Promise(r => setTimeout(r, 5000));
  }
}
async function waitEngine(): Promise<void> {
  for (let i = 0; i < 120; i++) {
    const ok = await fetch(`http://127.0.0.1:${config.aceServer.port}/health`).then(r => r.ok).catch(() => false);
    if (ok) return;
    await new Promise(r => setTimeout(r, 3000));
  }
  throw new Error('engine did not come back within 6 min');
}
function newestRunDir(prefixGlob: string, name: string): string | null {
  const root = config.aceServer.adapters;
  let best: { dir: string; mtime: number } | null = null;
  for (const top of fs.readdirSync(root)) {
    if (!top.startsWith(prefixGlob)) continue;
    const nameDir = path.join(root, top, name);
    if (!fs.existsSync(nameDir)) continue;
    for (const run of fs.readdirSync(nameDir)) {
      const cfg = path.join(nameDir, run, 'adapter_config.json');
      const alt = path.join(nameDir, run, 'lokr_weights.safetensors');
      if (!fs.existsSync(cfg) && !fs.existsSync(alt)) continue;
      const mtime = fs.statSync(path.join(nameDir, run)).mtimeMs;
      if (!best || mtime > best.mtime) best = { dir: path.join(nameDir, run), mtime };
    }
  }
  return best?.dir ?? null;
}

// Completion state of a DiT run from its log: complete when the epoch cap was
// reached or the target-loss stop fired; anything else is a partial run.
function runState(dir: string): { complete: boolean; epochsRun: number; reason: string } {
  try {
    const jl = JSON.parse(fs.readFileSync(path.join(dir, 'dit_train_log.json'), 'utf8')) as {
      epochs_run?: number; saved_reason?: string; epochs?: unknown[]; config?: { epochs?: number };
    };
    const run = jl.epochs_run ?? (jl.epochs?.length ?? 0);
    const cap = jl.config?.epochs ?? 0;
    const target = jl.saved_reason === 'target';
    const complete = target || (cap > 0 && run >= cap);
    return { complete, epochsRun: run, reason: target ? 'target reached' : complete ? 'epoch cap' : 'partial' };
  } catch {
    return { complete: false, epochsRun: 0, reason: 'no log' };
  }
}

// ── renders (same recipe as artist-token-ab.ts; the LM pass is shared per dataset) ──
const codesCache = new Map<string, AceRequest>();
async function awaitAce(jobId: string, what: string): Promise<void> {
  const deadline = Date.now() + 20 * 60 * 1000;
  for (;;) {
    if (Date.now() > deadline) { await aceClient.cancelJob(jobId).catch(() => {}); throw new Error(`${what} timed out`); }
    const st = await aceClient.pollJob(jobId);
    if (st.status === 'done') return;
    if (st.status === 'failed') throw new Error(`${what} failed — see the engine log`);
    if (st.status === 'cancelled') throw new Error(`${what} cancelled`);
    await new Promise(r => setTimeout(r, 500));
  }
}
type RenderOverride = { caption?: string; lyrics?: string; duration?: number; seed?: number; key?: string };
async function render(dsKey: string, adapterDir: string | null, out: string, ov: RenderOverride = {}): Promise<void> {
  const ds = DATASETS[dsKey];
  const cacheKey = ov.key ?? dsKey;
  const aceReq: AceRequest = {
    caption: ov.caption ?? ds.caption, lyrics: ov.lyrics ?? ds.lyrics, duration: ov.duration ?? 60, seed: ov.seed ?? 20260904,
    vocal_language: 'en',
    lm_batch_size: 1, lm_temperature: 0.85, lm_cfg_scale: 2, lm_top_p: 0.9, lm_top_k: 0,
    lm_negative_prompt: 'NO USER INPUT',
    use_cot_caption: true, task_type: 'text2music',
    synth_model: SYNTH_MODEL, lm_model: LM_MODEL, vae_model: VAE_MODEL, emb_model: EMB_MODEL,
    ...P50,
  };
  let lmOut = codesCache.get(cacheKey);
  if (!lmOut) {
    const lmJob = await aceClient.submitLm(aceReq);
    await awaitAce(lmJob, `LM (${cacheKey})`);
    const arr = await (await aceClient.getJobResult(lmJob)).json() as AceRequest[];
    if (!Array.isArray(arr) || !arr.length || !arr[0].audio_codes) throw new Error(`LM returned no codes for ${cacheKey}`);
    lmOut = arr[0];
    codesCache.set(cacheKey, lmOut);
    log(`LM plan (${cacheKey}): bpm=${lmOut.bpm} dur=${lmOut.duration} codes=${(lmOut.audio_codes || '').split(',').length}`);
  }
  const synthReq: AceRequest = {
    ...aceReq,
    audio_codes: lmOut.audio_codes, caption: lmOut.caption || aceReq.caption, lyrics: lmOut.lyrics,
    bpm: lmOut.bpm, duration: lmOut.duration, keyscale: lmOut.keyscale, timesignature: lmOut.timesignature,
    lm_seed: lmOut.lm_seed,
  };
  if (adapterDir) {
    synthReq.adapter = adapterDir;
    synthReq.adapter_scale = 1;
    synthReq.adapters = [{ name: adapterDir, scale: 1 }];
  }
  const sJob = await aceClient.submitSynth(synthReq, 'wav16');
  await awaitAce(sJob, `synth (${path.basename(out)})`);
  const buf = Buffer.from(await (await aceClient.getJobResult(sJob)).arrayBuffer());
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, buf);
}

// ── phase LM: the form default with the token on, evaluated ──
async function phaseLm(): Promise<void> {
  const name = 'mjd-formdefault';
  const exp = path.join(ROOT, '_experiments', 'artist-token', 'formdefault_4B');
  fs.mkdirSync(exp, { recursive: true });
  log(`LM: posting train-lm on mj_dangerous with an EMPTY option bag (adapterName ${name})`);
  const jobId = await post(`/datasets/${DATASETS.mj_dangerous.id}/train-lm`, { adapterName: name });
  const r = await waitJob(jobId, 'LM');
  log(`LM: ${r.status} in ${Math.round(r.secs)} s ${r.error ?? ''}`);
  if (r.status !== 'done') return;
  const dir = newestRunDir('lm-', name);
  if (!dir) { log('LM: no run dir found'); return; }
  fs.writeFileSync(path.join(exp, 'RUN_DIR.txt'), dir);
  await waitEngine();
  log(`LM: eval of ${dir}`);
  const gen = spawnSync('npx', ['tsx', 'scripts/lm-adapter-eval.ts', 'generate', '--dataset', 'mj_dangerous', '--adapter', dir,
    '--lm-model', LM_MODEL, '--seeds', '2', '--max-duration', '60', '--out', path.join(exp, 'lm-eval')],
    { cwd: path.join(ROOT, 'server'), shell: true, encoding: 'utf8' });
  fs.writeFileSync(path.join(exp, 'eval_generate.log'), (gen.stdout || '') + (gen.stderr || ''));
  const rep = spawnSync('npx', ['tsx', 'scripts/lm-adapter-eval.ts', 'report', '--run', path.join(exp, 'lm-eval')],
    { cwd: path.join(ROOT, 'server'), shell: true, encoding: 'utf8' });
  fs.writeFileSync(path.join(exp, 'eval_report.txt'), (rep.stdout || '') + (rep.stderr || ''));
  const verdict = (rep.stdout || '').split('\n').filter(l => /VERDICT|Unigram JS|Marginals|Transitions|OOV/.test(l)).join('\n');
  log(`LM eval:\n${verdict}`);
}

// ── phase DiT: the matrix ──
async function phaseDit(dsKeys: string[], target: number, epochs: number, rank: number, only: string[] = []): Promise<void> {
  fs.mkdirSync(LST, { recursive: true });
  const readme = path.join(LST, 'README.md');
  if (!fs.existsSync(readme)) {
    fs.writeFileSync(readme,
      `# DiT parameterization A/B — ${STAMP}\n\nAll arms: same dataset tensors, target loss ${target} (ma5), epoch cap ${epochs}, ` +
      `rank ${rank}/alpha ${rank * 2} unless noted, same seed/caption/lyrics, rendered through the app's engine at 60 s. ` +
      `00 = base model, no adapter. Trained through the app's train-dit route (form defaults for everything not listed).\n\n` +
      `| dataset | # | arm | epochs (cap ${epochs}) | final ma5 | train s | note |\n|---|---|---|---|---|---|---|\n`);
  }
  for (const dsKey of dsKeys) {
    const ds = DATASETS[dsKey];
    const outDir = path.join(LST, dsKey);
    fs.mkdirSync(outDir, { recursive: true });
    const baseWav = path.join(outDir, '00_base.wav');
    if (!fs.existsSync(baseWav)) {
      await waitEngine();
      log(`${dsKey}: base render`);
      await render(dsKey, null, baseWav);
    }
    let n = 0;
    for (const arm of arms(rank)) {
      n++;
      if (only.length && !only.includes(arm.key)) continue;
      const wav = path.join(outDir, `${String(n).padStart(2, '0')}_${arm.key}.wav`);
      if (fs.existsSync(wav)) { log(`${dsKey}/${arm.key}: exists, skipping`); continue; }
      const name = `ab-${dsKey}-${arm.key}`;
      const body = { adapterName: name, targetLoss: target, epochs, ...arm.body };
      let row: string;
      try {
        // An adapter that already exists is either COMPLETE (a previous runner
        // instance trained it but never rendered: reuse, training is the
        // expensive half) or PARTIAL (a paused/cancelled run). The train-dit
        // route resumes from the latest same-named adapter when initAdapter is
        // omitted, so a partial arm is re-posted for its remaining epochs.
        let dir = newestRunDir('dit-', name);
        let secs = 0;
        let priorEpochs = 0;
        if (dir) {
          const st = runState(dir);
          if (st.complete) {
            log(`${dsKey}/${arm.key}: adapter complete at ${dir} (${st.epochsRun} epochs, ${st.reason}), rendering only`);
          } else {
            priorEpochs = st.epochsRun;
            log(`${dsKey}/${arm.key}: partial adapter at ${dir} (${st.epochsRun}/${epochs} epochs), resuming for ${Math.max(1, epochs - st.epochsRun)} more`);
            dir = null;
          }
        }
        if (!dir) {
          const remaining = Math.max(1, epochs - priorEpochs);
          const postBody = { ...body, epochs: remaining };
          log(`${dsKey}/${arm.key}: posting train-dit ${JSON.stringify(arm.body)} epochs=${remaining}`);
          const jobId = await post(`/datasets/${ds.id}/train-dit`, postBody);
          const r = await waitJob(jobId, `${dsKey}/${arm.key}`);
          secs = r.secs;
          if (r.status !== 'done') {
            row = `| ${dsKey} | ${n} | ${arm.key} | — | — | ${Math.round(r.secs)} | FAILED: ${r.error ?? r.status} |\n`;
            fs.appendFileSync(readme, row);
            continue;
          }
          dir = newestRunDir('dit-', name);
          if (!dir) throw new Error('no run dir');
        }
        let ep = '?', ma5 = '?';
        try {
          const jl = JSON.parse(fs.readFileSync(path.join(dir, 'dit_train_log.json'), 'utf8')) as { epochs?: { loss: number; ma5?: number }[] };
          const eps = jl.epochs ?? [];
          ep = priorEpochs > 0 ? `${priorEpochs}+${eps.length}` : String(eps.length);
          const last = eps[eps.length - 1];
          ma5 = last ? (last.ma5 ?? last.loss).toFixed(4) : '?';
        } catch { /* row keeps ? */ }
        await waitEngine();
        log(`${dsKey}/${arm.key}: render from ${dir}`);
        await render(dsKey, dir, wav);
        row = `| ${dsKey} | ${n} | ${arm.key} | ${ep} | ${ma5} | ${Math.round(secs)} | ${arm.note} |\n`;
      } catch (e: any) {
        row = `| ${dsKey} | ${n} | ${arm.key} | — | — | — | ERROR: ${String(e?.message ?? e).slice(0, 120)} |\n`;
      }
      fs.appendFileSync(readme, row);
      log(row.trim());
    }
  }
}

// ── phase render: full-length renders of every existing arm with one plan ──
// Same caption/lyrics/seed for all, only the adapter changes. Partial adapters
// (a stopped arm) render as they are; the README says which.
async function phaseRender(dsKey: string, sub: string, ov: RenderOverride, rank: number, only: string[] = [], withBase = true): Promise<void> {
  const ds = DATASETS[dsKey];
  if (!ds) throw new Error(`unknown dataset ${dsKey}`);
  const outDir = path.join(LST, dsKey, sub);
  fs.mkdirSync(outDir, { recursive: true });
  const readme = path.join(outDir, 'README.md');
  if (!fs.existsSync(readme)) {
    fs.writeFileSync(readme,
      `# ${dsKey} — full-length renders (${sub})\n\nOne plan for all arms (same caption, lyrics, seed ${ov.seed ?? 20260904}, ` +
      `duration ${ov.duration ?? '?'} s), thirds DiT, 50 steps, guidance 20, md_hamiltonian_v2 / linear_quadratic / dynamic_cfg, ` +
      `no post-processing. Only the adapter changes. 00 = base.\n\n| # | arm | adapter | state |\n|---|---|---|---|\n`);
  }
  await waitEngine();
  const baseWav = path.join(outDir, '00_base.wav');
  if (withBase && !fs.existsSync(baseWav)) {
    log(`${dsKey}/${sub}: base render`);
    await render(dsKey, null, baseWav, ov);
    fs.appendFileSync(readme, `| 0 | base | — | no adapter |\n`);
  }
  let n = 0;
  for (const arm of arms(rank)) {
    n++;
    if (only.length && !only.includes(arm.key)) continue;
    const wav = path.join(outDir, `${String(n).padStart(2, '0')}_${arm.key}.wav`);
    if (fs.existsSync(wav)) { log(`${dsKey}/${sub}/${arm.key}: exists, skipping`); continue; }
    const dir = newestRunDir('dit-', `ab-${dsKey}-${arm.key}`);
    if (!dir) { log(`${dsKey}/${sub}/${arm.key}: no adapter, skipping`); fs.appendFileSync(readme, `| ${n} | ${arm.key} | — | not trained |\n`); continue; }
    const st = runState(dir);
    try {
      await waitEngine();
      log(`${dsKey}/${sub}/${arm.key}: render from ${dir} (${st.epochsRun} epochs, ${st.reason})`);
      await render(dsKey, dir, wav, ov);
      fs.appendFileSync(readme, `| ${n} | ${arm.key} | ${path.basename(path.dirname(dir))}/${path.basename(dir)} | ${st.epochsRun} epochs, ${st.reason} |\n`);
    } catch (e: any) {
      fs.appendFileSync(readme, `| ${n} | ${arm.key} | ${dir} | ERROR: ${String(e?.message ?? e).slice(0, 120)} |\n`);
    }
  }
  log(`${dsKey}/${sub}: renders done`);
}

// ── phase blind: N songs x (base + chosen arms), letters shuffled per song ──
// Files are song<k>_<letter>.wav; the letter->arm mapping is written ONLY to
// _key/KEY.json (do not open before rating). RATE.md is the scoring sheet.
function shuffled<T>(xs: T[], seed: number): T[] {
  const a = xs.slice();
  let x = seed >>> 0;
  for (let i = a.length - 1; i > 0; i--) {
    x = (x * 1664525 + 1013904223) >>> 0;
    const j = x % (i + 1);
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
async function phaseBlind(dsKey: string, sub: string, armKeys: string[], songs: number, seed: number, rank: number): Promise<void> {
  const outDir = path.join(LST, dsKey, sub);
  const keyDir = path.join(outDir, '_key');
  fs.mkdirSync(keyDir, { recursive: true });
  const keyPath = path.join(keyDir, 'KEY.json');
  const key: Record<string, Record<string, string>> = fs.existsSync(keyPath) ? JSON.parse(fs.readFileSync(keyPath, 'utf8')) : {};
  const items = ['base', ...armKeys];
  const letters = 'ABCDEFGHIJ'.slice(0, items.length).split('');
  const rate = path.join(outDir, 'RATE.md');
  if (!fs.existsSync(rate)) {
    fs.writeFileSync(rate, `# Blind rating — ${dsKey}\n\n${items.length} items per song (one is the base model, the rest are adapters), letters shuffled per song. ` +
      `Same plan per song (seed ${seed}), thirds DiT, 50 steps, no post-processing. Score each 1-10 on quality / expressiveness / likeness, or just rank them. ` +
      `Open _key/KEY.json only when done.\n\n`);
  }
  for (let k = 1; k <= songs; k++) {
    const slug = `song${k}`;
    const lyricsFile = path.join(outDir, `${slug}.lyrics.txt`);
    const captionFile = path.join(outDir, `${slug}.caption.txt`);
    if (!fs.existsSync(lyricsFile) || !fs.existsSync(captionFile)) throw new Error(`${slug}: lyrics/caption files missing in ${outDir}`);
    const src = fs.existsSync(path.join(outDir, `${slug}.SOURCE.txt`)) ? fs.readFileSync(path.join(outDir, `${slug}.SOURCE.txt`), 'utf8').trim() : slug;
    const durMatch = src.match(/(\d+) s\b/);
    const ov: RenderOverride = {
      lyrics: fs.readFileSync(lyricsFile, 'utf8'), caption: fs.readFileSync(captionFile, 'utf8').trim(),
      duration: durMatch ? Number(durMatch[1]) : 240, seed, key: `${dsKey}:${sub}:${slug}`,
    };
    if (!key[slug]) {
      const order = shuffled(items, seed * 7919 + k * 104729);
      key[slug] = {};
      letters.forEach((L, i) => { key[slug][L] = order[i]; });
      fs.writeFileSync(keyPath, JSON.stringify(key, null, 2));
    }
    if (!fs.readFileSync(rate, 'utf8').includes(`## ${slug}`)) {
      fs.appendFileSync(rate, `## ${slug} — ${src}\n\n| file | quality | expressiveness | likeness | notes |\n|---|---|---|---|---|\n` +
        letters.map(L => `| ${slug}_${L}.wav |  |  |  |  |`).join('\n') + '\n\n');
    }
    for (const L of letters) {
      const item = key[slug][L];
      const wav = path.join(outDir, `${slug}_${L}.wav`);
      if (fs.existsSync(wav)) { log(`${slug}_${L}: exists, skipping`); continue; }
      let dir: string | null = null;
      if (item !== 'base') {
        dir = newestRunDir('dit-', `ab-${dsKey}-${item}`);
        if (!dir) throw new Error(`${item}: no adapter`);
      }
      await waitEngine();
      log(`${slug}_${L}: render`);
      await render(dsKey, dir, wav, ov);
    }
    log(`${slug}: done`);
  }
  log(`${dsKey}/${sub}: blind set complete — key sealed in _key/KEY.json`);
}

// ═══════════════════════════════════════════════════════════════════════════
// Crop cap × depth study (night of 2026-09-04). Plain LoRA r128 throughout;
// the only things that move are the crop cap and the DEPTH — which loss rung's
// milestone snapshot ships. One training run per crop cap, trained toward 0.1
// with a snapshot at every 0.1 of ma5, so every depth rung comes from the same
// run (no depth-vs-seed confound). Two sealed sets per dataset:
//   depth: crop 800 at ma5 0.5 / 0.3 / 0.2 / 0.1 + base
//   crop:  550 / 800 / 1500 at ma5 0.3 + base + a hidden repeat of the 800
// The repeat is the same adapter rendered twice with the same seed, so it is
// the listener's own noise floor.
//
//   --phase cd        --datasets mj_dangerous,adtr_whatseparates   (train, then render, per dataset)
//   --phase cd-train  / --phase cd-blind / --phase cd-plan (dry run: songs + run state)
// ═══════════════════════════════════════════════════════════════════════════
const CD_LST    = path.join(ROOT, '_experiments', '_LISTENING', '2026-09-05-dit-crop-depth');
const CD_CROPS  = [550, 800, 1500];
const CD_TARGET = 0.1;
// Epoch caps per crop. Rob's own crop-800 LoKr run on mj_dangerous needed all
// 500 epochs to get near 0.3, and the LoRA A/B arm sat at 0.65 after 200, so
// the depth rungs are chosen from whatever the run actually reaches (see
// cdSetItems). The 1500 cap is lower because its epochs cost ~2x.
const CD_EPOCHS: Record<number, number> = { 550: 800, 800: 800, 1500: 600 };
// Lyric Studio generation ids (the dataset's own album profile). Two per dataset.
const CD_SONGS: Record<string, number[]> = {
  mj_dangerous: [4673, 3811],          // Front Page (134 bpm, 238 s), Girl in the Red Dress (120 bpm, 213 s)
  adtr_whatseparates: [4311, 3983],    // House Money (140 bpm, 196 s), Dead Air (112 bpm, 156 s)
};
// Item grammar: 'base' | 'c<crop>@<rung>' with an optional '#dup' marker.
// The sets are built from the rungs the runs actually reached — cdSetItems().
const cdName = (dsKey: string, crop: number) => `cd-${dsKey}-c${crop}`;
const hasWeights = (d: string) => fs.existsSync(path.join(d, 'adapter_model.safetensors'));

type CdLog = {
  config?: { crop?: number; crop_max?: number; epochs?: number };
  epochs_run?: number; saved_ma5?: number; saved_reason?: string; total_ms?: number;
  milestones?: { loss: number; epoch: number; path: string }[];
  epochs?: { ma5?: number }[];
};
function readCdLog(dir: string): CdLog | null {
  try { return JSON.parse(fs.readFileSync(path.join(dir, 'dit_train_log.json'), 'utf8')) as CdLog; } catch { return null; }
}
// Every run dir ever written for an adapter name (oldest first), including
// one that holds only a log + milestones (a run killed mid-way).
function allRunDirs(name: string): string[] {
  const root = config.aceServer.adapters;
  const out: string[] = [];
  for (const top of fs.readdirSync(root)) {
    if (!top.startsWith('dit-')) continue;
    const nameDir = path.join(root, top, name);
    if (!fs.existsSync(nameDir)) continue;
    for (const run of fs.readdirSync(nameDir)) {
      const d = path.join(nameDir, run);
      if (fs.statSync(d).isDirectory() && fs.existsSync(path.join(d, 'dit_train_log.json'))) out.push(d);
    }
  }
  return out.sort();
}
// Cumulative state of a cd run across its (possibly resumed) run dirs.
function cdRunSummary(name: string, cap = 0) {
  const dirs = allRunDirs(name);
  let epochs = 0, ms = 0, reached = false;
  let saved: number | null = null, crop: number | null = null;
  const rungs = new Map<string, { dir: string; epoch: number }>();
  for (const d of dirs) {
    const jl = readCdLog(d);
    if (!jl) continue;
    const eRun = jl.epochs_run ?? (jl.epochs?.length ?? 0);
    for (const m of jl.milestones ?? []) {
      const mp = path.join(d, m.path);
      if (hasWeights(mp)) rungs.set(m.loss.toFixed(1), { dir: mp, epoch: epochs + m.epoch });
    }
    epochs += eRun;
    ms += jl.total_ms ?? 0;
    if (jl.saved_reason === 'target') reached = true;
    if (typeof jl.saved_ma5 === 'number') saved = jl.saved_ma5;
    if (jl.config?.crop) crop = jl.config.crop;
  }
  const last = dirs.length ? dirs[dirs.length - 1] : null;
  return { dirs, last, epochs, minutes: ms / 60000, reached, saved, crop, rungs, complete: reached || (cap > 0 && epochs >= cap) };
}
// Rungs a run reached at or below 0.6 (the snapshots the prune keeps), shallowest first.
function cdRungsDesc(dsKey: string, crop: number): string[] {
  const s = cdRunSummary(cdName(dsKey, crop));
  const out = [...s.rungs.keys()].map(Number).filter(v => v <= 0.6 + 1e-9).sort((x, y) => y - x).map(v => v.toFixed(1));
  // The final export (best epoch) counts as the deepest rung when it went
  // meaningfully below the last snapshot: a run that ends at 0.31 without
  // crossing 0.3 has a depth the 0.4 snapshot does not represent.
  const deepest = out.length ? Number(out[out.length - 1]) : Infinity;
  if (s.last && hasWeights(s.last) && typeof s.saved === 'number' && s.saved <= deepest - 0.05) out.push('final');
  return out;
}
// depth: the crop-800 run at four depths spread from 0.6 to the deepest rung it
// reached. crop: every crop cap at the deepest rung ALL THREE reached, but no
// deeper than 0.3 (the shipped target), plus a hidden repeat of the 800.
function cdSetItems(dsKey: string): Record<string, string[]> {
  const r800 = cdRungsDesc(dsKey, 800);
  if (r800.length < 2) throw new Error(`${dsKey}: crop-800 run has ${r800.length} rung(s) at or below 0.6 — nothing to compare`);
  const k = Math.min(4, r800.length);
  const depthRungs = Array.from({ length: k }, (_, i) => r800[Math.round(i * (r800.length - 1) / (k - 1))]);
  // 'final' exports sit at different losses per run, so only snapshots count as common.
  const common = CD_CROPS.map(c => new Set(cdRungsDesc(dsKey, c).filter(r => r !== 'final'))).reduce((acc, st) => new Set([...acc].filter(x => st.has(x))));
  const commonAsc = [...common].map(Number).sort((x, y) => x - y);
  if (!commonAsc.length) throw new Error(`${dsKey}: no loss rung common to all three crops`);
  const rung = Math.max(0.3, commonAsc[0]).toFixed(1);
  return {
    depth: ['base', ...depthRungs.map(l => `c800@${l}`)],
    crop:  ['base', ...CD_CROPS.map(c => `c${c}@${rung}`), `c800@${rung}#dup`],
  };
}

async function ensureTensors(dsKey: string): Promise<void> {
  const ds = DATASETS[dsKey];
  const variantKey = SYNTH_MODEL.replace(/\.gguf$/i, '');
  const st = await (await fetch(`${API}/datasets/${ds.id}/preprocess`)).json() as
    { variants?: { variantKey: string; processed: number; total: number; failed: number }[] };
  const v = (st.variants ?? []).find(x => x.variantKey === variantKey);
  if (v && v.processed === v.total && v.failed === 0) { log(`${dsKey}: tensors present (${v.total} samples, ${variantKey})`); return; }
  log(`${dsKey}: no tensors for ${variantKey} — preprocessing`);
  const jobId = await post(`/datasets/${ds.id}/preprocess`, { ditModel: SYNTH_MODEL });
  const r = await waitJob(jobId, `${dsKey}/preprocess`);
  if (r.status !== 'done') throw new Error(`${dsKey}: preprocess ${r.status}: ${r.error ?? ''}`);
  log(`${dsKey}: preprocess done in ${Math.round(r.secs)} s`);
}

async function cdTrain(dsKey: string): Promise<void> {
  const ds = DATASETS[dsKey];
  fs.mkdirSync(path.join(CD_LST, dsKey), { recursive: true });
  const readme = path.join(CD_LST, dsKey, 'README.md');
  if (!fs.existsSync(readme)) {
    fs.writeFileSync(readme,
      `# ${dsKey} — crop cap × depth\n\nPlain LoRA r128 / alpha 256, Prodigy, bf16-f32 mirror, flash attention, ` +
      `target ${CD_TARGET} (ma5), epoch caps ${JSON.stringify(CD_EPOCHS)}, milestone snapshot at every 0.1 of ma5. One run per crop cap; ` +
      `everything else is the train-dit route's default. "rungs" = first epoch whose ma5 reached that value; the blind ` +
      `sets render those snapshots.\n\n| crop cap | crop used | epochs | min | saved ma5 | rungs (epoch) | state |\n|---|---|---|---|---|---|---|\n`);
  }
  await ensureTensors(dsKey);
  for (const crop of CD_CROPS) {
    const name = cdName(dsKey, crop);
    const cap = CD_EPOCHS[crop];
    for (let attempt = 0; attempt < 3; attempt++) {
      const s = cdRunSummary(name, cap);
      if (s.complete) { log(`${name}: complete (${s.epochs} epochs, saved ${s.saved}, ${s.reached ? 'target' : 'cap'})`); break; }
      const remaining = Math.max(1, cap - s.epochs);
      const body: Record<string, unknown> = {
        adapterName: name, adapterType: 'lora', rank: 128, alpha: 256,
        cropMax: crop, targetLoss: CD_TARGET, epochs: remaining,
        milestoneStep: 0.1, milestoneKeep: 8, attnBackend: 'flash',
      };
      if (s.last && !hasWeights(s.last)) {
        // Killed mid-run: no top-level export to continue from, so the route's
        // 'latest' would skip this dir. Continue from the deepest snapshot.
        const deepest = [...s.rungs.entries()].sort((x, y) => Number(x[0]) - Number(y[0]))[0];
        body.initAdapter = deepest ? deepest[1].dir : '';
      }
      log(`${name}: ${s.epochs ? `resuming after ${s.epochs} epochs` : 'from scratch'}, ${remaining} epochs ${JSON.stringify(body)}`);
      const jobId = await post(`/datasets/${ds.id}/train-dit`, body);
      const r = await waitJob(jobId, name);
      log(`${name}: ${r.status} after ${Math.round(r.secs)} s${r.error ? ` — ${r.error}` : ''}`);
      if (r.status === 'done') break;
    }
    const s = cdRunSummary(name, cap);
    const rungs = [...s.rungs.entries()].sort((x, y) => Number(y[0]) - Number(x[0])).map(([l, v]) => `${l}@${v.epoch}`).join(' ');
    fs.appendFileSync(readme, `| ${crop} | ${s.crop ?? '?'} | ${s.epochs} | ${s.minutes.toFixed(0)} | ${s.saved?.toFixed(3) ?? '?'} | ${rungs} | ${s.reached ? 'target' : s.complete ? 'epoch cap' : 'INCOMPLETE'} |\n`);
  }
}

type CdGen = { id: number; title: string; caption: string; lyrics: string; bpm: number; key: string; duration: number };
async function cdSongs(dsKey: string): Promise<{ slug: string; gen: CdGen; src: string }[]> {
  const ds = DATASETS[dsKey];
  const ids = CD_SONGS[dsKey];
  if (!ids) throw new Error(`${dsKey}: no songs configured`);
  const r = await (await fetch(`${API}/datasets/${ds.id}/ls-generations`)).json() as { artist: string; album: string; generations: CdGen[] };
  const dir = path.join(CD_LST, dsKey);
  fs.mkdirSync(dir, { recursive: true });
  return ids.map((id, i) => {
    const g = r.generations.find(x => x.id === id);
    if (!g) throw new Error(`${dsKey}: Lyric Studio generation #${id} not found`);
    if (!g.caption || !g.lyrics) throw new Error(`#${id} lacks caption/lyrics`);
    const slug = `song${i + 1}`;
    const src = `Lyric Studio #${g.id} "${g.title}" — ${r.artist} / ${r.album}, bpm ${g.bpm}, ${g.key}, ${g.duration} s`;
    for (const [f, c] of [[`${slug}.lyrics.txt`, g.lyrics], [`${slug}.caption.txt`, g.caption], [`${slug}.SOURCE.txt`, src + '\n']] as const) {
      const fp = path.join(dir, f);
      if (!fs.existsSync(fp)) fs.writeFileSync(fp, c);
    }
    return { slug, gen: g, src };
  });
}
function cdResolve(dsKey: string, item: string): { dir: string | null; note: string } {
  if (item === 'base') return { dir: null, note: 'base model, no adapter' };
  const m = item.replace(/#.*$/, '').match(/^c(\d+)@([\d.]+|final)$/);
  if (!m) throw new Error(`bad item ${item}`);
  const crop = Number(m[1]);
  const s = cdRunSummary(cdName(dsKey, crop));
  if (m[2] === 'final') {
    if (!s.last || !hasWeights(s.last)) throw new Error(`${item}: no final export`);
    return { dir: s.last, note: `crop cap ${crop} (used ${s.crop}), final export = best epoch, ma5 ${s.saved?.toFixed(3)} after ${s.epochs} epochs` };
  }
  const rung = Number(m[2]).toFixed(1);
  const hit = s.rungs.get(rung);
  if (hit) return { dir: hit.dir, note: `crop cap ${crop} (used ${s.crop}), snapshot at first ma5 <= ${rung} (epoch ${hit.epoch})` };
  // Rung never reached: ship the run's final export (its best epoch) and say so in the key.
  if (s.last && hasWeights(s.last)) {
    return { dir: s.last, note: `crop cap ${crop}: rung ${rung} NOT reached — final export, ma5 ${s.saved?.toFixed(3)} after ${s.epochs} epochs` };
  }
  throw new Error(`${item}: no adapter for rung ${rung}`);
}
function shuffleSecure<T>(xs: T[]): T[] {
  const a = xs.slice();
  for (let i = a.length - 1; i > 0; i--) { const j = crypto.randomInt(i + 1); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}
type CdKey = Record<string, Record<string, Record<string, { item: string; note: string }>>>;
async function cdBlind(dsKey: string, seed: number): Promise<void> {
  const base = path.join(CD_LST, dsKey);
  const keyDir = path.join(base, '_key');
  fs.mkdirSync(keyDir, { recursive: true });
  const keyPath = path.join(keyDir, 'KEY.json');
  const key: CdKey = fs.existsSync(keyPath) ? JSON.parse(fs.readFileSync(keyPath, 'utf8')) as CdKey : {};
  const saveKey = () => fs.writeFileSync(keyPath, JSON.stringify(key, null, 2));
  const songs = await cdSongs(dsKey);
  const rate = path.join(base, 'RATE.md');
  if (!fs.existsSync(rate)) {
    fs.writeFileSync(rate,
      `# Blind rating — ${dsKey} — crop cap × depth\n\nEvery adapter here is plain LoRA r128 on this dataset; only the training crop cap ` +
      `and the depth (which loss the shipped snapshot was taken at) differ. Two sets, ${songs.length} songs each, 5 files per song, ` +
      `one of which is the base model. Letters are shuffled per song and per set. Same LM plan per song (seed ${seed}), thirds DiT, ` +
      `50 steps, guidance 20, no post-processing.\n\n- depth/: the 800-frame crop at four depths\n- crop/: three crop caps at one depth\n\n` +
      `Score 1-10 on quality / expressiveness / likeness, or rank. Open _key/KEY.json only when done.\n\n`);
  }
  const sets = cdSetItems(dsKey);
  log(`${dsKey}: sets ${JSON.stringify(sets)}`);
  for (const [set, items] of Object.entries(sets)) {
    const letters = 'ABCDEFGH'.slice(0, items.length).split('');
    const setDir = path.join(base, set);
    fs.mkdirSync(setDir, { recursive: true });
    for (const sng of songs) {
      key[set] ??= {};
      if (!key[set][sng.slug]) {
        const order = shuffleSecure(items);
        key[set][sng.slug] = {};
        letters.forEach((L, i) => { key[set][sng.slug][L] = { item: order[i], note: '' }; });
        saveKey();
      }
      const hdr = `## ${set} — ${sng.slug}`;
      if (!fs.readFileSync(rate, 'utf8').includes(hdr)) {
        fs.appendFileSync(rate, `${hdr} — ${sng.src}\n\n| file | quality | expressiveness | likeness | notes |\n|---|---|---|---|---|\n` +
          letters.map(L => `| ${set}/${sng.slug}_${L}.wav |  |  |  |  |`).join('\n') + '\n\n');
      }
      const ov: RenderOverride = {
        lyrics: sng.gen.lyrics, caption: sng.gen.caption.trim(), duration: sng.gen.duration || 240, seed, key: `${dsKey}:cd:${sng.slug}`,
      };
      for (const L of letters) {
        const wav = path.join(setDir, `${sng.slug}_${L}.wav`);
        if (fs.existsSync(wav)) { log(`${dsKey} ${set}/${sng.slug}_${L}: exists, skipping`); continue; }
        const entry = key[set][sng.slug][L];
        const res = cdResolve(dsKey, entry.item);
        entry.note = res.note;
        saveKey();
        await waitEngine();
        log(`${dsKey} ${set}/${sng.slug}_${L}: render (${entry.item})`);
        await render(dsKey, res.dir, wav, ov);
      }
    }
  }
  log(`${dsKey}: blind sets complete — key sealed in _key/KEY.json`);
}
async function cdPlan(dsKey: string): Promise<void> {
  const songs = await cdSongs(dsKey);
  for (const s of songs) log(`${dsKey} ${s.slug}: ${s.src} (caption ${s.gen.caption.length} chars, lyrics ${s.gen.lyrics.length} chars)`);
  for (const c of CD_CROPS) {
    const s = cdRunSummary(cdName(dsKey, c), CD_EPOCHS[c]);
    log(`${cdName(dsKey, c)}: ${s.dirs.length} run dir(s), ${s.epochs} epochs, saved ${s.saved ?? '-'}, rungs [${[...s.rungs.keys()].join(' ')}], ${s.complete ? 'complete' : 'pending'}`);
  }
  try { log(`${dsKey}: sets ${JSON.stringify(cdSetItems(dsKey))}`); } catch (e: any) { log(`${dsKey}: sets not yet derivable — ${e.message}`); }
}

async function main() {
  const a = args();
  const phase = a.get('phase') || 'all';  // lm | dit | all | render | blind | cd | cd-train | cd-blind | cd-plan
  const dsKeys = (a.get('datasets') || 'mj_dangerous,dio_holydiver,carpenterbrut_trilogy').split(',').map(s => s.trim()).filter(Boolean);
  const target = Number(a.get('target') || 0.5);
  const epochs = Number(a.get('epochs') || 200);
  const rank = Number(a.get('rank') || 128);
  for (const k of dsKeys) if (!DATASETS[k]) throw new Error(`unknown dataset ${k}`);
  const up = await fetch(`${API}/jobs`).then(r => r.ok).catch(() => false);
  if (!up) throw new Error('Node server not reachable on :3001 — start dev.bat');
  if (phase === 'lm' || phase === 'all') await phaseLm();
  const only = (a.get('arms') || '').split(',').map(x => x.trim()).filter(Boolean);
  if (phase === 'dit' || phase === 'all') await phaseDit(dsKeys, target, epochs, rank, only);
  if (phase === 'blind') {
    // --arms rslora,lora,hira,pissa,dora --songs 3 [--sub blind] [--seed n]
    const armKeys = (a.get('arms') || 'rslora,lora,hira,pissa,dora').split(',').map(x => x.trim()).filter(Boolean);
    await phaseBlind(dsKeys[0], a.get('sub') || 'blind', armKeys, Number(a.get('songs') || 3), Number(a.get('seed') || 20260904), rank);
  }
  if (phase === 'cd' || phase === 'cd-train' || phase === 'cd-blind') {
    for (const k of dsKeys) {
      if (phase !== 'cd-blind') await cdTrain(k);
      if (phase !== 'cd-train') await cdBlind(k, Number(a.get('seed') || 20260904));
    }
  }
  if (phase === 'cd-plan') for (const k of dsKeys) await cdPlan(k);
  if (phase === 'render') {
    // --lyrics-file <path> --caption-file <path> [--duration s] [--seed n] [--sub full]
    const lyricsFile = a.get('lyrics-file'); const captionFile = a.get('caption-file');
    if (!lyricsFile || !captionFile) throw new Error('--phase render needs --lyrics-file and --caption-file');
    const ov: RenderOverride = {
      lyrics: fs.readFileSync(lyricsFile, 'utf8'), caption: fs.readFileSync(captionFile, 'utf8').trim(),
      duration: Number(a.get('duration') || 240), seed: Number(a.get('seed') || 20260904), key: `${dsKeys[0]}:${a.get('sub') || 'full'}`,
    };
    await phaseRender(dsKeys[0], a.get('sub') || 'full', ov, rank, only, a.get('base') !== '0');
  }
  log('done');
}

main().catch(e => { console.error(String(e?.stack || e)); process.exit(1); });
