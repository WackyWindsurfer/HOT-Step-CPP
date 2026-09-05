/**
 * mm3-crop-ab.ts — blind A/B of an MM3 LM adapter's crop regime, driven
 * THROUGH the running app's HTTP API and engine.
 *
 * Two MM3 planner-LM LoRA/LoKr adapters, trained identically except the crop
 * regime, then a blind-rated render set:
 *
 *   A "recipe":      maxFrames 750,  prefixFrames 4096, attnBackend 'exact'
 *                     — the shipped default (MM3_LM_DEFAULTS in mm3Train.ts).
 *   B "whole-track":  maxFrames = this dataset's longest track (frames),
 *                     prefixFrames 0, attnBackend 'flash'.
 *
 * Everything else (rank, alpha, optimizer, adapter type, crop-mode shares...)
 * is left at the mm3-train-lm route's own defaults on purpose: those defaults
 * ARE the recipe under test, and setting them here would just be a second
 * copy of MM3_LM_DEFAULTS that could drift from it.
 *
 *   npx tsx scripts/mm3-crop-ab.ts --phase plan  --dataset alk3_crimson
 *   npx tsx scripts/mm3-crop-ab.ts --phase train --dataset alk3_crimson
 *   npx tsx scripts/mm3-crop-ab.ts --phase blind --dataset alk3_crimson --songs 3
 *   npx tsx scripts/mm3-crop-ab.ts --phase all   --dataset alk3_crimson
 *
 * plan: resolves the dataset, the two logical run names, whatever already
 *   exists on disk for them, the chosen prompts, and the render requests it
 *   WOULD send — no GPU, no training, no rendering. Needs the app only for the
 *   dataset lookup and the prompt sources; if the app is not reachable it says
 *   so and stops there (everything before that point — the logical run names —
 *   needs nothing but the dataset slug). Default phase, so running the script
 *   with no --phase never starts a multi-hour job by accident.
 *
 * train: posts one mm3-train-lm run per arm (or resumes a partial one, or
 *   skips a complete one) and appends a row to the listening folder's
 *   README.md: steps, wall minutes, final loss, eval loss, ms/step, peak VRAM
 *   — all read from the run's own train-log.jsonl.
 *
 * blind: renders base + A + B + a hidden repeat of A for N songs (default 3),
 *   letters shuffled per song with crypto.randomInt, key sealed in
 *   _key/KEY.json, RATE.md scoring table. Prompts come from the dataset's
 *   linked Lyric Studio album: the MM3 Structured Caption
 *   (`generations.caption_mm3`) when a generation carries one, otherwise the
 *   dataset's own per-track `<stem>.mm3.txt` captions (held-out tracks when
 *   there are enough, else the first ones) paired with Lyric Studio lyrics —
 *   the README/SOURCE files say which was used. Renders go straight to the
 *   engine's /mm3/synth (same job queue ACE uses — aceClient.pollJob/
 *   getJobResult work unchanged), pinning the LM to q8_0 first: an MM3 LM
 *   adapter renders garbled on the f16 base (mm3-lm-adapter-training skill).
 *
 * Everything is re-runnable: a wav that already exists is skipped, a complete
 * training run is reused, a partial one is resumed.
 *
 * Needs the Node server on :3001 (dev.bat) for every phase except a fully
 * offline `plan`. Run from server/.
 */
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

import { aceClient } from '../src/services/aceClient.js';
import { config } from '../src/config.js';
import {
  mm3Synth, mm3SelectModel, type Mm3SynthRequest,
} from '../src/services/backends/minimax/client.js';
import { MM3_LM_ADAPTER_DEFAULT_SCALES } from '../src/services/backends/minimax/lmAdapter.js';
import { applyMm3Trigger, readMm3AdapterTrigger } from '../src/services/backends/minimax/trigger.js';
import {
  listMm3Runs, resumeOptionsFor, readMm3Run, type Mm3RunSummary,
} from '../src/services/training/mm3Runs.js';
import type { ResolvedMm3TrainLmOptions } from '../src/services/training/mm3Train.js';

const API        = 'http://127.0.0.1:3001/api/training';
const LIREEK_API = 'http://127.0.0.1:3001/api/lireek';
const ROOT       = path.resolve(process.cwd(), '..');  // run from server/
const STAMP      = '2026-09-06';
const LST_ROOT   = path.join(ROOT, '_experiments', '_LISTENING', `${STAMP}-mm3-crop-vs-prefix`);

// MM3's fixed frame rate throughout the pipeline (mm3-backend skill: "25 fps
// frames"). Only used here to turn a track's duration into a frame count for
// the whole-track arm; the engine derives max_frames from seconds itself at
// render time.
const MM3_FPS = 25;

const TRAIN_STEPS     = 500;
const TRAIN_SAVE_EVERY = 50;
const TRAIN_SEED      = 42;   // MM3_LM_DEFAULTS.seed — sent explicitly so the recipe is legible here too.

const RENDER_STEPS        = 30;  // the checkpoint's own default (mm3.flow.steps) — sent explicitly for the record.
const RENDER_DURATION_SEC = 75;  // fixed for every song/arm, within the requested 60-90 s band.
const RENDER_SEED         = 20260906;

function log(m: string) { console.log(`[${new Date().toISOString().slice(11, 19)}] ${m}`); }
function args() {
  const a = new Map<string, string>();
  for (let i = 2; i < process.argv.length; i += 2) a.set(process.argv[i].replace(/^--/, ''), process.argv[i + 1] ?? '');
  return a;
}

// ── job plumbing (copied from param-ab.ts, not imported — that script's
// bottom-level `main().catch(...)` would execute on import) ────────────────

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
function shuffleSecure<T>(xs: T[]): T[] {
  const a = xs.slice();
  for (let i = a.length - 1; i > 0; i--) { const j = crypto.randomInt(i + 1); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}
function writeIfAbsent(file: string, content: string): void {
  if (!fs.existsSync(file)) fs.writeFileSync(file, content);
}

// ── dataset lookup (through the app — no direct DB/fs access to the dataset
// itself, only to the training-run directories below, which have no API) ──

interface DatasetInfo { id: string; slug: string; name: string; sourceDir: string; }

async function resolveDataset(slug: string): Promise<DatasetInfo> {
  const r = await fetch(`${API}/datasets`);
  if (!r.ok) throw new Error(`GET /datasets failed: HTTP ${r.status}`);
  const j = await r.json() as { datasets?: Array<{ id: string; slug: string; name: string; sourceDir: string }> };
  const ds = (j.datasets ?? []).find(d => d.slug === slug);
  if (!ds) throw new Error(`no dataset with slug "${slug}"`);
  return { id: ds.id, slug: ds.slug, name: ds.name, sourceDir: ds.sourceDir };
}

/** The whole-track arm's maxFrames: this dataset's longest usable track,
 *  clamped to [64, 9000].
 *
 *  The clamp matters more than it looks: the mm3-train-lm route's own bounds
 *  check (`num()` in routes/training.ts) does NOT clamp an out-of-range
 *  maxFrames — a value outside [64, 9000] silently falls back to the route's
 *  DEFAULT (750) instead, which would quietly turn the whole-track arm into a
 *  second copy of the recipe arm. Clamping here, before the value is ever
 *  sent, is what avoids that. */
async function computeWholeTrackFrames(ds: DatasetInfo): Promise<number> {
  const r = await fetch(`${API}/datasets/${ds.id}`);
  if (!r.ok) throw new Error(`GET dataset detail failed: HTTP ${r.status}`);
  const detail = await r.json() as { samples?: Array<{ duration: number; excluded?: boolean; fileMissing?: boolean }> };
  const durations = (detail.samples ?? [])
    .filter(s => !s.excluded && !s.fileMissing && Number.isFinite(s.duration) && s.duration > 0)
    .map(s => s.duration);
  if (!durations.length) throw new Error(`${ds.slug}: no samples with a known duration — build the dataset first`);
  const longestSec = Math.max(...durations);
  const frames = Math.round(longestSec * MM3_FPS);
  const clamped = Math.min(9000, Math.max(64, frames));
  if (clamped !== frames) {
    log(`${ds.slug}: longest track is ${longestSec.toFixed(1)} s = ${frames} frames at ${MM3_FPS} fps, clamped to `
      + `${clamped} (mm3-train-lm's maxFrames bound is [64, 9000] and falls back to its DEFAULT — not a clamp — `
      + 'outside that range, so this script clamps first)');
  }
  return clamped;
}

// ── the two arms ─────────────────────────────────────────────────────────
//
// mm3-train-lm has no adapter-name field (unlike train-dit's adapterName): a
// run directory is always `<slug>-<timestamp>` (mm3RunName in mm3Train.ts).
// So the "logical" names below are this script's own bookkeeping labels, not
// literal directory names — an arm is found again by matching a run's
// RECORDED attnBackend / prefixFrames / maxFrames (via resumeOptionsFor,
// which reads the manifest or reconstructs from the log), never by name.

type ArmKey = 'recipe' | 'wholetrack';

interface ArmDef {
  key: ArmKey;
  label: string;
  attnBackend: 'exact' | 'flash';
  prefixFrames: number;
  /** Fixed for 'recipe'. 'wholetrack' derives its maxFrames from the dataset
   *  at train/plan time and is therefore not part of the match signature —
   *  attnBackend + prefixFrames alone already distinguish the two arms. */
  maxFrames?: number;
  note: string;
}

function armDefs(dsSlug: string): ArmDef[] {
  return [
    {
      key: 'recipe', label: `ab-${dsSlug}-crop750-prefix`,
      attnBackend: 'exact', prefixFrames: 4096, maxFrames: 750,
      note: 'the shipped default crop regime: 750-frame crop, 4096-frame frozen prefix, exact attention',
    },
    {
      key: 'wholetrack', label: `ab-${dsSlug}-wholetrack-flash`,
      attnBackend: 'flash', prefixFrames: 0,
      note: "whole-track crop (this dataset's longest track, clamped to [64, 9000]), no frozen prefix, flash attention",
    },
  ];
}

function armMatches(opts: ResolvedMm3TrainLmOptions, arm: ArmDef): boolean {
  if (opts.attnBackend !== arm.attnBackend) return false;
  if (opts.prefixFrames !== arm.prefixFrames) return false;
  if (arm.maxFrames !== undefined && opts.maxFrames !== arm.maxFrames) return false;
  return true;
}

function armTrainBody(arm: ArmDef, wholeTrackFrames: number): Record<string, unknown> {
  return {
    steps: TRAIN_STEPS,
    saveEvery: TRAIN_SAVE_EVERY,
    stopMode: 'steps',
    seed: TRAIN_SEED,
    maxFrames: arm.key === 'recipe' ? (arm.maxFrames ?? 750) : wholeTrackFrames,
    prefixFrames: arm.prefixFrames,
    attnBackend: arm.attnBackend,
    // rank / alpha / optimizer / adapterType / lokr* / cropMode / cropStartFrac
    // / cropEndFrac / depthLossWeight / holdout / ... are all left OUT on
    // purpose — the route's own defaults (MM3_LM_DEFAULTS) are the recipe
    // under test. previews are left off too (no `preview` field -> the
    // runner emits --no-pause).
  };
}

/** The newest run on disk (by updatedAt) whose recorded options match this
 *  arm's signature, for this dataset. Needs a real datasetId — offline (no
 *  app) this cannot be done reliably, because a manifest-carrying run is
 *  matched by datasetId, not by directory name; see phasePlan. */
function findArmRun(datasetId: string, slug: string, arm: ArmDef):
    { dir: string; runName: string; summary: Mm3RunSummary } | null {
  const runs = listMm3Runs(datasetId, slug);
  let best: { dir: string; runName: string; summary: Mm3RunSummary } | null = null;
  for (const r of runs) {
    const opts = resumeOptionsFor(r.dir);
    if (!opts || !armMatches(opts, arm)) continue;
    if (!best || r.updatedAt > best.summary.updatedAt) best = { dir: r.dir, runName: r.runName, summary: r };
  }
  return best;
}

/** The FINAL checkpoint's adapter file (highest step), lokr or lora — mirrors
 *  mm3Runs.ts's own checkpointsIn(), which looks for either. */
function pickFinalCheckpoint(dir: string): { file: string; step: number } | null {
  const run = readMm3Run(dir);
  if (!run || !run.checkpoints.length) return null;
  const last = run.checkpoints[run.checkpoints.length - 1];
  const lokr = path.join(last.dir, 'lokr_weights.safetensors');
  const lora = path.join(last.dir, 'adapter_model.safetensors');
  if (fs.existsSync(lokr)) return { file: lokr, step: last.step };
  if (fs.existsSync(lora)) return { file: lora, step: last.step };
  return null;
}

async function resolveArmCheckpoints(ds: DatasetInfo): Promise<Record<ArmKey, string | null>> {
  const out: Record<ArmKey, string | null> = { recipe: null, wholetrack: null };
  for (const arm of armDefs(ds.slug)) {
    const found = findArmRun(ds.id, ds.slug, arm);
    const ckpt = found ? pickFinalCheckpoint(found.dir) : null;
    out[arm.key] = ckpt ? ckpt.file : null;
  }
  return out;
}

// ── phase train ──────────────────────────────────────────────────────────

interface Mm3TrainLogStats { finalLoss?: number; evalLoss?: number; msPerStep?: number; peakVramMb?: number; }

/** steps / final loss / eval loss / ms-per-step / peak VRAM, read straight off
 *  the run's own train-log.jsonl (init/step/eval/vram/milestone JSONL —
 *  mm3TrainRunner.ts's relay() documents the event shapes). Deliberately a
 *  small local scan rather than importing mm3Runs.ts's readLog(), which is
 *  not exported and reads more than this needs. */
function readMm3TrainLogStats(dir: string): Mm3TrainLogStats {
  const out: Mm3TrainLogStats = {};
  let text = '';
  try { text = fs.readFileSync(path.join(dir, 'train-log.jsonl'), 'utf-8'); } catch { return out; }
  let msSum = 0, msCount = 0, peakVram = 0;
  for (const line of text.split(/\r?\n/)) {
    if (!line) continue;
    let ev: Record<string, unknown>;
    try { ev = JSON.parse(line) as Record<string, unknown>; } catch { continue; }
    const type = typeof ev.type === 'string' ? ev.type : '';
    if (type === 'step') {
      const loss = Number(ev.loss);
      if (Number.isFinite(loss)) out.finalLoss = loss;
      const ms = Number(ev.stepMs);
      if (Number.isFinite(ms)) { msSum += ms; msCount++; }
    } else if (type === 'eval') {
      const loss = Number(ev.loss);
      if (Number.isFinite(loss)) out.evalLoss = loss;
    } else if (type === 'vram') {
      const used = Number(ev.usedMb);
      if (Number.isFinite(used)) peakVram = Math.max(peakVram, used);
    }
  }
  if (msCount > 0) out.msPerStep = msSum / msCount;
  if (peakVram > 0) out.peakVramMb = peakVram;
  return out;
}

function ensureTrainReadme(lst: string): string {
  const readme = path.join(lst, 'README.md');
  if (!fs.existsSync(readme)) {
    fs.writeFileSync(readme,
      '# MM3 crop cap vs whole-track prefix — training\n\n'
      + 'Two MM3 LM adapter runs per dataset, through the app\'s mm3-train-lm route, identical except the crop regime. '
      + `Both: steps ${TRAIN_STEPS}, saveEvery ${TRAIN_SAVE_EVERY}, stopMode steps (no target loss), seed ${TRAIN_SEED}, `
      + 'previews off, and the route\'s own defaults for everything else — rank, alpha, optimizer and adapter type are '
      + 'left untouched, because the shipped recipe\'s defaults are what is under test.\n\n'
      + '- recipe (A): maxFrames 750, prefixFrames 4096, attnBackend exact.\n'
      + `- wholetrack (B): maxFrames = this dataset's longest track in frames at ${MM3_FPS} fps, clamped to [64, 9000], `
      + 'prefixFrames 0, attnBackend flash.\n\n'
      + 'mm3-train-lm has no adapter-name field: the run directory is always `<slug>-<timestamp>`. This script tracks '
      + 'each arm by matching a run\'s recorded attnBackend / prefixFrames / maxFrames rather than by name.\n\n'
      + 'The mm3-train-lm route\'s own bounds check (`num()` in routes/training.ts) does not clamp an out-of-range '
      + 'maxFrames or prefixFrames — it silently falls back to the route\'s DEFAULT instead. This script clamps the '
      + 'whole-track frame count to [64, 9000] itself before sending it, to avoid that trap.\n\n'
      + '| dataset | arm | run dir | steps | wall min | final loss | eval loss | ms/step | peak VRAM MB | note |\n'
      + '|---|---|---|---|---|---|---|---|---|---|\n');
  }
  return readme;
}

/** Trains one arm: skip if a matching run already reached TRAIN_STEPS,
 *  resume if a matching run is partial and resumable, otherwise start fresh.
 *  Returns the run directory, or null on failure. */
async function trainArm(ds: DatasetInfo, arm: ArmDef, wholeTrackFrames: number): Promise<string | null> {
  const found = findArmRun(ds.id, ds.slug, arm);
  if (found) {
    if (found.summary.outcome === 'completed' || found.summary.outcome === 'target-reached'
        || found.summary.lastStep >= TRAIN_STEPS) {
      log(`${arm.label}: already trained at ${found.runName} (${found.summary.lastStep} steps, ${found.summary.outcome}) — skipping`);
      return found.dir;
    }
    if (found.summary.resume) {
      log(`${arm.label}: resuming ${found.runName} from step ${found.summary.resume.step} toward ${TRAIN_STEPS}`);
      const jobId = await post(`/datasets/${ds.id}/mm3-resume-lm`, { runName: found.runName, steps: TRAIN_STEPS });
      const r = await waitJob(jobId, arm.label);
      log(`${arm.label}: ${r.status} in ${(r.secs / 60).toFixed(1)} min ${r.error ?? ''}`);
      if (r.status !== 'done') return null;
      await waitEngine();
      return found.dir;
    }
    log(`${arm.label}: existing run ${found.runName} is not resumable (outcome ${found.summary.outcome}, no saved `
      + 'state) — starting a fresh run instead');
  }
  const body = armTrainBody(arm, wholeTrackFrames);
  log(`${arm.label}: starting fresh — ${arm.note} — ${JSON.stringify(body)}`);
  const jobId = await post(`/datasets/${ds.id}/mm3-train-lm`, body);
  const r = await waitJob(jobId, arm.label);
  log(`${arm.label}: ${r.status} in ${(r.secs / 60).toFixed(1)} min ${r.error ?? ''}`);
  if (r.status !== 'done') return null;
  await waitEngine();
  const after = findArmRun(ds.id, ds.slug, arm);
  if (!after) { log(`${arm.label}: training reported done but no matching run was found on disk`); return null; }
  return after.dir;
}

async function phaseTrain(dsSlug: string): Promise<void> {
  const ds = await resolveDataset(dsSlug);
  const wholeTrackFrames = await computeWholeTrackFrames(ds);
  log(`${dsSlug}: whole-track maxFrames = ${wholeTrackFrames} (${MM3_FPS} fps, clamped to [64, 9000])`);
  const lst = path.join(LST_ROOT, dsSlug);
  fs.mkdirSync(lst, { recursive: true });
  const readme = ensureTrainReadme(lst);
  for (const arm of armDefs(dsSlug)) {
    const dir = await trainArm(ds, arm, wholeTrackFrames);
    const run = dir ? readMm3Run(dir) : null;
    const stats = dir ? readMm3TrainLogStats(dir) : null;
    const wallMin = run && run.startedAt ? (run.updatedAt - run.startedAt) / 60000 : undefined;
    const cells = [
      dsSlug, arm.key, dir ? path.basename(dir) : '—',
      run ? String(run.lastStep) : '—',
      wallMin !== undefined ? wallMin.toFixed(1) : '—',
      stats?.finalLoss !== undefined ? stats.finalLoss.toFixed(4) : '—',
      stats?.evalLoss !== undefined ? stats.evalLoss.toFixed(4) : '—',
      stats?.msPerStep !== undefined ? stats.msPerStep.toFixed(0) : '—',
      stats?.peakVramMb !== undefined ? String(stats.peakVramMb) : '—',
      arm.note,
    ];
    fs.appendFileSync(readme, `| ${cells.join(' | ')} |\n`);
    log(cells.slice(0, -1).join(' | '));
  }
}

// ── prompts (through the app) ───────────────────────────────────────────

interface LsGeneration { id: number; title: string; caption: string; lyrics: string; bpm: number; key: string; duration: number; }
interface LsGenerationsResponse { lyricsSetId: number; artist: string; album: string; generations: LsGeneration[]; }
interface LireekGenerationRow { id?: unknown; caption_mm3?: unknown; [k: string]: unknown; }

interface SongPrompt { slug: string; caption: string; lyrics: string; source: string; }

function mm3CaptionPath(sourceDir: string, filename: string): string {
  const stem = filename.replace(/\.[^.]*$/, '');
  return path.join(sourceDir, `${stem}.mm3.txt`);
}

/** N prompts for the dataset's linked Lyric Studio album.
 *
 *  Preferred source: a generation's own MM3 Structured Caption
 *  (`generations.caption_mm3` — GET .../ls-generations does not expose this
 *  column, so it is fetched separately from GET /api/lireek/generations and
 *  joined by id).
 *
 *  Fallback, used and NAMED in the source string when fewer than N
 *  generations carry one: the dataset's own per-track `<stem>.mm3.txt`
 *  captions (preferring held-out tracks — the trainer's own last-15%-of-rows
 *  rule, mirrored here — falling back to the first ones on a small dataset),
 *  paired with a Lyric Studio generation's lyrics. */
async function loadSongs(ds: DatasetInfo, n: number): Promise<SongPrompt[]> {
  const lsRes = await fetch(`${API}/datasets/${ds.id}/ls-generations`);
  if (!lsRes.ok) throw new Error(`GET ls-generations failed: HTTP ${lsRes.status}`);
  const ls = await lsRes.json() as LsGenerationsResponse;
  if (!ls.generations?.length) {
    throw new Error(`${ds.slug}: no Lyric Studio generations linked to this dataset's album — generate some in `
      + 'Lyric Studio first');
  }

  const mm3ById = new Map<number, string>();
  if (ls.lyricsSetId > 0) {
    const rawRes = await fetch(`${LIREEK_API}/generations?lyrics_set_id=${ls.lyricsSetId}`);
    if (rawRes.ok) {
      const rows = await rawRes.json() as LireekGenerationRow[];
      for (const row of rows) {
        const id = Number(row.id);
        const cm = typeof row.caption_mm3 === 'string' ? row.caption_mm3.trim() : '';
        if (Number.isFinite(id) && cm) mm3ById.set(id, cm);
      }
    }
  }

  const withMm3 = ls.generations.filter(g => mm3ById.has(g.id));
  if (withMm3.length >= n) {
    return withMm3.slice(0, n).map((g, i) => ({
      slug: `song${i + 1}`,
      caption: (mm3ById.get(g.id) ?? '').trim(),
      lyrics: g.lyrics,
      source: `Lyric Studio #${g.id} "${g.title}" MM3 caption (caption_mm3) — ${ls.artist} / ${ls.album}, ${g.duration} s`,
    }));
  }

  log(`${ds.slug}: only ${withMm3.length} of ${ls.generations.length} Lyric Studio generations carry an MM3 caption `
    + '(caption_mm3) — falling back to per-track .mm3.txt captions with Lyric Studio lyrics');
  const dsRes = await fetch(`${API}/datasets/${ds.id}`);
  if (!dsRes.ok) throw new Error(`GET dataset detail failed: HTTP ${dsRes.status}`);
  const detail = await dsRes.json() as {
    sourceDir: string; samples?: Array<{ filename: string; excluded?: boolean; fileMissing?: boolean }>;
  };
  const rows = (detail.samples ?? []).filter(s => !s.excluded && !s.fileMissing);
  const withCaption = rows.filter(s => fs.existsSync(mm3CaptionPath(detail.sourceDir, s.filename)));
  if (!withCaption.length) {
    throw new Error(`${ds.slug}: no per-track .mm3.txt captions in ${detail.sourceDir} either — generate MM3 `
      + 'captions in the Enhance panel (MOSS or Gemini) first, or link an album with MM3 captions in Lyric Studio');
  }
  // Mirrors mm3Preview.ts's holdoutRows(): the last ceil(0.15*n) rows, capped
  // at a quarter, disabled below 6 rows.
  const heldOutN = withCaption.length >= 6
    ? Math.max(1, Math.min(Math.floor(withCaption.length / 4), Math.ceil(0.15 * withCaption.length)))
    : 0;
  const pool = heldOutN > 0 ? withCaption.slice(-heldOutN) : withCaption;
  const out: SongPrompt[] = [];
  for (let i = 0; i < n; i++) {
    const track = pool[i % pool.length];
    const gen = ls.generations[i % ls.generations.length];
    const captionPath = mm3CaptionPath(detail.sourceDir, track.filename);
    out.push({
      slug: `song${i + 1}`,
      caption: fs.readFileSync(captionPath, 'utf-8').trim(),
      lyrics: gen.lyrics,
      source: `FALLBACK: per-track caption ${path.basename(captionPath)} (${heldOutN > 0 ? 'held-out' : 'first'} `
        + `track) + Lyric Studio #${gen.id} "${gen.title}" lyrics — ${ls.artist} / ${ls.album}`,
    });
  }
  return out;
}

// ── rendering (engine directly — same job queue ACE uses) ──────────────────

type Mm3RenderRequest = Mm3SynthRequest & {
  lm_adapter?: string;
  lm_adapter_mode?: 'runtime' | 'merge';
  lm_adapter_scale?: number;
  lm_adapter_scale_attn?: number;
  lm_adapter_scale_mlp?: number;
  lm_adapter_scale_early?: number;
  lm_adapter_scale_mid?: number;
  lm_adapter_scale_late?: number;
};

/** Adds the adapter's own trigger word to the caption, exactly the way
 *  generate.ts does for a normal /api/generate render (`readMm3AdapterTrigger`
 *  + `applyMm3Trigger`, idempotent, skipped when the sidecar says the trigger
 *  was never trained). Rendering goes straight to the engine here, bypassing
 *  generate.ts entirely, so this has to be done by hand or a trained
 *  adapter's identity binding — the whole point of a likeness comparison —
 *  would silently never reach the model. */
function withAdapterTrigger(caption: string, adapterFile: string): string {
  const tg = readMm3AdapterTrigger(adapterFile);
  if (tg.trigger && tg.prepend) return applyMm3Trigger(caption, tg.trigger);
  return caption;
}

function buildRenderRequest(adapterFile: string | null, ov: { caption: string; lyrics: string; seed: number }): Mm3RenderRequest {
  const caption = adapterFile ? withAdapterTrigger(ov.caption, adapterFile) : ov.caption;
  const req: Mm3RenderRequest = {
    caption, lyrics: ov.lyrics || '', duration: RENDER_DURATION_SEC, seed: ov.seed, steps: RENDER_STEPS,
  };
  if (adapterFile) {
    const d = MM3_LM_ADAPTER_DEFAULT_SCALES;
    req.lm_adapter = adapterFile;
    req.lm_adapter_mode = 'runtime';
    req.lm_adapter_scale = d.scale;
    req.lm_adapter_scale_attn = d.scaleAttn;
    req.lm_adapter_scale_mlp = d.scaleMlp;
    req.lm_adapter_scale_early = d.scaleEarly;
    req.lm_adapter_scale_mid = d.scaleMid;
    req.lm_adapter_scale_late = d.scaleLate;
  }
  return req;
}

async function awaitMm3Job(jobId: string, what: string): Promise<void> {
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

async function renderMm3(adapterFile: string | null, out: string, ov: { caption: string; lyrics: string; seed: number }): Promise<void> {
  const req = buildRenderRequest(adapterFile, ov);
  const resp = await mm3Synth(req);
  await awaitMm3Job(resp.job_id, path.basename(out));
  const buf = Buffer.from(await (await aceClient.getJobResult(resp.job_id)).arrayBuffer());
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, buf);
}

// ── phase blind ──────────────────────────────────────────────────────────

type Item = 'base' | 'A' | 'B' | 'Arepeat';

function rateHeader(dsSlug: string): string {
  return `# Blind rating — MM3 crop cap vs whole-track prefix — ${dsSlug}\n\n`
    + '4 files per song: base model (no adapter), the two trained arms (recipe crop-750/prefix vs whole-track/flash), '
    + 'and a hidden REPEAT of the recipe arm rendered a second time at the same seed. MM3 renders are bit-deterministic '
    + 'for a fixed seed, model files and adapter (mm3-backend / mm3-lm-adapter-training skills), so the repeat is the '
    + "listener's own noise floor: if you can tell it apart from its twin, treat every other judgement on this set with "
    + `the same grain of salt. Letters shuffled per song. Same caption/lyrics/seed/steps per song across all 4 files, `
    + `${RENDER_DURATION_SEC} s, runtime adapter mode at the default scales (attn 1.0 / MLP 1.0). Score each 1-10 on `
    + 'quality / expressiveness / likeness, or just rank them. Open _key/KEY.json only when done.\n\n';
}

async function phaseBlind(dsSlug: string, songs: number): Promise<void> {
  const ds = await resolveDataset(dsSlug);
  const armFiles = await resolveArmCheckpoints(ds);
  if (!armFiles.recipe || !armFiles.wholetrack) {
    throw new Error(`${dsSlug}: both arms must be trained first (recipe: ${armFiles.recipe ? 'ok' : 'missing'}, `
      + `wholetrack: ${armFiles.wholetrack ? 'ok' : 'missing'}) — run --phase train`);
  }
  const lst = path.join(LST_ROOT, dsSlug);
  fs.mkdirSync(lst, { recursive: true });

  log(`${dsSlug}: pinning the MM3 LM to q8_0 before rendering — an adapter renders garbled on the f16 base `
    + '(mm3-lm-adapter-training skill)');
  await mm3SelectModel({ lm: 'q8_0' });

  const songPrompts = await loadSongs(ds, songs);
  for (const s of songPrompts) {
    writeIfAbsent(path.join(lst, `${s.slug}.caption.txt`), `${s.caption}\n`);
    writeIfAbsent(path.join(lst, `${s.slug}.lyrics.txt`), `${s.lyrics}\n`);
    writeIfAbsent(path.join(lst, `${s.slug}.SOURCE.txt`), `${s.source}\n`);
  }

  const keyDir = path.join(lst, '_key');
  fs.mkdirSync(keyDir, { recursive: true });
  const keyPath = path.join(keyDir, 'KEY.json');
  const key: Record<string, Record<string, Item>> =
    fs.existsSync(keyPath) ? JSON.parse(fs.readFileSync(keyPath, 'utf-8')) as Record<string, Record<string, Item>> : {};
  const items: Item[] = ['base', 'A', 'B', 'Arepeat'];
  const letters = ['A', 'B', 'C', 'D'];
  const rate = path.join(lst, 'RATE.md');
  if (!fs.existsSync(rate)) fs.writeFileSync(rate, rateHeader(dsSlug));

  for (const s of songPrompts) {
    if (!key[s.slug]) {
      const order = shuffleSecure(items);
      const row: Record<string, Item> = {};
      letters.forEach((L, i) => { row[L] = order[i]; });
      key[s.slug] = row;
      fs.writeFileSync(keyPath, JSON.stringify(key, null, 2));
    }
    if (!fs.readFileSync(rate, 'utf-8').includes(`## ${s.slug}`)) {
      fs.appendFileSync(rate, `## ${s.slug} — ${s.source}\n\n| file | quality | expressiveness | likeness | notes |\n`
        + '|---|---|---|---|---|\n' + letters.map(L => `| ${s.slug}_${L}.wav |  |  |  |  |`).join('\n') + '\n\n');
    }
    for (const L of letters) {
      const item = key[s.slug][L];
      const wav = path.join(lst, `${s.slug}_${L}.wav`);
      if (fs.existsSync(wav)) { log(`${s.slug}_${L}: exists, skipping`); continue; }
      const adapterFile = item === 'base' ? null : (item === 'A' || item === 'Arepeat') ? armFiles.recipe : armFiles.wholetrack;
      log(`${s.slug}_${L}: render (${item})`);
      await renderMm3(adapterFile, wav, { caption: s.caption, lyrics: s.lyrics, seed: RENDER_SEED });
    }
  }
  log(`${dsSlug}: blind set complete — key sealed in _key/KEY.json`);
}

// ── phase plan (dry run) ─────────────────────────────────────────────────

async function phasePlan(dsSlug: string, songs: number): Promise<void> {
  log(`dataset slug: ${dsSlug}`);
  const armNames = armDefs(dsSlug).map(a => a.label);
  log('logical run names (mm3-train-lm has no adapter-name field, so actual run directories are '
    + "'<slug>-<timestamp>' — this script identifies each arm by its recorded attnBackend/prefixFrames/maxFrames): "
    + armNames.join(', '));

  const up = await fetch(`${API}/jobs`).then(r => r.ok).catch(() => false);
  if (!up) {
    log('app not reachable on :3001 — cannot resolve the dataset id, existing runs, or render prompts without it. '
      + 'Start dev.bat and re-run --phase plan.');
    return;
  }

  const ds = await resolveDataset(dsSlug);
  log(`dataset id: ${ds.id}`);
  const wholeTrackFrames = await computeWholeTrackFrames(ds);
  log(`whole-track maxFrames (this dataset's longest usable track at ${MM3_FPS} fps, clamped to [64, 9000]): `
    + `${wholeTrackFrames}`);

  for (const arm of armDefs(dsSlug)) {
    const found = findArmRun(ds.id, ds.slug, arm);
    const body = armTrainBody(arm, wholeTrackFrames);
    log(`${arm.label}: ${found
      ? `found on disk at ${found.runName} (${found.summary.outcome}, step ${found.summary.lastStep})`
      : 'not found — would start fresh'}`);
    log(`${arm.label}: POST /datasets/${ds.id}/mm3-train-lm ${JSON.stringify(body)}`);
  }

  const songPrompts = await loadSongs(ds, songs);
  const armFiles = await resolveArmCheckpoints(ds);
  const renderItems: Array<{ item: Item; adapter: string | null }> = [
    { item: 'base', adapter: null },
    { item: 'A', adapter: armFiles.recipe },
    { item: 'B', adapter: armFiles.wholetrack },
    { item: 'Arepeat', adapter: armFiles.recipe },
  ];
  for (const s of songPrompts) {
    log(`${s.slug}: ${s.source} (caption ${s.caption.length} chars, lyrics ${s.lyrics.length} chars)`);
    for (const it of renderItems) {
      const req = buildRenderRequest(it.adapter, { caption: s.caption, lyrics: s.lyrics, seed: RENDER_SEED });
      log(`${s.slug} ${it.item}: POST /mm3/synth ${JSON.stringify(req)}`);
    }
  }
}

// ── main ─────────────────────────────────────────────────────────────────

async function main() {
  const a = args();
  const phase = a.get('phase') || 'plan';  // plan | train | blind | all — plan is the safe default
  const dsSlug = a.get('dataset');
  if (!dsSlug) throw new Error('--dataset <slug> is required');
  const songs = Number(a.get('songs') || 3);
  if (!['plan', 'train', 'blind', 'all'].includes(phase)) throw new Error(`unknown phase ${phase}`);

  if (phase === 'plan') { await phasePlan(dsSlug, songs); log('done'); return; }

  const up = await fetch(`${API}/jobs`).then(r => r.ok).catch(() => false);
  if (!up) throw new Error('Node server not reachable on :3001 — start dev.bat');

  if (phase === 'train' || phase === 'all') await phaseTrain(dsSlug);
  if (phase === 'blind' || phase === 'all') await phaseBlind(dsSlug, songs);
  log('done');
}

main().catch(e => { console.error(String(e?.stack || e)); process.exit(1); });
