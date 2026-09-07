#!/usr/bin/env python3
"""Stage A of the MM3 end-of-song work: audio-derived trailing-silence trim for a training dataset.

For every labelled track with RVQ codes, run ffmpeg silencedetect on the SOURCE AUDIO, take the last silence
run only if it extends to the end of the file, and propose keep_frames = ceil((silence_start + margin) * 25).
Writes <dataset>/mm3-codes/codes/trim.json ({"<id>": keep_frames}), a report, and per-track boundary snippets
(the last WINDOW seconds up to the proposed cut plus TAIL_AFTER seconds beyond it, cut landing at a fixed
offset) so the boundaries can be listened to before any retrain. The trainer applies trim.json only under
--trim-trailing-silence. Thresholds are detection settings, not proven safe defaults; inspect the snippets.

Usage: python trim.py <dataset slug> [--noise -50] [--min-sil 0.3] [--margin 0.5] [--snippets <dir>] [--copy-to <dir>]
"""
import os, sys, json, math, glob, struct, shutil, subprocess, urllib.request

API = 'http://127.0.0.1:3001/api/training'
TRAIN = r'D:\Ace-Step-Latest\hot-step-cpp\server\data\training\datasets'
FPS = 25

def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default

def ffmpeg():
    c = glob.glob(r'D:\Ace-Step-Latest\hot-step-cpp\server\node_modules\ffmpeg-static\ffmpeg.exe')
    if not c: raise SystemExit('ffmpeg-static not found')
    return c[0]

def http(url):
    with urllib.request.urlopen(url, timeout=30) as r: return json.loads(r.read())

def n_frames(codes_path):
    return len(open(codes_path, 'rb').read()) // 32 - 1   # rows minus the warm-up row

def silence_tail(ff, audio, noise, min_sil):
    """(duration_s, tail_silence_start_s or None). Only a silence run that reaches the end counts."""
    r = subprocess.run([ff, '-hide_banner', '-nostats', '-i', audio, '-af', f'silencedetect=noise={noise}dB:d={min_sil}',
                        '-f', 'null', '-'], capture_output=True, text=True)
    dur = None; starts = []; ends = []
    for line in r.stderr.splitlines():
        if 'Duration:' in line and dur is None:
            h, m, s = line.split('Duration: ')[1].split(',')[0].split(':'); dur = int(h) * 3600 + int(m) * 60 + float(s)
        if 'silence_start:' in line: starts.append(float(line.split('silence_start:')[1].split()[0]))
        if 'silence_end:' in line: ends.append(float(line.split('silence_end:')[1].split()[0]))
    if dur is None or not starts: return dur, None
    # a run that reaches the end has no matching silence_end (ffmpeg only prints it when sound resumes)
    if len(ends) < len(starts): return dur, starts[-1]
    if dur - ends[-1] < 0.05: return dur, starts[-1]
    return dur, None

def main():
    slug = sys.argv[1]
    noise = arg('--noise', -50.0); min_sil = arg('--min-sil', 0.3); margin = arg('--margin', 0.5)
    snip_dir = arg('--snippets', os.path.join(r'D:\Ace-Step-Latest\hot-step-cpp\_experiments\_LISTENING', f'trim-boundaries-{slug}'))
    copy_to = arg('--copy-to', '')
    ff = ffmpeg()
    dsl = http(f'{API}/datasets'); dsl = dsl.get('datasets', dsl) if isinstance(dsl, dict) else dsl
    ds = next(d for d in dsl if d.get('slug') == slug or d.get('customTag') == slug or d.get('name') == slug)
    ds = http(f'{API}/datasets/{ds["id"]}'); ds = ds.get('dataset', ds)
    manifest = json.load(open(ds['datasetJsonPath'], encoding='utf-8'))
    codes_dir = os.path.join(TRAIN, slug, 'mm3-codes', 'codes')
    os.makedirs(snip_dir, exist_ok=True)
    trim = {}; rows = []
    WINDOW = 6.0; TAIL_AFTER = 2.0
    for s in manifest['samples']:
        cpath = os.path.join(codes_dir, s['id'] + '.codes')
        if not os.path.exists(cpath): continue
        nf = n_frames(cpath)
        dur, sil = silence_tail(ff, s['audio_path'], noise, min_sil)
        if sil is None:
            rows.append((s['filename'], nf, dur, None, None, 'no trailing silence detected; untouched')); continue
        keep_s = sil + margin
        keep = min(nf, int(math.ceil(keep_s * FPS)))
        if keep >= nf:
            rows.append((s['filename'], nf, dur, sil, None, 'silence shorter than the margin; untouched')); continue
        trim[s['id']] = keep
        # snippet: [cut - WINDOW, cut + TAIL_AFTER], so the proposed cut sits at WINDOW seconds
        cut_s = keep / FPS
        out = os.path.join(snip_dir, os.path.splitext(s['filename'])[0] + f'.cut@{WINDOW:.0f}s.wav')
        subprocess.run([ff, '-hide_banner', '-loglevel', 'error', '-y', '-ss', f'{max(0.0, cut_s - WINDOW):.3f}', '-t',
                        f'{WINDOW + TAIL_AFTER:.3f}', '-i', s['audio_path'], '-ac', '2', '-ar', '44100', out], check=False)
        rows.append((s['filename'], nf, dur, sil, keep, f'cut {nf - keep} frames ({(nf - keep) / FPS:.1f} s)'))
    json.dump(trim, open(os.path.join(codes_dir, 'trim.json'), 'w'), indent=1)
    rep = [f'# Trailing-silence trim proposal: {slug}\n',
           f'ffmpeg silencedetect noise {noise} dB, min {min_sil} s (trailing run only), margin {margin} s after silence_start. '
           f'Snippets: the proposed cut sits at exactly {WINDOW:.0f}.0 s in each file; everything after it is what training would drop.\n',
           '| track | frames | audio s | silence_start s | keep frames | action |', '|---|---|---|---|---|---|']
    for f, nf, dur, sil, keep, act in rows:
        rep.append(f'| {f} | {nf} | {dur:.2f} | {"" if sil is None else f"{sil:.2f}"} | {"" if keep is None else keep} | {act} |')
    rep.append(f'\ntrim.json written to {codes_dir} with {len(trim)} entries. The trainer only applies it under --trim-trailing-silence.')
    open(os.path.join(snip_dir, 'REPORT.md'), 'w', encoding='utf-8').write('\n'.join(rep) + '\n')
    print('\n'.join(rep))
    if copy_to:
        dst = os.path.join(copy_to, os.path.basename(snip_dir)); os.makedirs(dst, exist_ok=True)
        for f in os.listdir(snip_dir): shutil.copy2(os.path.join(snip_dir, f), dst)
        print('copied to', dst)

if __name__ == '__main__':
    main()
