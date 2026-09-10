# tools/vocal-end/vocal_end.py — where do the VOCALS sit in a track? (2026-09-10)
#
# Answers "when does singing start / stop, how much of the track is sung, how long is the instrumental tail" for any
# audio file, without lyric alignment (the MM3 LRC is a forced alignment and cannot answer this: it stamps every lyric
# line somewhere by construction). Built for the MM3 endings investigation (docs/plans/mm3-endings-checklist.md), useful
# for any dataset or render question about vocal placement.
#
# Two signals per file:
#   1. SuperSep (the app's engine, POST /supersep/separate?level=4, 2-stem BS-RoFormer) -> vocal stem -> RMS per 0.5 s
#      -> vocal-active windows above a threshold relative to the stem's loudest window (default -30 dB; the reading is
#      stable from -20 to -40), runs shorter than 1 s dropped -> first vocal, last vocal, terminal tail, active share,
#      longest interior gap. THIS is the reliable signal. Validated 2026-09-10: real songs show 6-10 s tails; a render
#      that ended at 207 s had its last vocal at 199 s.
#   2. whisper-cli (tools/whisper, -oj --split-on-word --max-len 1) on the vocal stem -> per-word timestamps. UNRELIABLE
#      on stems: it hallucinates words past the end of the file and inflates counts (654 "words" for a 199-word sheet).
#      Kept as a lyric-repeat hint only (words_heard >> lyric_words = the lyrics went round again).
#
# Needs the app's engine running (ace-server on :8085; SuperSep models installed) and a Whisper model in models/whisper.
# FLAC/other inputs are decoded to 44.1 kHz WAV in memory (the engine's reader takes WAV/MP3).
#
# Usage: python tools/vocal-end/vocal_end.py <audio> [<audio> ...] [--thresh -30] [--lyrics "<text>"]
#   Env: HOTSTEP_ROOT (checkout, default = this repo), HOTSTEP_ENGINE (default http://127.0.0.1:8085),
#        VOCAL_END_OUT (stems + JSON, default tools/vocal-end/out/). Lyrics default to the songs table (app renders)
#        matched by file basename, else none. Prints one JSON line per file, then a table; appends to out/results.json.
import json, os, re, subprocess, sys, time, urllib.request, urllib.error, sqlite3, io
import numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__))
class P:  # the two paths the script needs, without the experiment harness
    ROOT = os.environ.get('HOTSTEP_ROOT', os.path.abspath(os.path.join(HERE, '..', '..')))
    ENGINE = os.environ.get('HOTSTEP_ENGINE', 'http://127.0.0.1:8085')

WHISPER = os.path.join(P.ROOT, 'tools', 'whisper', 'whisper-cli.exe')
WMODEL = next((os.path.join(P.ROOT, 'models', 'whisper', m) for m in ('ggml-large-v3-turbo.bin', 'ggml-base.bin')
               if os.path.exists(os.path.join(P.ROOT, 'models', 'whisper', m))), None)
OUT = os.environ.get('VOCAL_END_OUT', os.path.join(HERE, 'out')); os.makedirs(OUT, exist_ok=True)
argv = sys.argv[1:]
THRESH = float(argv[argv.index('--thresh') + 1]) if '--thresh' in argv else -30.0
LYR = argv[argv.index('--lyrics') + 1] if '--lyrics' in argv else None
files = [a for i, a in enumerate(argv) if not a.startswith('--') and (i == 0 or argv[i - 1] not in ('--thresh', '--lyrics'))]

def http_raw(url, data=None, timeout=1800):
    req = urllib.request.Request(url, data=data, method='POST' if data is not None else 'GET',
                                 headers={'content-type': 'application/octet-stream'})
    with urllib.request.urlopen(req, timeout=timeout) as r: return r.read()

def separate(path):
    """Returns the vocal stem as (mono float32, sr) via the engine's SuperSep; cached as <OUT>/<stem>.vocals.wav."""
    cache = os.path.join(OUT, os.path.splitext(os.path.basename(path))[0] + '.vocals.wav')
    if os.path.exists(cache):
        y, sr = sf.read(cache, dtype='float32'); return (y.mean(axis=1) if y.ndim > 1 else y), sr
    # The engine's audio reader takes WAV/MP3; datasets are 96 kHz FLAC, so decode + resample to 44.1 kHz WAV in memory.
    if path.lower().endswith('.wav'): body = open(path, 'rb').read()
    else:
        y0, sr0 = sf.read(path, dtype='float32')
        if sr0 != 44100:
            import librosa; y0 = librosa.resample(y0.T, orig_sr=sr0, target_sr=44100).T
        buf = io.BytesIO(); sf.write(buf, y0, 44100, format='WAV', subtype='PCM_16'); body = buf.getvalue()
    sub = json.loads(http_raw(P.ENGINE + '/supersep/separate?level=4', body))
    jid = sub.get('job_id') or sub.get('id')
    if not jid: raise RuntimeError(f'supersep submit: {sub}')
    while True:
        time.sleep(3)
        try:
            r = json.loads(http_raw(f'{P.ENGINE}/supersep/result?id={jid}'))
            if r.get('stems'): break
        except urllib.error.HTTPError as e:
            if e.code not in (202, 404, 409, 425): raise
    names = [s.get('name', '') for s in r['stems']]
    idx = next((i for i, n in enumerate(names) if 'vocal' in n.lower()), None)
    if idx is None: raise RuntimeError(f'no vocal stem in {names}')
    wav = http_raw(f'{P.ENGINE}/supersep/serve?id={jid}&stem={idx}')
    try: http_raw(f'{P.ENGINE}/supersep/release?id={jid}', b'')
    except Exception: pass
    y, sr = sf.read(io.BytesIO(wav), dtype='float32')
    sf.write(cache, y, sr)
    return (y.mean(axis=1) if y.ndim > 1 else y), sr

def activity(y, sr, win=0.5, thresh_db=THRESH, min_len=1.0):
    n = int(win * sr); m = len(y) // n
    rms = np.array([np.sqrt(np.mean(y[i * n:(i + 1) * n] ** 2)) + 1e-9 for i in range(m)])
    db = 20 * np.log10(rms / rms.max())
    act = db > thresh_db
    # drop active runs shorter than min_len
    runs = []; i = 0
    while i < m:
        if act[i]:
            j = i
            while j < m and act[j]: j += 1
            if (j - i) * win >= min_len: runs.append((i * win, j * win))
            i = j
        else: i += 1
    dur = len(y) / sr
    if not runs: return {'duration': round(dur, 1), 'vocal_runs': 0, 'last_vocal_end': None, 'tail': round(dur, 1), 'active_frac': 0.0, 'longest_gap': round(dur, 1)}
    gaps = [runs[k + 1][0] - runs[k][1] for k in range(len(runs) - 1)]
    return {'duration': round(dur, 1), 'vocal_runs': len(runs), 'first_vocal': round(runs[0][0], 1),
            'last_vocal_end': round(runs[-1][1], 1), 'tail': round(dur - runs[-1][1], 1),
            'active_frac': round(sum(b - a for a, b in runs) / dur, 2), 'longest_gap': round(max(gaps), 1) if gaps else 0.0}

def whisper(stem_path, lyrics):
    js = stem_path + '.json'
    if not os.path.exists(js):
        args = [WHISPER, '-m', WMODEL, '-f', stem_path, '-oj', '--split-on-word', '--max-len', '1', '--beam-size', '5', '--no-prints', '--language', 'en']
        if lyrics: args += ['--prompt', ' '.join(re.sub(r'\[.*?\]', ' ', lyrics).split())[:900]]
        subprocess.run(args, capture_output=True, text=True, timeout=1800)
    if not os.path.exists(js): return {'whisper': 'no output'}
    j = json.load(open(js, encoding='utf-8'))
    segs = [s for s in j.get('transcription', []) if s.get('text', '').strip()]
    words = [(s['text'].strip(), s['offsets']['from'] / 1000, s['offsets']['to'] / 1000) for s in segs]
    out = {'words_heard': len(words), 'last_word_end': round(words[-1][2], 1) if words else None,
           'first_word': round(words[0][1], 1) if words else None}
    if lyrics:
        lw = [w for w in re.findall(r"[a-z']+", re.sub(r'\[.*?\]', ' ', lyrics.lower()))]
        out['lyric_words'] = len(lw)
        lines = [l.strip() for l in lyrics.splitlines() if l.strip() and not l.strip().startswith('[')]
        last_line = re.findall(r"[a-z']+", lines[-1].lower()) if lines else []
        heard = [w.lower().strip(",.!?\"") for w, a, b in words]
        # where does the final lyric line's word sequence occur (first and last hit)?
        hits = [k for k in range(len(heard) - len(last_line) + 1) if len(last_line) >= 3 and heard[k:k + len(last_line)] == last_line] if last_line else []
        out['final_line_hits_at'] = [round(words[k][1], 1) for k in hits]
    return out

def lyrics_for(path):
    if LYR: return LYR
    try:
        db = sqlite3.connect(os.path.join(P.ROOT, 'server', 'data', 'hotstep.db'))
        r = db.execute("SELECT lyrics FROM songs WHERE audio_url LIKE ?", ('%' + os.path.basename(path),)).fetchone()
        return r[0] if r else None
    except Exception: return None

results = []
for path in files:
    t0 = time.time(); y, sr = separate(path)
    a = activity(y, sr)
    stem = os.path.join(OUT, os.path.splitext(os.path.basename(path))[0] + '.vocals.wav')
    w = whisper(stem, lyrics_for(path))
    rec = {'file': os.path.basename(path), **a, **w, 'wall_s': round(time.time() - t0)}
    results.append(rec); print(json.dumps(rec), flush=True)
json.dump(results, open(os.path.join(OUT, 'results.json'), 'a'), indent=1)
print(f"\n{'file':52s} dur  last_vocal  tail  active  gap | words last_word  lyric_words  final_line_at")
for r in results:
    print(f"{r['file'][:52]:52s} {r['duration']:4.0f}  {str(r.get('last_vocal_end')):>6s}  {r['tail']:5.1f}  {r['active_frac']:.2f}  {r['longest_gap']:4.0f} | {r.get('words_heard','?'):>5}  {str(r.get('last_word_end')):>6s}  {r.get('lyric_words','?'):>5}  {r.get('final_line_hits_at','')}")
