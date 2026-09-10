#!/usr/bin/env python3
"""Lever 4b (docs/plans/2026-09-08-mm3-endings-status.md): an ON-POLICY ending corpus for one adapter.

The adapter's own plans never reach a state it recognises as an ending (exposure bias), so a prior built from
BASE plans teaches "be the base" (likeness cost, 2026-09-08). Instead: plan each training song with the ADAPTER,
cut the plan at several points, and let the frozen BASE continue from there to its own EOS (forced replay with
forced_continue, engine >= 2026-09-08). The last K frames of that continuation, with H frames of history in front
(adapter frames, then base frames), become an ending excerpt: "from where YOUR song actually is, this is how it
wraps up". Written in the history-bearing corpus layout the trainer rehearses behind its frozen prefix.

Usage: python build_onpolicy.py <out dir> --adapter <adapter_model.safetensors> --dataset <dataset id>
         [--trigger <tag>] [--k 500] [--history 2048] [--cuts 3500,4500,5500] [--seed 3000] [--max-songs 12]
"""
import os, sys, json, struct, time, hashlib, urllib.request

ENGINE = 'http://127.0.0.1:8085'; API = 'http://127.0.0.1:3001/api/training'
NC = 7

def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default

def http(method, url, body=None, timeout=3600, raw=False):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={'content-type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return (r.read(), dict(r.headers)) if raw else json.loads(r.read())

def plan(body):
    raw, hdr = http('POST', ENGINE + '/mm3/lm-plan?dump=1', body, raw=True)
    I = int(hdr['X-MM3-Iterations']); F = int(hdr['X-MM3-Frames']); eos = hdr['X-MM3-Eos'] == '1'
    ints = struct.unpack('<%di' % (I + I * NC), raw[-(I + I * NC) * 4:])
    return F, eos, list(ints[:I]), list(ints[I:])   # sem[I], ac[I*NC], iteration-major, entry 0 = warm-up

def write_codes(path, sem, ac, lo, hi):
    with open(path, 'wb') as f:
        for r in range(lo, hi): f.write(struct.pack('<8i', sem[r], *ac[r * NC:(r + 1) * NC]))

def main():
    out_dir = sys.argv[1]; adapter = arg('--adapter', ''); ds_id = arg('--dataset', ''); trigger = arg('--trigger', '')
    K = arg('--k', 500); H = arg('--history', 2048); seed0 = arg('--seed', 3000); max_songs = arg('--max-songs', 12)
    cuts = [int(x) for x in arg('--cuts', '3500,4500,5500').split(',')]
    assert adapter and ds_id, 'need --adapter and --dataset'
    os.makedirs(os.path.join(out_dir, 'codes'), exist_ok=True)
    ds = http('GET', f'{API}/datasets/{ds_id}'); ds = ds.get('dataset', ds)
    trigger = trigger or ds.get('customTag') or ''
    tracks = [s for s in ds.get('samples') or [] if not s.get('excluded') and (s.get('lyrics') or '').strip()
              and os.path.exists(os.path.splitext(s['audioPath'])[0] + '.mm3.txt')][:max_songs]
    http('POST', ENGINE + '/mm3/select-model', {'lm': 'q8_0', 'synth': ''}); http('POST', ENGINE + '/mm3/warm', {})
    samples = []; log = []
    for i, s in enumerate(tracks):
        cap = open(os.path.splitext(s['audioPath'])[0] + '.mm3.txt', encoding='utf-8').read().lstrip('﻿').lstrip()
        if trigger and not cap.lower().startswith(trigger.lower() + ','): cap = f'{trigger}, {cap}'
        tc = http('POST', ENGINE + '/mm3/tokenize-check', {'caption': cap, 'lyrics': s['lyrics'], 'prompt': True})
        if not tc.get('ok', True) or 'prompt' not in tc: log.append(f'{s["filename"]}: prompt rejected'); continue
        t0 = time.time()
        # 1. the adapter's own plan, to EOS or the cap
        Fa, eos_a, sem, ac = plan({'prompt': tc['prompt'], 'seed': seed0 + i, 'max_frames': 7500,
                                   'lm_adapter': adapter, 'lm_adapter_mode': 'runtime', 'lm_adapter_scale': 1.0})
        msg = f'{s["filename"][:40]}: adapter plan {Fa} frames eos {eos_a} ({time.time()-t0:.0f}s)'
        pid = hashlib.md5(f'{s["filename"]}:{seed0+i}'.encode()).hexdigest()[:8]
        n = 0
        for T in cuts:
            if T >= Fa - 25: continue
            # 2. the base continues from frame T (forced entries 0..T = warm-up + T frames), sampling to its EOS
            t1 = time.time()
            Fb, eos_b, sem_b, ac_b = plan({'prompt': tc['prompt'], 'seed': seed0 + i, 'max_frames': 7500, 'lm_adapter_clear': True,
                                           'forced_semantic': sem[:T + 1], 'forced_acoustic': ac[:(T + 1) * NC], 'forced_continue': True})
            cont = Fb - T
            if not eos_b or cont < 25 or Fb < K + 1:
                log.append(f'   cut {T}: base ran {cont} frames, eos {eos_b} -> skipped'); continue
            lo = max(0, Fb - K - H)          # rows lo..Fb inclusive: history + window + warm-up row
            sid = f'{pid}_c{T}'
            write_codes(os.path.join(out_dir, 'codes', sid + '.codes'), sem_b, ac_b, lo, Fb + 1)
            open(os.path.join(out_dir, sid + '.mm3.txt'), 'w', encoding='utf-8').write(cap.strip() + '\n')
            samples.append({'id': sid, 'filename': sid + '.plan', 'lyrics': s['lyrics'], 'frame_offset': lo, 'labeled': 'True',
                            'source_track': s['filename'], 'cut_frame': T, 'plan_frames': Fb, 'adapter_plan_frames': Fa,
                            'history': Fb - K - lo, 'seed': seed0 + i})
            log.append(f'   cut {T}: base continued {cont} frames to EOS at {Fb} ({time.time()-t1:.0f}s)'); n += 1
        log.append(msg + f' -> {n} excerpts'); print(msg, f'-> {n} excerpts', flush=True)
        json.dump({'metadata': {'kind': 'mm3_onpolicy_endings', 'k': K, 'history': H, 'cuts': cuts, 'adapter': adapter, 'dataset': ds_id,
                                'lm': 'mm3-lm-q8_0', 'built': time.strftime('%Y-%m-%d %H:%M')}, 'samples': samples},
                  open(os.path.join(out_dir, 'dataset.json'), 'w', encoding='utf-8'), indent=1)
    json.dump({'kind': 'mm3_forced_codes', 'producer': 'tools/mm3-reg-corpus/build_onpolicy.py', 'tracks': len(samples), 'warmup_row': True,
               'columns': 'semantic,acoustic1..7', 'note': 'adapter plan cut at cut_frame, base continuation to EOS; row 0 dropped by the loader'},
              open(os.path.join(out_dir, 'codes', 'codes.json'), 'w'), indent=1)
    open(os.path.join(out_dir, 'BUILD.log'), 'w', encoding='utf-8').write('\n'.join(log) + '\n')
    print(f'{len(samples)} excerpts in {out_dir}')

if __name__ == '__main__':
    main()
