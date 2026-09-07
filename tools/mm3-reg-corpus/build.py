#!/usr/bin/env python3
"""Stage C of the MM3 end-of-song work: a regularisation corpus of BASE-MODEL PLAN EXCERPTS that end at a real EOS.

For N prompts (Lyric Studio generations that carry an MM3 caption, excluding the artists under training), plan with
the pristine base LM via POST /mm3/lm-plan until it emits EOS, then cut two excerpts per plan:
  <id>_start  frames 0..K-1        frame_offset 0
  <id>_end    frames F-K..F-1      frame_offset F-K   (the excerpt ends where the base chose EOS)
Each excerpt is written in the trainer's .codes layout (int32 [rows, 8], row 0 = warm-up) so the whole sample
fits a K-frame crop: the trainer then marks at_end and supervises EOS on the end excerpts, and under
--crop-anchor song places the frames at the positions the base saw (frame_offset, read from dataset.json).
Output folder holds dataset.json, <id>.mm3.txt captions and codes/<id>.codes; pass it as
regularisation: { corpusDir } to POST /api/training/datasets/:id/mm3-train-lm.

Usage: python build.py <out dir> [--n 12] [--k 500] [--exclude "green day,system of a down"] [--seed 1000]
       [--history H --out2 <dir>]   also write a twin corpus (same plans, same ids) whose END excerpts carry H
                                    frames of history before the K-frame window: a trainer from 2026-09-07 on
                                    windows the last K frames and rehearses them behind its frozen KV prefix
                                    built from that history (Phase 4 step 2b).
"""
import os, sys, json, struct, time, hashlib, sqlite3, urllib.request

ENGINE = 'http://127.0.0.1:8085'
DB = r'D:\Ace-Step-Latest\hot-step-cpp\server\data\hotstep.db'
NC = 7

def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default

def http(method, url, body=None, timeout=1800, raw=False):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={'content-type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return (r.read(), dict(r.headers)) if raw else json.loads(r.read())

def pick_prompts(n, exclude):
    db = sqlite3.connect(DB)
    rows = db.execute("select g.id, g.title, g.lyrics, g.caption_mm3, g.profile_id, p.profile_data from generations g "
                      "join profiles p on p.id = g.profile_id where length(coalesce(g.caption_mm3,'')) > 200 "
                      "and length(coalesce(g.lyrics,'')) > 200 order by g.id desc").fetchall()
    out = []; seen = set()
    excl = [x.strip().lower() for x in (exclude or '').split(',') if x.strip()]   # comma list of artists to leave out
    for gid, title, lyrics, cap, pid, pdata in rows:
        if any(x in (pdata or '').lower() for x in excl): continue
        if pid in seen: continue          # one song per profile: spread the prior over artists
        seen.add(pid); out.append((gid, title, lyrics, cap))
        if len(out) >= n: break
    return out

def write_codes(path, rows):
    with open(path, 'wb') as f:
        for r in rows: f.write(struct.pack('<8i', *r))

def main():
    out_dir = sys.argv[1]; n = arg('--n', 12); K = arg('--k', 500); exclude = arg('--exclude', 'green day'); seed0 = arg('--seed', 1000)
    H = arg('--history', 0); out2 = arg('--out2', '')
    outs = [(out_dir, 0)] + ([(out2, H)] if H > 0 and out2 else [])
    for d, _ in outs: os.makedirs(os.path.join(d, 'codes'), exist_ok=True)
    http('POST', ENGINE + '/mm3/select-model', {'lm': 'q8_0', 'synth': ''}); http('POST', ENGINE + '/mm3/warm', {})
    samples = {d: [] for d, _ in outs}; log = []
    for i, (gid, title, lyrics, cap) in enumerate(pick_prompts(n, exclude)):
        tc = http('POST', ENGINE + '/mm3/tokenize-check', {'caption': cap, 'lyrics': lyrics, 'prompt': True})
        if not tc.get('ok', True) or 'prompt' not in tc: log.append(f'#{gid} {title}: prompt rejected'); continue
        t0 = time.time()
        body, hdr = http('POST', ENGINE + '/mm3/lm-plan?dump=1', {'prompt': tc['prompt'], 'seed': seed0 + i, 'max_frames': 7500, 'lm_adapter_clear': True}, raw=True)
        I = int(hdr['X-MM3-Iterations']); F = int(hdr['X-MM3-Frames']); eos = hdr['X-MM3-Eos'] == '1'
        ints = struct.unpack('<%di' % (I + I * NC), body[-(I + I * NC) * 4:]); sem = ints[:I]; ac = ints[I:]
        rows = [(sem[r],) + tuple(ac[r * NC:(r + 1) * NC]) for r in range(I)]   # row 0 = warm-up, row r = frame r
        msg = f'#{gid} "{title}": {F} frames, eos {eos}, {time.time()-t0:.0f}s'
        if not eos or F < 2 * K + H: log.append(msg + ' -> skipped'); print(msg, '-> skipped', flush=True); continue
        pid = hashlib.md5(f'{gid}:{seed0+i}'.encode()).hexdigest()[:8]
        for d, h in outs:
            # the end excerpt with h frames of history in front (h = 0: the plain K-frame excerpt)
            for tag, sel, off in (('start', rows[0:K + 1], 0), ('end', rows[F - K - h:F + 1], F - K - h)):
                sid = f'{pid}_{tag}'
                write_codes(os.path.join(d, 'codes', sid + '.codes'), sel)
                open(os.path.join(d, sid + '.mm3.txt'), 'w', encoding='utf-8').write(cap.strip() + '\n')
                samples[d].append({'id': sid, 'filename': sid + '.plan', 'lyrics': lyrics, 'frame_offset': off, 'labeled': 'True',
                                   'source_generation': gid, 'title': title, 'plan_frames': F, 'seed': seed0 + i,
                                   'history': h if tag == 'end' else 0})
            json.dump({'metadata': {'kind': 'mm3_base_plan_excerpts', 'k': K, 'history': h, 'lm': 'mm3-lm-q8_0', 'built': time.strftime('%Y-%m-%d %H:%M')},
                       'samples': samples[d]}, open(os.path.join(d, 'dataset.json'), 'w', encoding='utf-8'), indent=1)
        log.append(msg + ' -> 2 excerpts'); print(msg, '-> 2 excerpts', flush=True)
    for d, h in outs:
        json.dump({'kind': 'mm3_forced_codes', 'producer': 'tools/mm3-reg-corpus/build.py', 'tracks': len(samples[d]), 'warmup_row': True, 'history': h,
                   'columns': 'semantic,acoustic1..7', 'note': 'base-plan excerpts; row 0 is the frame before the excerpt (dropped by the loader)'},
                  open(os.path.join(d, 'codes', 'codes.json'), 'w'), indent=1)
        open(os.path.join(d, 'BUILD.log'), 'w', encoding='utf-8').write('\n'.join(log) + '\n')
        print(f'{len(samples[d])} excerpts in {d} (history {h})')

if __name__ == '__main__':
    main()
