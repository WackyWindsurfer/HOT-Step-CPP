# vocal-end: where the vocals sit in a track

`vocal_end.py` separates the vocal stem with the app's SuperSep, reads vocal activity from the stem's energy
(first vocal, last vocal, terminal instrumental tail, active share, longest gap), and runs whisper-cli per word on
the stem as a lyric-repeat hint. The energy trace is the trustworthy signal; Whisper on stems hallucinates.

Needs the engine running (`dev.bat`) with the SuperSep models installed, and a Whisper model in `models/whisper`.

```
py -3.13 tools/vocal-end/vocal_end.py server/data/audio/<song>.wav
py -3.13 tools/vocal-end/vocal_end.py "M:/Datasets/<album>/"*.flac
```

Output: one JSON line per file, a summary table, stems and JSON under `tools/vocal-end/out/` (gitignored).

Why it exists: the MM3 endings investigation (docs/plans/mm3-endings-checklist.md, 2026-09-10) needed to know
whether capped renders had finished singing. The LRC could not say (forced alignment); this could. It showed one
adapter failing three different ways (instrumental tail, sparse vocals, lyric loop) and that a long-song album and
a short-song album can have the same vocal share (0.68 vs 0.65) despite a 3x length difference.
