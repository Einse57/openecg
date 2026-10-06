# Sample ECG windows

Generated locally (not committed as `.npy` — see `.gitignore`).

```bash
python scripts/make_samples.py              # synth sinus @ 250/500 Hz
python scripts/make_samples.py --with-mitdb # + PhysioNet MIT-BIH 100 (wfdb)
```

Used by `bench_openvino.py` and `scripts/agree_openvino.py`.
