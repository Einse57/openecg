#!/usr/bin/env python3
"""Create 10 s sample windows for OpenVINO smoke / bench.

Writes synthetic sinus ECGs always; optionally downloads MIT-BIH record 100
(PhysioNet, requires network + wfdb).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "samples"


def synth_sinus(rr_ms=750, n_beats=14, fs=250, seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(rr_ms * (n_beats + 1) * fs / 1000)
    n_target = int(10 * fs)
    t = np.arange(n) / fs
    sig = 0.03 * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 0.008, n)
    qrs = np.hanning(max(3, int(0.08 * fs)))
    pw = 0.25 * np.hanning(max(3, int(0.10 * fs)))
    tw = 0.35 * np.hanning(max(3, int(0.18 * fs)))
    for i in range(n_beats):
        c = int((i + 1) * rr_ms * fs / 1000)
        if c >= n - len(qrs):
            break
        lo = c - len(qrs) // 2
        sig[lo : lo + len(qrs)] += qrs
        p0 = c - int(0.18 * fs)
        if p0 >= 0:
            sig[p0 : p0 + len(pw)] += pw
        t0 = c + int(0.16 * fs)
        if t0 + len(tw) < n:
            sig[t0 : t0 + len(tw)] += tw
    if len(sig) >= n_target:
        sig = sig[:n_target]
    else:
        pad = np.zeros(n_target, dtype=np.float64)
        pad[: len(sig)] = sig
        sig = pad
    return sig.astype(np.float32)


def fetch_mitdb100() -> None:
    import wfdb
    from scipy.signal import resample_poly

    rec = wfdb.rdrecord("100", pn_dir="mitdb", channels=[0])
    sig = rec.p_signal[:, 0].astype(np.float64)
    sig = sig[: int(12 * 360)]
    s250 = resample_poly(sig, 25, 36).astype(np.float32)[:2500]
    s500 = resample_poly(sig, 125, 90).astype(np.float32)[:5000]
    np.save(OUT / "mitdb100_10s_250hz.npy", s250)
    np.save(OUT / "mitdb100_10s_500hz.npy", s500)
    print("wrote mitdb100", s250.shape, s500.shape)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-mitdb", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "synth_sinus_10s_250hz.npy", synth_sinus(fs=250))
    np.save(OUT / "synth_sinus_10s_500hz.npy", synth_sinus(fs=500))
    print("wrote synth_sinus 250/500 Hz")
    if args.with_mitdb:
        fetch_mitdb100()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
