#!/usr/bin/env python3
"""
post_training_analysis.py
─────────────────────────
Run after GoT-SAC training completes to:
  1. Export every TensorBoard scalar tag → CSV
  2. Plot every scalar tag → PNG  (dark theme, raw + smoothed)
  3. Plot every reward curve .npy  → PNG

Works from both inside the Docker container (/workspace/...) and
on the host machine (auto-detects the correct base path).

Usage:
    python3 post_training_analysis.py
"""

import os
import csv
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Resolve base paths (container vs host) ────────────────────────────────────
CONTAINER_BASE = '/workspace/catkin_ws/src/gtrl/scripts/SAC'
HOST_BASE       = os.path.join(os.path.dirname(os.path.abspath(__file__)))

BASE = CONTAINER_BASE if os.path.isdir(CONTAINER_BASE) else HOST_BASE

LOG_DIR    = os.path.join(BASE, 'runs', 'gtrl')
EXPORT_DIR = os.path.join(BASE, 'results')
CURVES_DIR = os.path.join(BASE, 'final_curves')

os.makedirs(EXPORT_DIR, exist_ok=True)

# ── Dark-theme plot helpers ───────────────────────────────────────────────────
BG_OUTER  = '#0d1117'
BG_INNER  = '#161b22'
COLOR_RAW = '#58a6ff'
COLOR_FG  = '#e6edf3'
COLOR_MUT = '#8b949e'
COLOR_GRD = '#21262d'
COLOR_BRD = '#30363d'

def make_fig(title: str, xlabel: str = 'Step / Episode', ylabel: str = 'Value'):
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG_OUTER)
    ax.set_facecolor(BG_INNER)
    ax.set_title(title, color=COLOR_FG, fontsize=13, fontweight='bold', pad=14)
    ax.set_xlabel(xlabel, color=COLOR_MUT, fontsize=11, labelpad=8)
    ax.set_ylabel(ylabel, color=COLOR_MUT, fontsize=11, labelpad=8)
    ax.tick_params(colors=COLOR_MUT, labelsize=9)
    ax.grid(True, color=COLOR_GRD, linestyle='--', alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color(COLOR_BRD)
    return fig, ax

def smooth(values, w):
    """Simple moving-average smoothing."""
    if len(values) <= w:
        return np.array(values)
    kernel = np.ones(w) / w
    return np.convolve(values, kernel, mode='same')

def save_fig(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG_OUTER)
    plt.close(fig)
    print(f'  [PNG] {path}')


# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — TensorBoard scalar export (CSV + PNG)
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{"="*60}')
print(f'  Post-Training Analysis')
print(f'  Base path: {BASE}')
print(f'{"="*60}')

if os.path.isdir(LOG_DIR):
    try:
        from torch.utils.tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        print(f'\n[1/2] Loading TensorBoard events from:\n      {LOG_DIR}')

        ea = EventAccumulator(LOG_DIR)
        ea.Reload()
        tags = ea.Tags().get('scalars', [])

        if not tags:
            print('  No scalar tags found — skipping TensorBoard export.')
        else:
            print(f'  Found {len(tags)} tag(s): {tags}')
            for tag in tags:
                events = ea.Scalars(tag)
                if not events:
                    continue

                steps  = [e.step  for e in events]
                values = [e.value for e in events]
                safe   = tag.replace('/', '_')

                # CSV
                csv_path = os.path.join(EXPORT_DIR, f'{safe}.csv')
                with open(csv_path, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['step', 'value'])
                    w.writerows(zip(steps, values))
                print(f'  [CSV] {csv_path}')

                # PNG
                fig, ax = make_fig(f'{tag}  ({len(values)} points)')
                ax.plot(steps, values, color=COLOR_RAW, alpha=0.35,
                        linewidth=0.9, label='Raw')
                if len(values) > 5:
                    w_size = min(30, max(3, len(values) // 10))
                    sm = smooth(values, w_size)
                    ax.plot(steps, sm, color=COLOR_RAW, linewidth=2.0,
                            label=f'Smoothed (w={w_size})')
                ax.legend(facecolor=BG_INNER, edgecolor=COLOR_BRD,
                          labelcolor=COLOR_FG, loc='best')
                save_fig(fig, os.path.join(EXPORT_DIR, f'{safe}.png'))

    except ImportError:
        print('  torch.utils.tensorboard not available — skipping TensorBoard export.')
    except Exception as exc:
        print(f'  ERROR during TensorBoard export: {exc}')
else:
    print(f'  TensorBoard log dir not found: {LOG_DIR}  — skipping.')


# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — Reward curve .npy plots
# ══════════════════════════════════════════════════════════════════════════════
npy_files = sorted(glob.glob(os.path.join(CURVES_DIR, '*.npy')))
print(f'\n[2/2] Reward curves in:\n      {CURVES_DIR}')

if not npy_files:
    print('  No .npy files found — skipping.')
else:
    print(f'  Found {len(npy_files)} file(s).')
    for fpath in npy_files:
        try:
            data = np.load(fpath, allow_pickle=True)
            if data.ndim == 0 or len(data) == 0:
                print(f'  Skipping empty file: {fpath}')
                continue

            title = os.path.basename(fpath)
            fig, ax = make_fig(title, xlabel='Episode', ylabel='Mean Reward (20-ep window)')
            ax.plot(data, color=COLOR_RAW, alpha=0.4, linewidth=0.8, label='Raw')
            w_size = min(30, max(1, len(data) // 10))
            sm = smooth(data, w_size)
            ax.plot(sm, color=COLOR_RAW, linewidth=2.2, label=f'Smoothed (w={w_size})')
            ax.legend(facecolor=BG_INNER, edgecolor=COLOR_BRD,
                      labelcolor=COLOR_FG, loc='best')
            out = os.path.join(EXPORT_DIR, title.replace('.npy', '_curve.png'))
            save_fig(fig, out)
        except Exception as exc:
            print(f'  ERROR processing {fpath}: {exc}')

print(f'\n  All outputs saved to:\n  {EXPORT_DIR}\n')
