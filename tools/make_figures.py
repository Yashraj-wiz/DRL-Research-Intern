#!/usr/bin/env python3
"""
make_figures.py
───────────────
Rebuilds the README figures and prints the tables used in the README.

Inputs (inside the repo):
  catkin_ws/src/gtrl/scripts/SAC/runs/<session>/events.out.tfevents.*
  catkin_ws/src/gtrl/scripts/SAC/results/evaluation_summary.json
Outputs:
  docs/figures/training_goal_rate.png   (+ _dark.png)
  docs/figures/checkpoint_eval.png      (+ _dark.png)

TensorBoard event files are parsed directly, so only numpy and matplotlib
are needed.

Usage:
    python3 tools/make_figures.py
"""

import glob
import json
import os
import re
import struct
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAC_DIR = os.path.join(ROOT, 'catkin_ws', 'src', 'gtrl', 'scripts', 'SAC')
OUT_DIR = os.path.join(ROOT, 'docs', 'figures')

# Training sessions in chronological order: (folder under runs/, label)
SESSIONS = [
    ('1_scratch_seed525',  'From scratch, seed 525'),
    ('2_scratch_seed420',  'From scratch, seed 420'),
    ('3_finetune_seed420', 'Fine-tune of #2 from episode 800 (stopped)'),
    ('4_run2a',            'Run 2, launch A'),
    ('5_run2b',            'Run 2, launch B'),
    ('6_run3',             'Run 3, fine-tune of #4 from episode 1100'),
]
WINDOW       = 100  # episodes in the rolling goal rate
MIN_EPISODES = 20   # the rolling rate is not drawn before this many episodes
EVAL_EPISODES = 5   # test episodes per checkpoint (eval_entrypoint.sh)

THEMES = {
    'light': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', muted='#898781',
                  grid='#e1e0d9', axis='#c3c2b7', series='#2a78d6'),
    'dark':  dict(surface='#1a1a19', ink='#f0efec', ink2='#c3c2b7', muted='#898781',
                  grid='#2c2c2a', axis='#383835', series='#3987e5'),
}

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Helvetica Neue', 'Arial', 'DejaVu Sans']


# ── TensorBoard event files (TFRecord framing + protobuf) ─────────────────────
def _varint(buf, pos):
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, pos
        shift += 7


def _fields(buf):
    """Yield (field number, wire type, value) for each field of a protobuf message."""
    pos = 0
    while pos < len(buf):
        key, pos = _varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 1:
            value, pos = buf[pos:pos + 8], pos + 8
        elif wire == 2:
            length, pos = _varint(buf, pos)
            value, pos = buf[pos:pos + length], pos + length
        elif wire == 5:
            value, pos = buf[pos:pos + 4], pos + 4
        else:
            raise ValueError(f'unsupported protobuf wire type {wire}')
        yield field, wire, value


def read_scalars(path):
    """Return {tag: [values in logging order]} for the scalars in one event file."""
    with open(path, 'rb') as f:
        data = f.read()
    scalars, pos = {}, 0
    while pos + 12 <= len(data):
        (length,) = struct.unpack_from('<Q', data, pos)
        record = data[pos + 12:pos + 12 + length]
        pos += 12 + length + 4                      # length, crc, data, crc
        for field, wire, summary in _fields(record):
            if field != 5 or wire != 2:             # Event.summary
                continue
            for _, _, value in _fields(summary):    # Summary.value
                tag = number = None
                for f, w, v in _fields(value):
                    if f == 1:                      # Value.tag
                        tag = v.decode('utf-8')
                    elif f == 2 and w == 5:         # Value.simple_value
                        number = struct.unpack('<f', v)[0]
                if tag is not None and number is not None:
                    scalars.setdefault(tag, []).append(number)
    return scalars


# ── Data ──────────────────────────────────────────────────────────────────────
def load_session(folder):
    files = glob.glob(os.path.join(SAC_DIR, 'runs', folder, 'events.out.tfevents.*'))
    if len(files) != 1:
        raise SystemExit(f'expected one event file in runs/{folder}, found {len(files)}')
    scalars = read_scalars(files[0])
    goal = np.array(scalars['Reward/Target']) > 0         # +100 goal bonus
    collision = np.array(scalars['Reward/Collision']) < 0  # -100 collision penalty
    assert len(goal) == len(collision), folder
    return goal, collision


def rolling_rate(flags):
    csum = np.concatenate([[0], np.cumsum(flags)])
    end = np.arange(1, len(flags) + 1)
    start = np.maximum(0, end - WINDOW)
    rate = (csum[end] - csum[start]) / (end - start)
    rate[end < MIN_EPISODES] = np.nan
    return rate


def checkpoint_episode(name):
    return int(re.search(r'_ep(\d+)_', name).group(1))


# ── Plot helpers ──────────────────────────────────────────────────────────────
def new_figure(t, title, subtitle):
    fig, ax = plt.subplots(figsize=(10, 4.4), facecolor=t['surface'])
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.14)
    ax.set_facecolor(t['surface'])
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(t['axis'])
    ax.spines['bottom'].set_linewidth(0.8)
    ax.grid(axis='y', color=t['grid'], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=t['muted'], labelsize=9, length=0, pad=6)
    fig.text(0.07, 0.94, title, color=t['ink'], fontsize=13, fontweight='semibold')
    fig.text(0.07, 0.885, subtitle, color=t['ink2'], fontsize=9.5)
    return fig, ax


def save(fig, t, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=160, facecolor=t['surface'])
    plt.close(fig)
    print(f'  [PNG] {os.path.relpath(path, ROOT)}')


def plot_goal_rate(sessions, t, name):
    fig, ax = new_figure(
        t, 'Share of training episodes that reach the goal',
        f'Rolling mean over {WINDOW} episodes within each session; sessions #1–#6 in '
        'chronological order (labels: session, goal rate over the whole session)')
    offset = 0
    for i, s in enumerate(sessions, start=1):
        n = len(s['goal'])
        x = offset + np.arange(1, n + 1)
        ax.plot(x, 100 * rolling_rate(s['goal']), color=t['series'], linewidth=1.6,
                solid_capstyle='round', solid_joinstyle='round')
        if i > 1:
            ax.axvline(offset + 0.5, color=t['axis'], linewidth=0.8)
        mid = offset + n / 2
        ax.text(mid, 109, f'#{i}', ha='center', va='center', fontsize=9.5,
                color=t['ink'], fontweight='semibold')
        ax.text(mid, 102, f'{100 * s["goal"].mean():.1f}%', ha='center', va='center',
                fontsize=8.5, color=t['ink2'])
        offset += n
    ax.set_xlim(0, offset)
    ax.set_ylim(0, 114)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.set_xlabel('Training episodes, all sessions back to back', color=t['ink2'], fontsize=9.5)
    save(fig, t, name)


def plot_checkpoint_eval(results, best, t, name):
    fig, ax = new_figure(
        t, 'Run-2 checkpoints: average test reward',
        f'{len(results)} checkpoints × {EVAL_EPISODES} episodes with the deterministic '
        'policy (random start, goal ≥ 6 m away, ≤ 300 steps)')
    names = list(results)
    x = np.array([checkpoint_episode(k) for k in names])
    y = np.array([results[k]['average_reward'] for k in names])
    clean = np.array([results[k]['total_collisions'] == 0 for k in names])
    ax.scatter(x[clean], y[clean], s=46, color=t['series'], edgecolors=t['surface'],
               linewidths=1.0, zorder=3, label=f'no collision in {EVAL_EPISODES} episodes')
    ax.scatter(x[~clean], y[~clean], s=46, facecolors=t['surface'], edgecolors=t['series'],
               linewidths=1.5, zorder=3, label=f'1–3 of {EVAL_EPISODES} episodes ended in a collision')
    bx, by = checkpoint_episode(best), results[best]['average_reward']
    ax.annotate(f'best: {best.rsplit("_seed", 1)[0]}  ({by:.1f})', xy=(bx, by),
                xytext=(bx - 80, by + 26), ha='right', va='center', fontsize=8.5,
                color=t['ink2'], arrowprops=dict(arrowstyle='-', color=t['muted'], lw=0.8))
    ax.set_xlim(0, 2050)
    ax.set_ylim(0, 320)
    ax.set_xlabel('Training episode at which the checkpoint was saved', color=t['ink2'], fontsize=9.5)
    ax.set_ylabel('Average episode reward', color=t['ink2'], fontsize=9.5)
    leg = ax.legend(loc='lower right', frameon=False, fontsize=8.5)
    for text in leg.get_texts():
        text.set_color(t['ink2'])
    save(fig, t, name)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    sys.stdout.reconfigure(encoding='utf-8')  # tables use → and × (Windows consoles default to cp1252)
    sessions = []
    for folder, label in SESSIONS:
        goal, collision = load_session(folder)
        sessions.append(dict(label=label, goal=goal, collision=collision))

    with open(os.path.join(SAC_DIR, 'results', 'evaluation_summary.json')) as f:
        summary = json.load(f)
    results = {k: v for k, v in summary['all_results'].items() if 'average_reward' in v}
    best = max(results, key=lambda k: results[k]['average_reward'])

    print('\nFigures:')
    for theme, t in THEMES.items():
        suffix = '' if theme == 'light' else '_dark'
        plot_goal_rate(sessions, t, f'training_goal_rate{suffix}.png')
        plot_checkpoint_eval(results, best, t, f'checkpoint_eval{suffix}.png')

    print('\nTraining sessions:\n')
    print('| # | Session | Episodes | Reached the goal | Ended in a collision '
          f'| Goal rate, first → last {WINDOW} episodes |')
    print('|---|---|--:|--:|--:|:-:|')
    for i, s in enumerate(sessions, start=1):
        n, g, c = len(s['goal']), int(s['goal'].sum()), int(s['collision'].sum())
        print(f'| {i} | {s["label"]} | {n:,} | {g:,} ({100 * g / n:.1f}%) | '
              f'{c:,} ({100 * c / n:.1f}%) | {100 * s["goal"][:WINDOW].mean():.0f}% → '
              f'{100 * s["goal"][-WINDOW:].mean():.0f}% |')
    total = sum(len(s['goal']) for s in sessions)
    print(f'\nTotal: {total:,} episodes')

    rewards = np.array([v['average_reward'] for v in results.values()])
    collisions = np.array([v['total_collisions'] for v in results.values()])
    print(f'\nCheckpoint evaluation ({len(results)} checkpoints × {EVAL_EPISODES} episodes):')
    print(f'  average reward: mean {rewards.mean():.1f}, median {np.median(rewards):.1f}, '
          f'range {rewards.min():.1f} – {rewards.max():.1f}')
    print(f'  checkpoints with no collision: {(collisions == 0).sum()} of {len(results)}; '
          f'episodes ending in a collision: {collisions.sum()} of {EVAL_EPISODES * len(results)} '
          f'({100 * collisions.sum() / (EVAL_EPISODES * len(results)):.1f}%)')
    print('\n| Checkpoint | Average reward | Episodes with a collision |')
    print('|---|--:|--:|')
    for k in sorted(results, key=lambda k: -results[k]['average_reward'])[:5]:
        print(f'| `{k}` | {results[k]["average_reward"]:.1f} | '
              f'{results[k]["total_collisions"]} / {EVAL_EPISODES} |')


if __name__ == '__main__':
    main()
