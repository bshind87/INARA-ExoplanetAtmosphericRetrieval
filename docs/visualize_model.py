#!/usr/bin/env python
"""
Generate a publication-quality architecture diagram for SpectralResNet.
Output: docs/resnet_architecture.png  (and .pdf)

Run:  python docs/visualize_model.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
from pathlib import Path

# ── Colour palette ─────────────────────────────────────────────────────────────
C_STEM    = '#1A4A7A'   # dark navy  — Stem
C_S1      = '#2166AC'   # steel blue — Stage 1
C_S2      = '#4393C3'   # mid blue   — Stage 2
C_S3      = '#74ADD1'   # light blue — Stage 3
C_S4      = '#ABD9E9'   # pale blue  — Stage 4
C_POOL    = '#5B8C5A'   # forest green — pooling
C_SHARED  = '#8B4A8B'   # purple     — shared projection
C_HEAD    = '#B2182B'   # crimson    — molecule heads
C_INPUT   = '#2D4A2D'   # dark green — input
C_OUTPUT  = '#7A3A00'   # brown      — output
C_SKIP    = '#E08020'   # orange     — skip connections
C_BG      = '#F7F9FC'
C_TEXT_LIGHT = '#FFFFFF'
C_TEXT_DARK  = '#1A1A2E'
C_ARROW   = '#555555'
C_BORDER  = '#CCCCCC'

FONT = 'DejaVu Sans'

fig = plt.figure(figsize=(22, 14), facecolor=C_BG)
ax  = fig.add_axes([0, 0, 1, 1], facecolor=C_BG)
ax.set_xlim(0, 22)
ax.set_ylim(0, 14)
ax.axis('off')

# ── Helper: rounded box ──────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, color, text_lines, text_color=C_TEXT_LIGHT,
         radius=0.25, alpha=1.0, fontsize=9, border=None):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0",
                         facecolor=color, edgecolor=border or color,
                         linewidth=1.2, alpha=alpha, zorder=3)
    ax.add_patch(box)
    # Text
    n = len(text_lines)
    for i, (line, fs, bold) in enumerate(text_lines):
        ty = y + h/2 + (n/2 - i - 0.5) * (fs * 0.014 + 0.05)
        ax.text(x + w/2, ty, line,
                ha='center', va='center',
                fontsize=fs, fontweight='bold' if bold else 'normal',
                color=text_color, fontfamily=FONT, zorder=4)

def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.8, style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle='arc3,rad=0.0'),
                zorder=5)

def label(ax, x, y, text, color=C_TEXT_DARK, fs=8.5, ha='center', bold=False):
    ax.text(x, y, text, ha=ha, va='center', fontsize=fs,
            color=color, fontfamily=FONT,
            fontweight='bold' if bold else 'normal', zorder=6)

def dim_pill(ax, x, y, text, color='#EEEEEE', tc='#333333'):
    box = FancyBboxPatch((x - 0.55, y - 0.17), 1.1, 0.34,
                         boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor=tc, linewidth=0.8, zorder=5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=7.5, color=tc, fontfamily=FONT, zorder=6)

# ════════════════════════════════════════════════════════════════════════════════
# TITLE
# ════════════════════════════════════════════════════════════════════════════════
ax.text(11, 13.5, 'SpectralResNet — 1D Residual Network for Atmospheric Retrieval',
        ha='center', va='center', fontsize=16, fontweight='bold',
        color=C_TEXT_DARK, fontfamily=FONT)
ax.text(11, 13.05, 'Input: CLIMA profile  (Batch × 12 channels × 101 altitude levels)   →   Output: 12 log₁₀ molecular abundances',
        ha='center', va='center', fontsize=10, color='#555555', fontfamily=FONT)

# Separator line
ax.plot([0.5, 21.5], [12.72, 12.72], color=C_BORDER, lw=1.2)

# ════════════════════════════════════════════════════════════════════════════════
# MAIN VERTICAL FLOW (left column, y from top to bottom)
# ════════════════════════════════════════════════════════════════════════════════
# Column positions
BX, BW = 0.55, 4.2   # main blocks x, width
SX      = 5.2         # shape annotation x
DX      = 6.8         # description x start

# y positions (centre of each block, top to bottom)
YS = [12.15, 10.9, 9.55, 8.2, 6.85, 5.5, 4.3, 3.1]
BH = 0.85

blocks = [
    # (color, title, subtitle)
    (C_INPUT,  'INPUT',          'CLIMA Profile'),
    (C_STEM,   'STEM',           'Conv1d → BN → ReLU'),
    (C_S1,     'STAGE 1  (×2)',  'ResBlock(64→64, stride=1)'),
    (C_S2,     'STAGE 2  (×2)',  'ResBlock(64→128, stride=2)'),
    (C_S3,     'STAGE 3  (×2)',  'ResBlock(128→256, stride=2)'),
    (C_S4,     'STAGE 4  (×2)',  'ResBlock(256→512, stride=2)'),
    (C_POOL,   'GLOBAL AVG POOL', 'AdaptiveAvgPool1d(1) → Flatten'),
    (C_SHARED, 'SHARED PROJ.',   'Dropout(0.3) → FC(512→256) → LN → ReLU'),
]

shapes = [
    ('B, 12, 101',  '#2D4A2D', '#AADDAA'),
    ('B, 64, 101',  C_STEM,    '#BDD5EE'),
    ('B, 64, 101',  C_S1,      '#BDD5EE'),
    ('B, 128, 51',  C_S2,      '#BDD5EE'),
    ('B, 256, 26',  C_S3,      '#BDD5EE'),
    ('B, 512, 13',  C_S4,      '#BDD5EE'),
    ('B, 512',      C_POOL,    '#C8E6C9'),
    ('B, 256',      C_SHARED,  '#E1BEE7'),
]

descs = [
    '12 CLIMA channels   ×   101 altitude levels   (Z-normalised)',
    'Conv1d(12→64, kernel=11, stride=1, pad=5)   +   BatchNorm1d   +   ReLU',
    '2 × ResBlock: [Conv1d→BN→ReLU→Conv1d→BN  +  skip]   stride=1  →  no resolution change',
    '2 × ResBlock: stride=2 first block  →  halves sequence length   64ch→128ch',
    '2 × ResBlock: stride=2  →  halves again   128ch→256ch',
    '2 × ResBlock: stride=2  →  final compression   256ch→512ch   13 altitude pts remain',
    'Squeeze altitude dimension  →  single 512-d vector per sample',
    'Shared representation for all 12 heads   (256-d)',
]

for i, ((col, title, sub), (shape, sc, pill_c), desc) in enumerate(
        zip(blocks, shapes, descs)):
    y_top = YS[i] - BH/2
    # Main block
    rbox(ax, BX, y_top, BW, BH, col,
         [(title, 10, True), (sub, 8.5, False)])
    # Shape pill
    dim_pill(ax, SX + 0.6, YS[i], shape, pill_c, sc)
    # Description
    label(ax, DX, YS[i] + 0.06, desc, color=C_TEXT_DARK, fs=8.5, ha='left')
    # Arrow to next block
    if i < len(YS) - 1:
        arrow(ax, BX + BW/2, y_top, BX + BW/2, YS[i+1] + BH/2 + 0.02,
              color=C_ARROW, lw=2.0)

# ── Column headers ─────────────────────────────────────────────────────────
label(ax, BX + BW/2, 12.52, 'LAYER', color='#888888', fs=8, bold=True)
label(ax, SX + 0.6,  12.52, 'OUTPUT SHAPE', color='#888888', fs=8, bold=True)
label(ax, DX + 3.0,  12.52, 'DESCRIPTION', color='#888888', fs=8, bold=True)
ax.plot([0.5, 14.2], [12.38, 12.38], color=C_BORDER, lw=0.8, ls='--')

# ════════════════════════════════════════════════════════════════════════════════
# RESBLOCK DETAIL  (right panel, middle height)
# ════════════════════════════════════════════════════════════════════════════════
RX = 14.6   # right panel x start

# Panel background
panel = FancyBboxPatch((RX - 0.2, 4.5), 7.3, 7.9,
                        boxstyle="round,pad=0.15",
                        facecolor='#FFFFFF', edgecolor='#CCCCCC',
                        linewidth=1.5, zorder=2)
ax.add_patch(panel)
label(ax, RX + 3.3, 12.18, 'ResBlock1D — Detail', color=C_STEM, fs=12, bold=True)
ax.plot([RX - 0.05, RX + 6.9], [11.9, 11.9], color=C_BORDER, lw=1.0)

# ResBlock internal flow
RBX = RX + 0.6
RBW = 3.8
rb_blocks = [
    (RBX, 11.35, RBW, 0.58, C_STEM,  'CONV 1D',   'kernel=3, stride=s, pad=1'),
    (RBX, 10.45, RBW, 0.58, C_STEM,  'BATCH NORM', 'BatchNorm1d(out_ch)'),
    (RBX,  9.55, RBW, 0.58, C_S1,   'ReLU',        'in-place activation'),
    (RBX,  8.65, RBW, 0.58, C_STEM,  'CONV 1D',    'kernel=3, stride=1, pad=1'),
    (RBX,  7.75, RBW, 0.58, C_STEM,  'BATCH NORM', 'BatchNorm1d(out_ch)'),
]
for (rx, ry, rw, rh, rc, rt, rs) in rb_blocks:
    rbox(ax, rx, ry, rw, rh, rc,
         [(rt, 9.5, True), (rs, 8, False)])

# Arrows inside ResBlock
for i in range(len(rb_blocks) - 1):
    _, y_cur, _, h_cur, _, _, _ = rb_blocks[i]
    _, y_nxt, _, _, _, _, _    = rb_blocks[i+1]
    ax.annotate('', xy=(RBX + RBW/2, y_nxt + 0.58),
                xytext=(RBX + RBW/2, y_cur),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.6), zorder=5)

# Skip connection (right side, goes around)
skip_x = RBX + RBW + 0.35
# vertical line from input top to ADD level
_, y_top_rb, _, h_top_rb, _, _, _ = rb_blocks[0]
y_skip_start = y_top_rb + h_top_rb
y_skip_end   = rb_blocks[-1][1]   # bottom of last block

ax.annotate('', xy=(RBX + RBW/2, y_skip_end - 0.0),
            xytext=(RBX + RBW, y_top_rb + 0.29),
            arrowprops=dict(
                arrowstyle='->', color=C_SKIP, lw=2.2,
                connectionstyle='arc3,rad=-0.55'),
            zorder=5)
label(ax, RBX + RBW + 1.55, (y_skip_start + y_skip_end)/2 + 0.1,
      'SKIP\nCONNECTION', color=C_SKIP, fs=8.5, bold=True, ha='center')

# If size changes label
rbox(ax, RBX + RBW + 0.75, 9.7, 2.05, 0.7,
     '#FFF3E0',
     [('IF stride>1', 7.5, True), ('or in_ch≠out_ch:', 7, False)],
     text_color='#7A4A00', border='#E0A040')
rbox(ax, RBX + RBW + 0.75, 8.85, 2.05, 0.7,
     '#FFF3E0',
     [('1×1 Conv1d', 8, True), ('+ BatchNorm1d', 7.5, False)],
     text_color='#7A4A00', border='#E0A040')

# ADD + ReLU final
add_x = RBX + 0.7
add_y = 7.05
rbox(ax, add_x, add_y, 2.4, 0.56,
     C_S2,
     [('⊕  ADD  +  ReLU', 10, True)],
     text_color='white')
ax.annotate('', xy=(add_x + 1.2, add_y + 0.56),
            xytext=(RBX + RBW/2, rb_blocks[-1][1]),
            arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.6), zorder=5)

# Input label at top of ResBlock
label(ax, RBX + RBW/2, y_top_rb + 0.74, '↓  x  (input to block)', color='#555555', fs=8.5)
# Output label below ADD
label(ax, add_x + 1.2, add_y - 0.22, '↓  output', color='#555555', fs=8.5)

# Equation
label(ax, RX + 3.3, 6.6, 'output = ReLU( F(x) + shortcut(x) )',
      color=C_STEM, fs=10, bold=True, ha='center')
label(ax, RX + 3.3, 6.25, 'F(x) = Conv → BN → ReLU → Conv → BN',
      color='#555555', fs=9, ha='center')

# ════════════════════════════════════════════════════════════════════════════════
# PER-MOLECULE HEADS  (bottom right panel)
# ════════════════════════════════════════════════════════════════════════════════
HP = FancyBboxPatch((RX - 0.2, 0.35), 7.3, 3.85,
                    boxstyle="round,pad=0.15",
                    facecolor='#FFF5F5', edgecolor='#FFAAAA',
                    linewidth=1.5, zorder=2)
ax.add_patch(HP)
label(ax, RX + 3.3, 3.98, 'Per-Molecule Output Heads (×12)', color=C_HEAD, fs=11, bold=True)
ax.plot([RX - 0.05, RX + 6.9], [3.72, 3.72], color='#FFAAAA', lw=1.0)

# Shared representation feeds into heads
rbox(ax, RX + 0.5, 3.25, 5.6, 0.44,
     '#E8D0E8',
     [('Shared 256-d Representation  →  broadcasts to all 12 heads', 9, True)],
     text_color='#5A0A5A', border='#AA66AA')

# 12 molecule heads as mini blocks
mols = ['H₂O', 'CO₂', 'O₂', 'O₃', 'CH₄', 'N₂', 'N₂O', 'CO', 'H₂', 'H₂S', 'SO₂', 'NH₃']
head_depth = {
    'H₂O': 2, 'CO₂': 2, 'O₂': 2, 'O₃': 2, 'CH₄': 2, 'N₂': 2,
    'N₂O': 2, 'CO': 2,  'H₂': 2, 'H₂S': 3, 'SO₂': 3, 'NH₃': 3
}
head_drop = {
    'H₂O': 0.30, 'CO₂': 0.20, 'O₂': 0.20, 'O₃': 0.30, 'CH₄': 0.30,
    'N₂': 0.20,  'N₂O': 0.35, 'CO': 0.30, 'H₂': 0.30, 'H₂S': 0.35,
    'SO₂': 0.40, 'NH₃': 0.40
}
N_HEADS = 12
head_w = (7.3 - 0.5) / N_HEADS
for i, mol in enumerate(mols):
    hx = RX - 0.2 + 0.25 + i * head_w
    hy = 1.65
    hw = head_w - 0.08
    depth = head_depth[mol]
    drop  = head_drop[mol]
    # Color gradient: 2-layer = lighter, 3-layer = darker crimson
    col_h = '#D45A6A' if depth == 3 else '#E8909A'
    rbox(ax, hx, hy, hw, 1.38, col_h,
         [(mol, 8.5, True),
          (f'{depth}L', 7, False),
          (f'p={drop}', 6.5, False)],
         text_color='white', border=C_HEAD)
    # Arrow from shared rep
    ax.annotate('', xy=(hx + hw/2, hy + 1.38),
                xytext=(RX + 0.5 + 5.6 * (i + 0.5)/N_HEADS, 3.25),
                arrowprops=dict(arrowstyle='->', color='#AA66AA', lw=0.9), zorder=5)
    # Output label (scalar)
    label(ax, hx + hw/2, hy - 0.22, 'scalar', color='#888888', fs=7)

label(ax, RX + 3.3, 0.75, 'Output: (Batch, 12)   —   log₁₀ surface volume mixing ratios',
      color=C_HEAD, fs=9.5, bold=True, ha='center')

# Legend: 2L vs 3L heads
rbox(ax, RX + 0.3, 0.42, 1.5, 0.28, '#E8909A',
     [('2-layer head  (p=0.20–0.30)', 7.5, False)], text_color='white')
rbox(ax, RX + 2.0, 0.42, 1.5, 0.28, '#D45A6A',
     [('3-layer head  (p=0.35–0.40)', 7.5, False)], text_color='white')
label(ax, RX + 4.2, 0.56, '← Trace / harder molecules get deeper heads + higher dropout',
      color='#888888', fs=7.5, ha='left')

# ════════════════════════════════════════════════════════════════════════════════
# BOTTOM SECTION — Main flow continues below shared
# ════════════════════════════════════════════════════════════════════════════════
# Arrow from Shared projection block (main left column, y=3.1) down toward heads
shared_block_bottom_y = YS[-1] - BH/2
arrow(ax, BX + BW/2, shared_block_bottom_y, BX + BW/2, 2.05,
      color=C_SHARED, lw=2.0)

# Arrow from bottom of main flow to head panel
rbox(ax, BX, 1.55, BW, 0.6,
     C_SHARED,
     [('HEADS ×12', 10, True), ('12 × MLP → log₁₀ abundance', 8.5, False)])
arrow(ax, BX + BW, 1.85, RX - 0.2, 2.6, color=C_HEAD, lw=2.0, style='->')

# ════════════════════════════════════════════════════════════════════════════════
# SIDE ANNOTATIONS — Loss weights
# ════════════════════════════════════════════════════════════════════════════════
AX2 = 14.35
anno_panel = FancyBboxPatch((AX2 - 0.15, 0.35), 0.2, 0)  # placeholder

# Loss weight note below heads panel
lw_x = 0.55
lw_y = 2.0
rbox(ax, lw_x, lw_y, 4.2, 0.9,
     '#FFF8E1',
     [('Loss: Weighted MSE', 9, True),
      ('w ∈ {1.0 (N₂) … 2.0 (SO₂, NH₃)}', 8.5, False),
      ('Upweights trace / hard molecules', 8, False)],
     text_color='#5A3E00', border='#E0B040')

# Augmentation note
rbox(ax, lw_x, 1.0, 4.2, 0.85,
     '#E8F5E9',
     [('Training augmentation', 9, True),
      ('x  <-  x + N(0, 0.01)  (train only)', 8.5, False),
      ('Forces robust feature learning', 8, False)],
     text_color='#1A4A1A', border='#4CAF50')

# ════════════════════════════════════════════════════════════════════════════════
# OUTER BORDER
# ════════════════════════════════════════════════════════════════════════════════
outer = FancyBboxPatch((0.1, 0.1), 21.8, 13.78,
                        boxstyle="round,pad=0.1",
                        facecolor='none', edgecolor='#CCCCCC',
                        linewidth=2.0, zorder=1)
ax.add_patch(outer)

# Footer
ax.text(11, 0.35, 'CS 6140 · ML · Northeastern University · Spring 2026   |   '
        'Shantanu Wankhare  ·  Bhalchandra Shinde  ·  Asad Mulani',
        ha='center', va='center', fontsize=8.5, color='#888888', fontfamily=FONT)

# ── Save ─────────────────────────────────────────────────────────────────────
out_dir = Path(__file__).parent
fig.savefig(out_dir / 'resnet_architecture.png', dpi=180,
            bbox_inches='tight', facecolor=C_BG)
fig.savefig(out_dir / 'resnet_architecture.pdf',
            bbox_inches='tight', facecolor=C_BG)
print(f'Saved: {out_dir}/resnet_architecture.png')
print(f'Saved: {out_dir}/resnet_architecture.pdf')
plt.close()
