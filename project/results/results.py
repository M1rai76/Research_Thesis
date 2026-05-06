import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

strategies = ['combined', 'edge_case', 'role_framing', 'zero_shot', 'cot']
base = [0.744, 0.640, 0.665, 0.677, 0.677]
plus = [0.677, 0.598, 0.610, 0.628, 0.634]
gap  = [0.067, 0.042, 0.055, 0.049, 0.043]

width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Thesis A — Prompt Strategy Results (Llama 3.3 70B on HumanEval+)',
             fontsize=13, fontweight='bold', y=1.02)

# --- Chart 1: Grouped bar (sorted by base desc) ---
sort_idx_base = np.argsort(base)[::-1]
strategies_bar = [strategies[i] for i in sort_idx_base]
base_sorted = [base[i] for i in sort_idx_base]
plus_sorted = [plus[i] for i in sort_idx_base]
gap_sorted_bar = [gap[i] for i in sort_idx_base]

x = np.arange(len(strategies_bar))

bars1 = ax1.bar(x - width/2, base_sorted, width, label='Base pass@1', color='#378ADD', alpha=0.9)
bars2 = ax1.bar(x + width/2, plus_sorted, width, label='Plus pass@1', color='#1D9E75', alpha=0.9)

for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

ax1.set_xticks(x)
ax1.set_xticklabels(strategies_bar, rotation=15, ha='right', fontsize=10)
ax1.set_ylim(0.55, 0.80)
ax1.set_ylabel('pass@1 score')
ax1.set_title('Correctness vs Robustness per Strategy')
ax1.legend()
ax1.spines[['top', 'right']].set_visible(False)

# Add a light grey bracket + label over the 'combined' bar pair
if 'combined' in strategies_bar:
    idx_comb = strategies_bar.index('combined')
    left = x[idx_comb] - width/2
    right = x[idx_comb] + width/2
    top = max(base_sorted[idx_comb], plus_sorted[idx_comb])
    bracket_y = top + 0.015
    ax1.plot([left, left, right, right],
             [bracket_y - 0.002, bracket_y, bracket_y, bracket_y - 0.002],
             color='lightgray', linewidth=1.2, zorder=3)
    ax1.text(x[idx_comb], bracket_y + 0.003,
             f'gap = {gap_sorted_bar[idx_comb]:.3f}',
             ha='center', va='bottom', fontsize=9, color='gray')

# --- Chart 2: Robustness gap dot plot (sorted by gap asc, smallest at top) ---
sort_idx_gap = np.argsort(gap)  # ascending
strategies_gap = [strategies[i] for i in sort_idx_gap]
gap_sorted = [gap[i] for i in sort_idx_gap]

y = np.arange(len(strategies_gap))
colors = ['#E24B4A' if s == 'combined' else '#378ADD' for s in strategies_gap]

ax2.hlines(y=y, xmin=0, xmax=gap_sorted, color='#D3D1C7', linewidth=1.5, zorder=1)
ax2.scatter(gap_sorted, y, color=colors, s=120, zorder=2)

for i, (g, s) in enumerate(zip(gap_sorted, strategies_gap)):
    ax2.text(g + 0.001, i, f'{g:.3f}', va='center', fontsize=9)

ax2.set_yticks(y)
ax2.set_yticklabels(strategies_gap, fontsize=10)
ax2.set_xlabel('Robustness gap (Base − Plus pass@1)')
ax2.set_title('Robustness Gap by Strategy\n(smaller = more robust)')
ax2.spines[['top', 'right']].set_visible(False)
ax2.set_xlim(0, 0.085)

# Small red dashed vertical line to anchor combined gap (use the combined value from original data)
combined_gap_value = next(g for s, g in zip(strategies, gap) if s == 'combined')
ax2.axvline(x=combined_gap_value, color='#E24B4A', linestyle='--', alpha=0.4, linewidth=1.2)

# Put largest gap at the bottom (viewer travels top=safer -> bottom=dangerous)
ax2.invert_yaxis()

red_patch = mpatches.Patch(color='#E24B4A', label='Largest gap (worst robustness)')
ax2.legend(handles=[red_patch], fontsize=9)

plt.tight_layout()
plt.savefig('thesis_a_results.png', dpi=300, bbox_inches='tight')
plt.show()