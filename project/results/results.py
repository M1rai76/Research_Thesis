import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

humaneval = {
    'COP':          {'base': 0.805, 'plus': 0.738, 'gap': 0.067},
    'CoT':          {'base': 0.677, 'plus': 0.634, 'gap': 0.043},
    'Zero Shot':    {'base': 0.677, 'plus': 0.628, 'gap': 0.049},
    'Role Framing': {'base': 0.665, 'plus': 0.610, 'gap': 0.055},
    'Edge Case':    {'base': 0.640, 'plus': 0.598, 'gap': 0.042},
}

mbpp = {
    'Zero Shot':    {'base': 0.876, 'plus': 0.714, 'gap': 0.162},
    'CoT':          {'base': 0.865, 'plus': 0.717, 'gap': 0.148},
    'COP':          {'base': 0.865, 'plus': 0.725, 'gap': 0.140},
    'Role Framing': {'base': 0.857, 'plus': 0.706, 'gap': 0.151},
    'Edge Case':    {'base': 0.860, 'plus': 0.690, 'gap': 0.170},
}

BAR_BASE  = '#378ADD'
BAR_PLUS  = '#1D9E75'
DOT_COLOR = '#378ADD'
DOT_WORST = '#E24B4A'

def plot_benchmark(data, title, filename, dot_xmin):
    sorted_bar = sorted(data.items(), key=lambda x: x[1]['base'], reverse=True)
    labels_bar = [k for k, _ in sorted_bar]
    base_vals  = [v['base'] for _, v in sorted_bar]
    plus_vals  = [v['plus'] for _, v in sorted_bar]

    sorted_dot = sorted(data.items(), key=lambda x: x[1]['gap'])
    labels_dot = [k for k, _ in sorted_dot]
    gap_vals   = [v['gap'] for _, v in sorted_dot]
    worst_gap  = max(gap_vals)

    fig, (ax_bar, ax_dot) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Prompt Strategy Evaluation — {title} (Llama 3.3 70B)',
                 fontsize=13, fontweight='bold', y=1.02)

    x = np.arange(len(labels_bar))
    width = 0.35

    # Grouped bar
    b1 = ax_bar.bar(x - width/2, base_vals, width, label='Base pass@1', color=BAR_BASE, alpha=0.9)
    b2 = ax_bar.bar(x + width/2, plus_vals, width, label='Plus pass@1', color=BAR_PLUS,  alpha=0.9)

    for bar in b1:
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8.5)
    for bar in b2:
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8.5)

    # Gap annotation on worst gap strategy
    worst_gap_label = max(data, key=lambda k: data[k]['gap'])
    wi = labels_bar.index(worst_gap_label)

    mid_y = (base_vals[wi] + plus_vals[wi]) / 2

    ax_bar.annotate('',
        xy=(x[wi] - width/2, base_vals[wi]),
        xytext=(x[wi] - width/2, plus_vals[wi]),
        arrowprops=dict(arrowstyle='<->', color='#E24B4A', lw=1.4))

    ax_bar.text(x[wi] - width/2 - 0.05, mid_y,
                f' gap\n {data[worst_gap_label]["gap"]:.3f}',
                ha='right', va='center', fontsize=8.5,
                color='#E24B4A', fontweight='bold')

    ax_bar.set_ylim(min(plus_vals) - 0.05, max(base_vals) + 0.06)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels_bar, fontsize=10)
    ax_bar.set_ylabel('pass@1 score', fontsize=10)
    ax_bar.set_title('Correctness vs Robustness', fontsize=11, fontweight='bold')
    ax_bar.legend(fontsize=9)
    ax_bar.spines[['top', 'right']].set_visible(False)

    # Dot plot
    colors = [DOT_WORST if g == worst_gap else DOT_COLOR for g in gap_vals]
    ax_dot.hlines(y=labels_dot, xmin=dot_xmin, xmax=gap_vals,
                  color='#D3D1C7', linewidth=1.5, zorder=1)
    ax_dot.scatter(gap_vals, labels_dot, color=colors, s=130, zorder=2)
    ax_dot.axvline(x=worst_gap, color='#E24B4A', linestyle='--', linewidth=1.2, alpha=0.7)

    for g, s in zip(gap_vals, labels_dot):
        ax_dot.text(g + (worst_gap * 0.02), s, f'{g:.3f}', va='center', fontsize=9)

    ax_dot.set_xlabel('Robustness gap (Base − Plus pass@1)', fontsize=10)
    ax_dot.set_title('Robustness Gap by Strategy', fontsize=11, fontweight='bold')
    ax_dot.set_xlim(dot_xmin, worst_gap + worst_gap * 0.15)
    ax_dot.spines[['top', 'right']].set_visible(False)

    red_patch = mpatches.Patch(color=DOT_WORST, label='Largest gap (worst robustness)')
    ax_dot.legend(handles=[red_patch], fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f'Saved: {filename}')

plot_benchmark(humaneval, 'HumanEval+', 'humaneval_results.png', dot_xmin=0.00)
plot_benchmark(mbpp,      'MBPP+',      'mbpp_results.png',      dot_xmin=0.12)