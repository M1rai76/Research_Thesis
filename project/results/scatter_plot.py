import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 5))

he = {
    'COP':          (0.805, 0.067),
    'CoT':          (0.677, 0.043),
    'Zero Shot':    (0.677, 0.049),
    'Role Framing': (0.665, 0.055),
    'Edge Case':    (0.640, 0.042),
}

mbpp = {
    'COP':          (0.865, 0.140),
    'CoT':          (0.865, 0.148),
    'Zero Shot':    (0.876, 0.162),
    'Role Framing': (0.857, 0.151),
    'Edge Case':    (0.860, 0.170),
}

for label, (x, y) in he.items():
    ax.scatter(x, y, color='#378ADD', s=100, zorder=3)
    ax.annotate(label, (x, y), textcoords='offset points',
                xytext=(6, 4), fontsize=8, color='#378ADD')

for label, (x, y) in mbpp.items():
    ax.scatter(x, y, color='#E24B4A', s=100, zorder=3)
    ax.annotate(label, (x, y), textcoords='offset points',
                xytext=(6, 4), fontsize=8, color='#E24B4A')

# Highlight the empty ideal quadrant
ax.axhspan(0, 0.04, xmin=0.7, xmax=1.0,
           alpha=0.08, color='green')
ax.text(0.85, 0.02, 'ideal zone\n(empty)',
        ha='center', fontsize=8, color='green', style='italic')

ax.set_xlabel('Correctness (Base pass@1)', fontsize=10)
ax.set_ylabel('Robustness Gap', fontsize=10)
ax.set_title('No strategy occupies the ideal zone', fontsize=11, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)

from matplotlib.lines import Line2D
legend = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#378ADD', markersize=9, label='HumanEval+'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#E24B4A', markersize=9, label='MBPP+'),
]
ax.legend(handles=legend, fontsize=9)

plt.tight_layout()
plt.savefig('key_findings_scatter.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()