import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

ORACLE_BG    = '#FAD7A0'
ORACLE_EDGE  = '#CA6F1E'
METRIC_BG    = '#D2B4DE'
METRIC_EDGE  = '#7D3C98'
ARROW_COLOR  = '#555555'
TITLE_COLOR  = '#1A1A1A'

def draw_card(ax, x, y, w, h, title, subtitle, body, metric, bg, edge):
    rect = mpatches.FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.1",
                                    linewidth=1.5,
                                    edgecolor=edge,
                                    facecolor=bg)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.28, title,
            ha='center', va='top', fontsize=13, fontweight='bold', color=TITLE_COLOR)
    ax.text(x + w/2, y + h - 0.65, subtitle,
            ha='center', va='top', fontsize=10, color='#6E2F00', style='italic')
    ax.text(x + w/2, y + h - 1.15, body,
            ha='center', va='top', fontsize=10, color='#333333')
    ax.text(x + w/2, y + 0.28, metric,
            ha='center', va='bottom', fontsize=10.5,
            fontweight='bold', color=ORACLE_EDGE,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=ORACLE_EDGE, linewidth=1))

def draw_metric_card(ax, x, y, w, h, title, formula, note, bg, edge):
    rect = mpatches.FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.1",
                                    linewidth=1.5,
                                    edgecolor=edge,
                                    facecolor=bg)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.28, title,
            ha='center', va='top', fontsize=13, fontweight='bold', color=TITLE_COLOR)
    ax.text(x + w/2, y + h/2, formula,
            ha='center', va='center', fontsize=12,
            fontweight='bold', color='#5B2C8D',
            fontfamily='monospace')
    ax.text(x + w/2, y + 0.28, note,
            ha='center', va='bottom', fontsize=9.5,
            color='#6C3483', style='italic')

# Oracle Cards (top row)
oracles = [
    ("Correctness Oracle", "Unit Tests",    "Runs standard\nHumanEval+ test suite",    "→ Base pass@1"),
    ("Robustness Oracle",  "Stress Inputs", "80× edge cases\nvia EvalPlus expansion",  "→ Plus pass@1"),
    ("Safety Oracle",      "AST Scan",      "Checks for input guards\n& validation",   "→ Guard present?"),
]

card_w, card_h = 3.6, 2.8
gap = 0.6
start_x = (14 - (3 * card_w + 2 * gap)) / 2
top_y = 2.9

for i, (title, subtitle, body, metric) in enumerate(oracles):
    cx = start_x + i * (card_w + gap)
    draw_card(ax, cx, top_y, card_w, card_h, title, subtitle, body, metric,
              ORACLE_BG, ORACLE_EDGE)

# Arrows from oracle cards to metric row
metric_centers = [3.2, 10.8]
arrow_origins  = [start_x + 0.5*card_w, start_x + card_w + gap + 0.5*card_w,
                  start_x + 2*(card_w + gap) + 0.5*card_w]

for ox in arrow_origins[:2]:
    ax.annotate("", xy=(metric_centers[0], 2.75), xytext=(ox, top_y),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=1.2))

ax.annotate("", xy=(metric_centers[1], 2.75), xytext=(arrow_origins[2], top_y),
            arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=1.2))

# Metric Cards (bottom row)
m_card_w, m_card_h = 3.8, 2.4
draw_metric_card(ax, 1.3, 0.2, m_card_w, m_card_h,
                 "Robustness Gap",
                 "Base pass@1 − Plus pass@1",
                 "absolute drop under stress  ·  lower = better",
                 METRIC_BG, METRIC_EDGE)

draw_metric_card(ax, 8.9, 0.2, m_card_w, m_card_h,
                 "Robustness Ratio",
                 "Plus pass@1 / Base pass@1",
                 "normalised retention  ·  higher = better",
                 METRIC_BG, METRIC_EDGE)

# Slide title
ax.text(7, 5.82, "Evaluation Framework — Metrics & Oracles",
        ha='center', va='top', fontsize=15, fontweight='bold', color=TITLE_COLOR)

plt.tight_layout()
plt.savefig('metrics_oracles_slide.png', dpi=300, bbox_inches='tight',
            facecolor='white')
plt.show()