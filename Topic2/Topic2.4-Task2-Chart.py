# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 13:32:40 2025

@author: vital
"""

import matplotlib.pyplot as plt
import numpy as np

# Data for T=0.7
strategies = [
    "70B CoT Suppression.", "8B CoT Suppression.", "70B Basic CoT", "8B Basic CoT", "8B Self-Consistency."
]
accuracy_07 = [0.54, 0.02, 0.82, 0.62, 0.76]
cost_07 = [0.00115, 0.000149, 0.01084, 0.002293, 0.011315]

# Data for T=0.0
accuracy_00 = [0.60, 0.00, 0.74, 0.64, 0.74]
cost_00 = [0.001474, 0.000149, 0.010214, 0.002193, 0.008778]

x = np.arange(len(strategies))
width = 0.36  # Wider bars for better overlap

fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Pastel colors
pastel_blue = "#9AD0EC"
pastel_mint = "#A3E4DB"

# Plot Accuracy (Top)
rects1 = axs[0].bar(x - width/2, accuracy_07, width, label='T=0.7', color=pastel_blue)
rects2 = axs[0].bar(x + width/2, accuracy_00, width, label='T=0.0', color=pastel_mint)
axs[0].set_ylabel('Accuracy', fontsize=14, fontweight='bold')
axs[0].set_title('LLM Reasoning Strategies Comparison: Accuracy vs. Cost (50 Questions)', fontsize=16)
axs[0].set_xticks(x)
axs[0].set_xticklabels(strategies, fontsize=13)
axs[0].legend(fontsize=12)
axs[0].set_ylim(0, 1.05)
axs[0].grid(axis='y', linestyle='--', alpha=0.7)

# Annotate accuracy bars with black text
for rect in rects1:
    axs[0].text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.02, 
                f"{rect.get_height()*100:.0f}%", ha='center', va='bottom', 
                fontsize=11, color='black', fontweight='bold')
for rect in rects2:
    axs[0].text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.02, 
                f"{rect.get_height()*100:.0f}%", ha='center', va='bottom', 
                fontsize=11, color='black', fontweight='bold')


# Plot Cost (Bottom)
rects3 = axs[1].bar(x - width/2, cost_07, width, label='T=0.7', color=pastel_blue)
rects4 = axs[1].bar(x + width/2, cost_00, width, label='T=0.0', color=pastel_mint)
axs[1].set_ylabel('Token Cost (USD)', fontsize=14, fontweight='bold')
axs[1].set_xlabel('Model / Strategy', fontsize=14, fontweight='bold')
axs[1].set_xticks(x)
axs[1].set_xticklabels(strategies, fontsize=13)
axs[1].legend(fontsize=12)
axs[1].set_ylim(0, max(cost_07 + cost_00) * 1.25)
axs[1].grid(axis='y', linestyle='--', alpha=0.7)

# Annotate cost bars with black text
for rect in rects3:
    axs[1].text(rect.get_x() + rect.get_width()/2, rect.get_height() + max(cost_07 + cost_00)*0.02,
                f"${rect.get_height():.5f}", ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')
for rect in rects4:
    axs[1].text(rect.get_x() + rect.get_width()/2, rect.get_height() + max(cost_07 + cost_00)*0.02,
                f"${rect.get_height():.5f}", ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')


plt.tight_layout()
plt.show()
