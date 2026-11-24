# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 23:04:00 2025

@author: thoma
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
fig_folder = '../Figure/'

import numpy as np
import matplotlib.pyplot as plt
import ot

def numpy_to_latex(matrix):
    rows, cols = matrix.shape

    def format_row(row):
        return " & ".join(map(str, row))

    top_rows = matrix[:3, :]
    bottom_rows = matrix[-3:, :]

    if cols > 6:
        top_rows = np.hstack((top_rows[:, :3], np.full((3,1), '...'), top_rows[:, -3:]))
        bottom_rows = np.hstack((bottom_rows[:, :3], np.full((3,1), '...'), bottom_rows[:, -3:]))
    latex_matrix = "\\begin{bmatrix}\n"
    for i, row in enumerate(top_rows):
        latex_matrix += format_row(row) + " \\\\\n"
    if rows > 6:
        latex_matrix += "\\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots & \\vdots \\\\\n"
    for i, row in enumerate(bottom_rows):
        latex_matrix += format_row(row) + " \\\\\n"
    latex_matrix += "\\end{bmatrix}"
    return latex_matrix

# =============================================================================
# Patterns
# =============================================================================

pattern1 = np.array([
    [[0, 1, 2, 1, 0],
     [2, 9, 6, 5, 2],
     [1, 5, 5, 3, 1],
     [2, 3, 4, 3, 2],
     [0, 1, 2, 1, 0]],

    [[0, 2, 3, 2, 1],
     [1, 5, 5, 4, 3],
     [2, 5, 8, 3, 2],
     [1, 4, 4, 3, 3],
     [1, 2, 1, 2, 1]],

    [[0, 2, 4, 3, 0],
     [2, 2, 6, 5, 4],
     [2, 5, 6, 9, 3],
     [2, 2, 6, 5, 4],
     [2, 2, 4, 3, 0]]
])

pattern2 = np.array([
    [[1, 0, 1, 0, 1],
     [1, 9, 6, 4, 0],
     [0, 6, 3, 3, 3],
     [2, 2, 3, 2, 0],
     [0, 1, 2, 1, 2]],

    [[1, 1, 3, 2, 0],
     [3, 6, 6, 5, 2],
     [3, 6, 9, 2, 2],
     [3, 5, 2, 1, 2],
     [0, 3, 4, 3, 0]],

    [[1, 2, 3, 2, 1],
     [1, 2, 8, 7, 2],
     [1, 4, 8, 9, 5],
     [1, 2, 8, 7, 2],
     [0, 1, 2, 1, 3]]
])

pattern3 = np.array([
    [[1, 1, 1, 1, 1],
     [1, 4, 5, 4, 1],
     [1, 5, 8, 5, 1],
     [1, 4, 5, 4, 1],
     [1, 1, 1, 1, 1]],

    [[1, 1, 1, 1, 1],
     [1, 4, 5, 4, 1],
     [1, 5, 8, 5, 1],
     [1, 4, 5, 4, 1],
     [1, 1, 1, 1, 1]],

    [[1, 1, 1, 1, 1],
     [1, 4, 5, 4, 1],
     [1, 5, 8, 5, 1],
     [1, 4, 5, 4, 1],
     [1, 1, 1, 1, 1]],
])

# =============================================================================
# Appendix J 
# =============================================================================

sub_array = pattern1.copy()
non_zero_indices = np.argwhere(sub_array!=0)
coordinates_1 = [(idx[0],idx[1],idx[2]) for idx in non_zero_indices]
coordinates_1 = np.array(coordinates_1)
weights_1 = sub_array[sub_array != 0]
weights_1 = weights_1 / np.sum(weights_1)

sub_array_2 = pattern2.copy()
zero_indices_2 = np.argwhere(sub_array_2!=0)
coordinates_2 = [(idx[0],idx[1],idx[2]) for idx in zero_indices_2]
coordinates_2 = np.array(coordinates_2)
weights_2 = sub_array_2[sub_array_2!=0]
weights_2 = weights_2 / np.sum(weights_2)

d_met=ot.dist(coordinates_1,coordinates_2,metric='euclidean')
d_sub=ot.emd2(weights_1,weights_2,d_met,return_matrix=True,log=True)
latex_matrix_1 = numpy_to_latex(np.round(d_sub[1]['G'],4))
latex_matrix_2 = numpy_to_latex(np.round(d_met,4))

sub_array_3 = pattern3.copy()
zero_indices_3 = np.argwhere(sub_array_3!=0)
coordinates_3 = [(idx[0],idx[1],idx[2]) for idx in zero_indices_3]
coordinates_3 = np.array(coordinates_3)
weights_3 = sub_array_3[sub_array_3!=0]
weights_3 = weights_3 / np.sum(weights_3)

d_met_2=ot.dist(coordinates_1,coordinates_3,metric='euclidean')
d_sub_2=ot.emd2(weights_1,weights_3,d_met_2,return_matrix=True,log=True)
latex_matrix_3 = numpy_to_latex(np.round(d_sub_2[1]['G'],4))
latex_matrix_4 = numpy_to_latex(np.round(d_met_2,4))

print('EMD pattern 1-2 and 1-3 :', (d_sub[0], d_sub_2[0]))
print('Cell-wise absolute difference 1-2 and 1-3 :', abs(pattern1-pattern2).sum(), abs(pattern1-pattern3).sum())
print('Transport matrix 1-2 :\n', latex_matrix_1)
print('Distance matrix 1-2 :\n', latex_matrix_2)
print('Transport matrix 1-3 :\n', latex_matrix_3)
print('Distance matrix 1-3 :\n', latex_matrix_4)

# Difference after a neighbor-weighted average (kernel)
kernel = np.array(
    [[0.01, 0.05, 0.1, 0.05, 0.01],
     [0.05, 0.3, 0.5, 0.3, 0.05],
     [0.1, 0.5,   1  , 0.5, 0.1],
     [0.05, 0.3, 0.5, 0.3, 0.05],
     [0.01, 0.05, 0.1, 0.05, 0.01]])

sum_tot_1=[]
sum_tot_2=[]
for i in range(3):
    sum_tot_1.append(abs((pattern1[i]*kernel).sum()-(pattern2[i]*kernel).sum()))
    sum_tot_2.append(abs((pattern1[i]*kernel).sum()-(pattern3[i]*kernel).sum()))
print('Kernel difference 1-2 :', sum(sum_tot_1))
print('Kernel difference 1-3 :', sum(sum_tot_2))

# =============================================================================
# Figure 3
# =============================================================================

fig, axes = plt.subplots(3, 3, figsize=(9, 9))
cmap = "Reds"
time_labels = ["Month 1", "Month 2", "Month 3"]
for row, pattern in enumerate([pattern1, pattern2, pattern3]):
    for col in range(3):
        ax = axes[row, col]
        im = ax.imshow(pattern[col], cmap=cmap, vmin=0, vmax=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(time_labels[col],fontsize=20)
axes[0, 0].set_ylabel("Pattern 1", fontsize=20, fontweight="bold")
axes[1, 0].set_ylabel("Pattern 2", fontsize=20, fontweight="bold")
axes[2, 0].set_ylabel("Pattern 3", fontsize=20, fontweight="bold")
plt.tight_layout()
plt.savefig(fig_folder+'Figure_3.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Figure J.1
# =============================================================================

fig, axes = plt.subplots(3, 3, figsize=(9, 9))
cmap = "Reds"
time_labels = ["Month 1", "Month 2", "Month 3"]
for row, pattern in enumerate([pattern1, pattern2, pattern3]):
    for col in range(3):
        ax = axes[row, col]
        im = ax.imshow(pattern[col], cmap=cmap, vmin=0, vmax=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for i in range(pattern[col].shape[0]):
            for j in range(pattern[col].shape[1]):
                color = "white" if pattern[col][i, j] > 5 else "black"
                ax.text(j, i, f"{pattern[col][i, j]:.0f}",
                        ha="center", va="center", color=color, fontsize=10)
        if row == 0:
            ax.set_title(time_labels[col])
axes[0, 0].set_ylabel("Pattern 1", fontsize=12, fontweight="bold")
axes[1, 0].set_ylabel("Pattern 2", fontsize=12, fontweight="bold")
axes[2, 0].set_ylabel("Pattern 3", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(fig_folder+'Figure_J1.png', dpi=300, bbox_inches='tight')
plt.show()
