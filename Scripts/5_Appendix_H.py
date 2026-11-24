# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 23:59:53 2025

@author: thoma
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
data_folder = '../Data/'
fig_folder = '../Figure/'
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
np.random.seed(1)
import matplotlib.pyplot as plt
import math

# =============================================================================
# Functions
# =============================================================================

def euclidean_distance(p1, p2):
    return math.sqrt((p2['x'] - p1['x']) ** 2 + (p2['y'] - p1['y']) ** 2)

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

def group_points(points, threshold):
    n = len(points)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if euclidean_distance(points[i], points[j]) <= threshold:
                uf.union(i, j)
    groups = {}
    for i in range(n):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(points[i])
    return list(groups.values())

# =============================================================================
# Data: 3D array of fatalities (PRIO-Grid x month)
# =============================================================================

df_input=pd.read_csv(data_folder+'df_input_pg.csv',index_col=0,parse_dates=True)
df_input.columns = list(map(int, df_input.columns))
df_input.loc[515,148759]=0 #Ethiopia
df_prio = pd.read_csv(data_folder+'prio.csv',index_col=0,parse_dates=True)
df_prio=df_prio.loc[df_input.columns]
df_prio_xy = df_prio.iloc[:,:2]
max_row = df_prio_xy['row'].max()
max_col = df_prio_xy['col'].max()
max_time = len(df_input.index)
array_3d = np.full((max_row, max_col, max_time), np.nan)
for pg_id, (row, col) in df_prio_xy.iterrows():
    input_column = df_input[pg_id]
    for time_index, value in enumerate(input_column):
        array_3d[row-1, col-1, time_index] = value

array_3d=array_3d[df_prio_xy['row'].min():,df_prio_xy['col'].min():,:]
array_3d_zero = np.nan_to_num(array_3d, nan=0)

# =============================================================================
# Figure H.1 
# =============================================================================

thres_gr = 2
r_real=[]
r_cell=[]
r_train=[]
for num in range(120):
    traj_bo=[]
    for x_b in range(0,array_3d_zero.shape[0]):
        for y_b in range(0,array_3d_zero.shape[1]):
            if (array_3d_zero[x_b,y_b,(-18-num):(-6-num)]==0).all()==False :
                traj_bo.append({'x': x_b,'y': y_b})
    r_train.append(len(traj_bo))
    grouped_points = group_points(traj_bo,thres_gr)
    final_traj=[]
    for gr_li in grouped_points:
        df_p = pd.DataFrame(gr_li)
        if len(df_p)>1:
            x_min = df_p.iloc[:,0].min()
            x_max = df_p.iloc[:,0].max()
            y_min = df_p.iloc[:,1].min()
            y_max = df_p.iloc[:,1].max()
            final_traj.append({'xmin':x_min,'xmax':x_max,'ymin': y_min,'ymax':y_max})
    dict_final_traj = {i:final_traj[i] for i in range(len(final_traj))}
    list_pat=[]
    for i in range(len(dict_final_traj)):
        coor = dict_final_traj[i]
        df_prio_sub = df_prio[(df_prio['row']>=int(coor['xmin']+88))&
                              (df_prio['row']<=int(coor['xmax']+88))&
                              (df_prio['col']>=int(coor['ymin']+311))&
                              (df_prio['col']<=int(coor['ymax']+311))]
        list_pat.extend(df_prio_sub.index.tolist())
    list_pat = list(set(list_pat))
    if num==0:
        df_input_test = df_input.iloc[-6:].loc[:,~df_input.columns.isin(list_pat)]
        sum_b = df_input.iloc[-6:].loc[:,df_input.columns.isin(list_pat)].sum().sum()
        sum_cell = (df_input.iloc[-6:].loc[:,df_input.columns.isin(list_pat)].sum()>0).sum()
    else:
        df_input_test = df_input.iloc[(-6-num):-num].loc[:,~df_input.columns.isin(list_pat)]
        sum_b = df_input.iloc[(-6-num):-num].loc[:,df_input.columns.isin(list_pat)].sum().sum()
        sum_cell = (df_input.iloc[(-6-num):-num].loc[:,df_input.columns.isin(list_pat)].sum()>0).sum()
    r_real.append(df_input_test.sum().sum()/sum_b)
    r_cell.append((df_input_test.sum()>0).sum()/sum_cell)

fig, ax1 = plt.subplots(figsize=(10,6))
ax1.plot((1-pd.Series(r_real))*100, label='% of Covered Fatalities', color='darkred')
ax1.plot((1-pd.Series(r_cell))*100, label='% of Covered Active Cells', color='dimgrey')
ax1.set_ylim(0, 100)
ax1.set_ylabel('Percentage (%)', fontsize=18)
ax1.tick_params(axis='y', labelsize=16)
ax2 = ax1.twinx()
ax2.plot(pd.Series(r_train), label='Cells included in the Active Zones', color='black')
ax2.set_ylabel('Number of Cells', fontsize=18)
ax2.tick_params(axis='y', labelsize=16)
ax1.set_xticks([*range(0,140,20)], fontsize=16)
ax1.set_xticklabels(['June 2023','Oct 2021','Feb 2020', 'June 2018',
                     'Oct 2016','Feb 2015','June 2013'], fontsize=14)
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=15)
plt.savefig(fig_folder+'Figure_H1.png', dpi=300, bbox_inches='tight')
plt.show()
