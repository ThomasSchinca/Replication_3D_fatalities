# -*- coding: utf-8 -*-
"""
Created on Mon May 13 18:36:57 2024

@author: thoma
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
data_folder = '../Data/'

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
np.random.seed(1)
import ot
import pickle
from math import tanh
import math

# =============================================================================
# Functions
# =============================================================================

def compute_overlap(row1, row2):
    overlap = 1
    union = 1
    for i in range(3):  # There are 3 dimensions (x, y, z)
        low1, high1 = row1[2*i], row1[2*i+1]
        low2, high2 = row2[2*i], row2[2*i+1]
        intersect_low = max(low1, low2)
        intersect_high = min(high1, high2)
        union_low = min(low1, low2)
        union_high = max(high1, high2)

        if intersect_high > intersect_low:
            overlap *= (intersect_high - intersect_low)
        else:
            overlap = 0
            break
        union *= (union_high - union_low)
    return overlap / union if union > 0 else 0

def filter_overlaps(df):
    while True:
        to_drop = set()
        for i, row1 in df.iterrows():
            for j, row2 in df.iterrows():
                if i < j:
                    overlap_ratio = compute_overlap(row1[:6], row2[:6])
                    if overlap_ratio > 0.25:
                        to_drop.add(i if row1[6] > row2[6] else j)
        if not to_drop:
            break
        df = df.drop(index=to_drop)
    return df

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
n_months = array_3d_zero.shape[2]

thres_gr = 2

# Input window start (in months before the end of the data) and file suffix
periods = [(-24, ''),       # Jan 2023
           (-36, '_2022'),  # Jan 2022
           (-18, '_sub1'),  # July 2023
           (-30, '_sub2')]  # July 2022

# =========================================================================
# Active zones identification
# =========================================================================


for inp_start, suffix in periods:
    inp_end = inp_start + 12

    traj_bo=[]
    for x_b in range(0,array_3d_zero.shape[0]):
        for y_b in range(0,array_3d_zero.shape[1]):
            if (array_3d_zero[x_b,y_b,inp_start:inp_end]==0).all()==False :
                traj_bo.append({'x': x_b,'y': y_b})
    grouped_points = group_points(traj_bo,thres_gr)

    final_traj=[]
    final_traj_all=[]
    for gr_li in grouped_points:
        df_p = pd.DataFrame(gr_li)
        x_min = df_p.iloc[:,0].min()
        x_max = df_p.iloc[:,0].max()
        y_min = df_p.iloc[:,1].min()
        y_max = df_p.iloc[:,1].max()
        final_traj_all.append({'xmin':x_min,'xmax':x_max,'ymin': y_min,'ymax':y_max})
        if len(df_p)>1: # Isolated active cells are not considered as zones
            final_traj.append({'xmin':x_min,'xmax':x_max,'ymin': y_min,'ymax':y_max})

    dict_final_traj = {i:final_traj[i] for i in range(len(final_traj))}
    with open(data_folder+f'Results/final_traj{suffix}.pkl', 'wb') as f:
        pickle.dump(dict_final_traj, f)
    if suffix == '':
        # All zones, including isolated cells (Figure 4c)
        all_dict_final_traj = {i:final_traj_all[i] for i in range(len(final_traj_all))}
        with open(data_folder+'Results/all_final_traj.pkl', 'wb') as f:
            pickle.dump(all_dict_final_traj, f)

    # =========================================================================
    # Matching process
    # =========================================================================

    dict_inp= []
    dict_mat= []
    for coor in final_traj:
        sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,inp_start:inp_end]
        dict_inp.append(sub_array)
        bound_1=np.array([[0,0,0],list(sub_array.shape)])
        non_zero_indices = np.argwhere(sub_array!=0)
        coordinates_1 = [(idx[0],idx[1],idx[2]) for idx in non_zero_indices]
        coordinates_1 = np.array(coordinates_1)
        coordinates_1 = (coordinates_1 - bound_1.min(axis=0)) / (bound_1.max(axis=0)-1 - bound_1.min(axis=0))
        coordinates_1 = np.nan_to_num(coordinates_1, nan=0.5) # one-cell-wide zones (0/0), as for coordinates_2
        if len(non_zero_indices)>1:
            weights_1 = sub_array[sub_array != 0]
            weights_1 = weights_1 / np.sum(weights_1)
        else:
            weights_1=np.array([1])
        dist_arr=[]
        # Historical search: the sequences and their 6-month past future end before the input window
        for x in range(0,array_3d_zero.shape[0]-2,max(2,int(sub_array.shape[0]/2))):
            for y in range(0,array_3d_zero.shape[1]-2,max(2,int(sub_array.shape[1]/2))):
                for z in range(0,n_months+inp_start-6,6):
                    for x_r in [-int(sub_array.shape[0]/4),0,int(sub_array.shape[0]/4)]:
                        for y_r in [-int(sub_array.shape[1]/4),0,int(sub_array.shape[1]/4)]:
                            for z_r in [-3,0,3]:
                                sub_array_2 = array_3d_zero[x:x+x_r+sub_array.shape[0],y:y+y_r+sub_array.shape[1],z:z+z_r+sub_array.shape[2]]
                                bound=np.array([[0,0,0],list(sub_array_2.shape)])
                                zero_indices_2 = np.argwhere(sub_array_2!=0)
                                if ((sub_array_2.flatten()==0).all()==False) & (abs(tanh(np.log(len(zero_indices_2)/len(non_zero_indices)))) <= 0.6):
                                    coordinates_2 = [(idx[0],idx[1],idx[2]) for idx in zero_indices_2]
                                    coordinates_2 = np.array(coordinates_2)
                                    coordinates_2 = (coordinates_2 - bound.min(axis=0)) / (bound.max(axis=0)-1 - bound.min(axis=0))
                                    coordinates_2 = np.nan_to_num(coordinates_2, nan=0.5)
                                    if len(zero_indices_2)>1:
                                        weights_2 = sub_array_2[sub_array_2!=0]
                                        weights_2 = weights_2 / np.sum(weights_2)
                                    else:
                                        weights_2=np.array([1])
                                    d_met=ot.dist(coordinates_1,coordinates_2,metric='euclidean')
                                    d_min=ot.emd2(weights_1,weights_2,d_met)
                                    best_rota=0
                                    if len(zero_indices_2)>1:
                                        for i in range(3):
                                            sub_array_3 = np.rot90(sub_array_2, k=i+1, axes=(0, 1))
                                            non_zero_indices_3 = np.argwhere(sub_array_3 != 0)
                                            bound=np.array([[0,0,0],list(sub_array_3.shape)])
                                            coordinates_3 = [(idx[0], idx[1], idx[2]) for idx in non_zero_indices_3]
                                            coordinates_3 = np.array(coordinates_3)
                                            coordinates_3 = (coordinates_3 - bound.min(axis=0)) / (bound.max(axis=0)-1 - bound.min(axis=0))
                                            coordinates_3 = np.nan_to_num(coordinates_3, nan=0.5)
                                            weights_3 = sub_array_3[sub_array_3!=0]
                                            weights_3 = weights_3 / np.sum(weights_3)
                                            d_met=ot.dist(coordinates_1,coordinates_3,metric='euclidean')
                                            d_sub=ot.emd2(weights_1,weights_3,d_met)
                                            if d_min>d_sub:
                                                d_min=d_sub
                                                best_rota=i+1
                                    else:
                                        best_rota=0
                                    dist_arr.append([x,x+x_r+sub_array.shape[0],y,y+y_r+sub_array.shape[1],z,z+z_r+sub_array.shape[2],d_min,best_rota,abs(tanh(np.log(len(zero_indices_2)/len(non_zero_indices))))])
        dist_arr=pd.DataFrame(dist_arr)
        dist_arr['Sum'] = dist_arr[6] + dist_arr[8]
        dist_arr=dist_arr.sort_values(['Sum'])
        dist_arr = dist_arr.iloc[:500,:]
        dist_arr = filter_overlaps(dist_arr)
        dict_mat.append(dist_arr)

    dict_save_mat = {i:dict_mat[i] for i in range(len(dict_mat))}
    with open(data_folder+f'Results/matches{suffix}.pkl', 'wb') as f:
        pickle.dump(dict_save_mat, f)
    dict_save_inp = {i:dict_inp[i] for i in range(len(dict_inp))}
    with open(data_folder+f'Results/input{suffix}.pkl', 'wb') as f:
        pickle.dump(dict_save_inp, f)
