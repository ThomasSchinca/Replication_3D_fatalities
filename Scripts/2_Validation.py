# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:56:05 2024

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
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_1samp
import pickle
from math import tanh
from scipy.spatial.distance import pdist, squareform
import math
import itertools

# =============================================================================
# Functions
# =============================================================================

def cluster_based_on_threshold(array_list, threshold):
    last_column = np.array([arr[:, 3] for arr in array_list])
    distances = squareform(pdist(last_column, metric='euclidean'))
    cluster_labels = np.full(len(array_list), -1)
    current_cluster = 0
    for i in range(len(array_list)):
        if cluster_labels[i] == -1:
            cluster_labels[i] = current_cluster
            for j in range(i + 1, len(array_list)):
                if distances[i, j] <= threshold:
                    cluster_labels[j] = current_cluster
            current_cluster += 1
    return cluster_labels

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

def calculate_EMD(df_test,df_input_test,weights_cor,coordinates,df_prio,coor):
    df_test= df_test.fillna(0)
    df_test_coor = df_test.reset_index().melt(id_vars='index', var_name='prio', value_name='Value')
    df_test_coor = pd.merge(df_test_coor, df_prio, left_on='prio', right_index=True, how='inner')
    df_test_coor= np.array(df_test_coor[['row', 'col','index','Value']])
    df_test_coor[:,0] = df_test_coor[:,0]-coor['xmin']-88
    df_test_coor[:,1] = df_test_coor[:,1]-coor['ymin']-311
    df_test_coor[:,2] = df_test_coor[:,2]+12
    df_test_coor= df_test_coor[df_test_coor[:,3]>0]
    weights_test = weights_cor[coordinates[:,2]>=12]
    coordinates_test = coordinates[coordinates[:,2]>=12]
    df_test_coor = pd.DataFrame(df_test_coor, columns=['col0', 'col1', 'col2', 'col3'])
    df_true = pd.DataFrame(np.append(coordinates_test,weights_test.reshape(len(weights_test),1),axis=1), columns=['col0', 'col1', 'col2', 'col3'])

    if len(np.array(df_true.iloc[:,:3]))>1:
        coordinates_true = np.array(df_true.iloc[:,:3])
        weights_true = np.array(df_true.iloc[:,3])
        weights_true = weights_true / np.sum(weights_true)
        if len(np.array(df_test_coor.iloc[:,:3]))<1:
            coordinates_2 = np.array(df_true.iloc[:,:3])
            weights_2 = [1/len(coordinates_2)]*len(coordinates_2)
        else:
            coordinates_2 = np.array(df_test_coor.iloc[:,:3])
            weights_2 = np.array(df_test_coor.iloc[:,3])
            weights_2 = weights_2 / np.sum(weights_2)
        d_met_2=ot.dist(coordinates_2,coordinates_true,metric='euclidean')
        d_min_2=ot.emd2(weights_2,weights_true,d_met_2)
    else:
        if len(np.array(df_test_coor.iloc[:,:3]))>1:
            coordinates_2 = np.array(df_test_coor.iloc[:,:3])
            weights_2 = np.array(df_test_coor.iloc[:,3])
            weights_2 = weights_2 / np.sum(weights_2)
            coordinates_true = np.array(df_test_coor.iloc[:,:3])
            weights_true = [1/len(coordinates_true)]*len(coordinates_true)
            d_met_2=ot.dist(coordinates_2,coordinates_true,metric='euclidean')
            d_min_2=ot.emd2(weights_2,weights_true,d_met_2)
        else:
            d_min_2 = 0
    return d_min_2

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
# Active zones identification (input Jan-Dec 2020)
# =============================================================================

thres_gr = 2

traj_bo=[]
for x_b in range(0,array_3d_zero.shape[0]):
    for y_b in range(0,array_3d_zero.shape[1]):
        if (array_3d_zero[x_b,y_b,-48:-36]==0).all()==False :
            traj_bo.append({'x': x_b,'y': y_b})
grouped_points = group_points(traj_bo,thres_gr)

final_traj=[]
for gr_li in grouped_points:
    df_p = pd.DataFrame(gr_li)
    x_min = df_p.iloc[:,0].min()
    x_max = df_p.iloc[:,0].max()
    y_min = df_p.iloc[:,1].min()
    y_max = df_p.iloc[:,1].max()
    if (array_3d_zero[x_min:x_max+1,y_min:y_max+1,-48:-36].sum(axis=2)>0).sum()>1:
        final_traj.append({'xmin':x_min,'xmax':x_max,'ymin': y_min,'ymax':y_max})

dict_final_traj = {i:final_traj[i] for i in range(len(final_traj))}
with open(data_folder+'Results/final_traj_val.pkl', 'wb') as f:
    pickle.dump(dict_final_traj, f)

# =============================================================================
# Matching process
# =============================================================================

dict_inp= []
dict_mat= []
for coor in final_traj:
    sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-48:-36]
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
    for x in range(0,array_3d_zero.shape[0]-2,max(2,int(sub_array.shape[0]/2))):
        for y in range(0,array_3d_zero.shape[1]-2,max(2,int(sub_array.shape[1]/2))):
            for z in range(0,array_3d_zero.shape[2]-60,6):
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
with open(data_folder+'Results/matches_val.pkl', 'wb') as f:
    pickle.dump(dict_save_mat, f)
dict_save_inp = {i:dict_inp[i] for i in range(len(dict_inp))}
with open(data_folder+'Results/input_val.pkl', 'wb') as f:
    pickle.dump(dict_save_inp, f)

# =============================================================================
# Grid search on the validation set (forecast Jan-Jun 2021)
# =============================================================================

h=6

df_grid_search=pd.DataFrame(columns=['Thres_EMD','Thres_NP','Thres_Clu','Min_Match','Err_Group','T_val_Group','MSE','T_val_MSE','EMD','T_val_EMD'])
thres_emd_r = [0.15, 0.17, 0.19]
thres_np_r = np.round(np.arange(0.05, 0.11, 0.02), 3)
thres_clu_r = np.round(np.arange(600,1200,100), 1)
mini_mat_r = np.round(np.arange(7,12,1), 1)
grid_search = list(itertools.product(thres_emd_r, thres_np_r, thres_clu_r, mini_mat_r))

for params in grid_search:
    df_output_tot = pd.DataFrame()
    df_true_tot = pd.DataFrame()

    thres_emd, thres_np, thres_clu, mini_mat = params
    mini_mat = int(round(mini_mat))
    df_output = pd.DataFrame(index=range(h),columns=df_prio.index)
    gr_b_tot=[]
    for enu,df_match in enumerate(dict_mat):
        if len(df_match[(df_match.iloc[:,6]<thres_emd) & (df_match.iloc[:,8]<thres_np)])>mini_mat:
            df_match= df_match[(df_match.iloc[:,6]<thres_emd) & (df_match.iloc[:,8]<thres_np)]
        else:
            df_match= df_match.iloc[:mini_mat,:]
        source = dict_inp[enu]
        source_coor = np.meshgrid(np.arange(source.shape[0]), np.arange(source.shape[1]), np.arange(h), indexing='ij')
        source_np= np.column_stack((source_coor[0].ravel(), source_coor[1].ravel(), source_coor[2].ravel()))
        l_mat=[]
        for i in range(len(df_match)):
            sub_a = array_3d_zero[df_match.iloc[i,0]:df_match.iloc[i,1],df_match.iloc[i,2]:df_match.iloc[i,3],df_match.iloc[i,5]:df_match.iloc[i,5]+h]
            sub_a = np.rot90(sub_a, k=df_match.iloc[i,7], axes=(0, 1))
            matc = array_3d_zero[df_match.iloc[i,0]:df_match.iloc[i,1],df_match.iloc[i,2]:df_match.iloc[i,3],df_match.iloc[i,4]:df_match.iloc[i,5]]
            sub_a = (sub_a - matc.min()) / (matc.max()-matc.min())
            x_coords, y_coords, z_coords = np.meshgrid(np.arange(sub_a.shape[0]), np.arange(sub_a.shape[1]), np.arange(sub_a.shape[2]), indexing='ij')
            x_norm = x_coords.ravel() / (x_coords.max())
            y_norm = y_coords.ravel() / (y_coords.max())
            z_norm = z_coords.ravel() / (z_coords.max())
            sub_a = sub_a.reshape(-1, 1)
            sub_a = np.column_stack((x_norm, y_norm, z_norm, sub_a.ravel()))
            sub_a[:, :3] *= tuple(np.array(list(source.shape[:2])+[h]) - 1)
            if len(sub_a) < source.shape[0]*source.shape[1]*h:
                sub_a[:,:3]=np.round(sub_a[:, :3]).astype(int)
                for row in source_np:
                    if not np.any(np.all(sub_a[:, :3] == row, axis=1)):
                        new_row = np.append(row, 0)
                        sub_a = np.vstack([sub_a, new_row])
                reordered_sub_a = np.zeros_like(sub_a)
                for ki, row in enumerate(source_np):
                    index = np.where(np.all(sub_a[:, :3] == row, axis=1))[0][0]
                    reordered_sub_a[ki] = sub_a[index]
                sub_a = reordered_sub_a.copy()
                if len(sub_a) > source.shape[0]*source.shape[1]*h:
                    sub_a[:,:3]=np.round(sub_a[:, :3]).astype(int)
                    unique_rows, indices, inverse_indices = np.unique(sub_a[:, :3], axis=0, return_index=True, return_inverse=True)
                    means_sub = np.zeros((unique_rows.shape[0], sub_a.shape[1]))
                    means_sub[:, :3] = unique_rows
                    means_sub[:, 3] = np.bincount(inverse_indices, weights=sub_a[:, 3])
                    sub_a = means_sub.copy()
            elif len(sub_a) > source.shape[0]*source.shape[1]*h:
                sub_a[:,:3]=np.round(sub_a[:, :3]).astype(int)
                unique_rows, indices, inverse_indices = np.unique(sub_a[:, :3], axis=0, return_index=True, return_inverse=True)
                means_sub = np.zeros((unique_rows.shape[0], sub_a.shape[1]))
                means_sub[:, :3] = unique_rows
                means_sub[:, 3] = np.bincount(inverse_indices, weights=sub_a[:, 3])
                sub_a = means_sub.copy()
                if len(sub_a) < source.shape[0]*source.shape[1]*h:
                    for row in source_np:
                        if not np.any(np.all(sub_a[:, :3] == row, axis=1)):
                            new_row = np.append(row, 0)
                            sub_a = np.vstack([sub_a, new_row])
                    reordered_sub_a = np.zeros_like(sub_a)
                    for ki, row in enumerate(source_np):
                        index = np.where(np.all(sub_a[:, :3] == row, axis=1))[0][0]
                        reordered_sub_a[ki] = sub_a[index]
                    sub_a = reordered_sub_a.copy()
            l_mat.append(sub_a)

        cluster_labels = cluster_based_on_threshold(l_mat, (source.shape[0]*source.shape[1]*h/thres_clu))
        l_mat_sub = [l_mat[i] for i in range(len(cluster_labels)) if pd.Series(cluster_labels).value_counts().loc[cluster_labels[i]] == pd.Series(cluster_labels).value_counts().max()]
        last_sub_val = np.array([arr[:, 3] for arr in l_mat_sub])
        last_sub_val = last_sub_val.mean(axis=0)
        last_sub_val = last_sub_val*(source.max()-source.min()) + source.min()
        df_g_mean = np.column_stack((source_np,last_sub_val))
        df_g_mean[:,0] = df_g_mean[:,0]+final_traj[enu]['xmin']
        df_g_mean[:,1] = df_g_mean[:,1]+final_traj[enu]['ymin']
        df_g_mean=pd.DataFrame(df_g_mean)
        gr_b=[]
        for i in range(len(df_g_mean)):
            try:
                df_output.loc[int(df_g_mean.iloc[i,2]),df_prio[(df_prio.iloc[:,0]==int(df_g_mean.iloc[i,0])+88) & (df_prio.iloc[:,1]==int(df_g_mean.iloc[i,1])+311)].index]=df_g_mean.iloc[i,3]
                gr_b.append(df_prio[(df_prio.iloc[:,0]==int(df_g_mean.iloc[i,0])+88) & (df_prio.iloc[:,1]==int(df_g_mean.iloc[i,1])+311)].index[0])
            except:
                pass
        gr_b_tot.append(pd.Series(gr_b).unique().tolist())
    mse_ratio_pattern=[]
    emd_ratio=[]
    for enu,list_pat in enumerate(gr_b_tot):
        df_test = df_output.loc[:,list_pat]
        df_input_test = df_input.iloc[-36:-36+h].loc[:,list_pat]

        df_output_tot = pd.concat([df_output_tot,df_test.reset_index(drop=True)],axis=1)
        df_true_tot = pd.concat([df_true_tot,df_input_test.reset_index(drop=True)],axis=1)
        mse_ratio_pattern.append(abs(df_test.sum().sum()- df_input_test.sum().sum()))

        coor = final_traj[enu]
        sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-48:-30]
        non_zero_indices = np.argwhere(sub_array!=0)
        coordinates = [(idx[0],idx[1],idx[2]) for idx in non_zero_indices]
        coordinates = np.array(coordinates)
        weights_1 = sub_array[sub_array != 0]
        emd_ratio.append(calculate_EMD(df_test,df_input_test,weights_1,coordinates,df_prio,coor))
    mse_pg=[]
    for i in range(len(df_output_tot.columns)):
        err_sf = mean_squared_error(df_true_tot.iloc[:,i], df_output_tot.iloc[:,i])
        mse_pg.append(err_sf)

    df_grid_search.loc[len(df_grid_search)] = [params[0],params[1],params[2],params[3],pd.Series(mse_ratio_pattern).mean(),ttest_1samp(pd.Series(mse_ratio_pattern), 0)[0],pd.Series(mse_pg).mean(),ttest_1samp(pd.Series(mse_pg), 0)[0],pd.Series(emd_ratio).mean(),ttest_1samp(pd.Series(emd_ratio), 0)[0]]

df_grid_search.to_csv(data_folder+'Results/df_grid_val.csv')
