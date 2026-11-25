# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:21:45 2024

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
import ot
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_1samp, t
import pickle
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from scipy.spatial import distance
from scipy.sparse.csgraph import minimum_spanning_tree
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import plotly.graph_objects as go
from plotly.offline import plot
import scipy.stats as stats
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import trimesh
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

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

def mean_and_ci(data):
    mean = np.mean(data)
    ci = stats.sem(data) * stats.t.ppf((1+0.95)/2,len(data)-1)
    return mean, ci

def calculate_EMD(df_test,df_input_test,df_preds_test_2_test,weights_cor,coordinates,df_prio,year,coor):
    df_test= df_test.fillna(0)
    df_views_coor = df_preds_test_2_test.reset_index().melt(id_vars='month_id', var_name='prio', value_name='Value')
    df_views_coor = pd.merge(df_views_coor, df_prio, left_on='prio', right_index=True, how='inner')
    df_views_coor= np.array(df_views_coor[['row', 'col','month_id','Value']])
    df_views_coor[:,0] = df_views_coor[:,0]-coor['xmin']-88
    df_views_coor[:,1] = df_views_coor[:,1]-coor['ymin']-311
    if year == 2022:
        df_views_coor[:,2] = df_views_coor[:,2]-505+12
    elif year == 2023:
        df_views_coor[:,2] = df_views_coor[:,2]-505
    elif year == '2023_s':
        df_views_coor[:,2] = df_views_coor[:,2]-505-6
    elif year == '2022_s':
        df_views_coor[:,2] = df_views_coor[:,2]-505+6
        
    df_views_coor= df_views_coor[df_views_coor[:,3]>0]
    
    df_test_coor = df_test.reset_index().melt(id_vars='index', var_name='prio', value_name='Value')
    df_test_coor = pd.merge(df_test_coor, df_prio, left_on='prio', right_index=True, how='inner')
    df_test_coor= np.array(df_test_coor[['row', 'col','index','Value']])
    df_test_coor[:,0] = df_test_coor[:,0]-coor['xmin']-88
    df_test_coor[:,1] = df_test_coor[:,1]-coor['ymin']-311
    df_test_coor[:,2] = df_test_coor[:,2]+12
    df_test_coor= df_test_coor[df_test_coor[:,3]>0]
    weights_test = weights_cor[coordinates[:,2]>=12]
    coordinates_test = coordinates[coordinates[:,2]>=12]
    
    df_views_coor = pd.DataFrame(df_views_coor, columns=['col0', 'col1', 'col2', 'col3'])
    df_test_coor = pd.DataFrame(df_test_coor, columns=['col0', 'col1', 'col2', 'col3'])
    df_true = pd.DataFrame(np.append(coordinates_test,weights_test.reshape(len(weights_test),1),axis=1), columns=['col0', 'col1', 'col2', 'col3'])
    
    if len(np.array(df_true.iloc[:,:3]))>1:
        
        coordinates_true = np.array(df_true.iloc[:,:3])
        weights_true = np.array(df_true.iloc[:,3])
        weights_true = weights_true / np.sum(weights_true)
        
        if len(np.array(df_views_coor.iloc[:,:3]))<1:
            coordinates_1 = np.array(df_true.iloc[:,:3])
            weights_1 = [1/len(coordinates_1)]*len(coordinates_1)
        else:
            coordinates_1 = np.array(df_views_coor.iloc[:,:3])
            weights_1 = np.array(df_views_coor.iloc[:,3])
            weights_1 = weights_1 / np.sum(weights_1)
            
        if len(np.array(df_test_coor.iloc[:,:3]))<1:
            coordinates_2 = np.array(df_true.iloc[:,:3])
            weights_2 = [1/len(coordinates_2)]*len(coordinates_2)
        else:
            coordinates_2 = np.array(df_test_coor.iloc[:,:3])
            weights_2 = np.array(df_test_coor.iloc[:,3])
            weights_2 = weights_2 / np.sum(weights_2)

        d_met_1=ot.dist(coordinates_1,coordinates_true,metric='euclidean')
        d_min_1=ot.emd2(weights_1,weights_true,d_met_1)
        
        d_met_2=ot.dist(coordinates_2,coordinates_true,metric='euclidean')
        d_min_2=ot.emd2(weights_2,weights_true,d_met_2)
    
    else:
        if len(np.array(df_views_coor.iloc[:,:3]))>1:
            coordinates_1 = np.array(df_views_coor.iloc[:,:3])
            weights_1 = np.array(df_views_coor.iloc[:,3])
            weights_1 = weights_1 / np.sum(weights_1)
            coordinates_true = np.array(df_views_coor.iloc[:,:3])
            weights_true = [1/len(coordinates_true)]*len(coordinates_true)
            d_met_1=ot.dist(coordinates_1,coordinates_true,metric='euclidean')
            d_min_1=ot.emd2(weights_1,weights_true,d_met_1)
        else:
            d_min_1 = 0 
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
    return np.log((d_min_1+1)/(d_min_2+1))

def calculate_EMD_d(df_test,df_input_test,weights_cor,coordinates,df_prio,year,coor):
    df_test.reset_index(drop=True,inplace=True)
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
# Data: 3D array of fatalities 
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
array_3d_na = array_3d.copy()
array_3d_na[array_3d_na==0]=np.nan
array_3d_zero = np.nan_to_num(array_3d, nan=0)

# =============================================================================
# Load VIEWS forecasts and matching results
# =============================================================================

h=6
df_preds_views = pd.read_csv(data_folder+'views_pred_2pg.csv',index_col=0,parse_dates=True) 
df_preds_views.columns = map(int,df_preds_views.columns)   
df_preds_views_first = pd.read_csv(data_folder+'views_pred_1pg.csv',index_col=0,parse_dates=True) 
df_preds_views_first.columns = map(int,df_preds_views_first.columns)   

df_preds_views_half_1 = pd.read_csv(data_folder+'views_pred_06_2023.csv',index_col=0,parse_dates=True) 
df_preds_views_half_1.columns = map(int,df_preds_views_first.columns)  
df_preds_views_half_2 = pd.read_csv(data_folder+'views_pred_06_2022.csv',index_col=0,parse_dates=True) 
df_preds_views_half_2.columns = map(int,df_preds_views_first.columns) 

with open(data_folder+'Results/matches.pkl', 'rb') as f:
    dict_save_mat = pickle.load(f)
with open(data_folder+'Results/input.pkl', 'rb') as f:
    dict_save_inp = pickle.load(f)  
dict_inp = [dict_save_inp[key] for key in dict_save_inp.keys()]
dict_mat = [dict_save_mat[key] for key in dict_save_mat.keys()]    
with open(data_folder+'Results/matches_2022.pkl', 'rb') as f:
    dict_save_mat_2022 = pickle.load(f)
with open(data_folder+'Results/input_2022.pkl', 'rb') as f:
    dict_save_inp_2022 = pickle.load(f)  
dict_inp_2022 = [dict_save_inp_2022[key] for key in dict_save_inp_2022.keys()]
dict_mat_2022 = [dict_save_mat_2022[key] for key in dict_save_mat_2022.keys()] 
with open(data_folder+'Results/matches_sub1.pkl', 'rb') as f:
    dict_save_mat_2023_sub = pickle.load(f)
with open(data_folder+'Results/input_sub1.pkl', 'rb') as f:
    dict_save_inp_2023_sub = pickle.load(f)  
dict_inp_sub1 = [dict_save_inp_2023_sub[key] for key in dict_save_inp_2023_sub.keys()]
dict_mat_sub1 = [dict_save_mat_2023_sub[key] for key in dict_save_mat_2023_sub.keys()]
with open(data_folder+'Results/matches_sub2.pkl', 'rb') as f:
    dict_save_mat_2022_sub = pickle.load(f)
with open(data_folder+'Results/input_sub2.pkl', 'rb') as f:
    dict_save_inp_2022_sub = pickle.load(f)  
dict_inp_sub2 = [dict_save_inp_2022_sub[key] for key in dict_save_inp_2022_sub.keys()]
dict_mat_sub2 = [dict_save_mat_2022_sub[key] for key in dict_save_mat_2022_sub.keys()]

with open(data_folder+'Results/final_traj.pkl', 'rb') as f:
    dict_final_traj = pickle.load(f)
with open(data_folder+'Results/final_traj_2022.pkl', 'rb') as f:
    dict_final_traj_2022 = pickle.load(f)
with open(data_folder+'Results/final_traj_sub1.pkl', 'rb') as f:
    dict_final_traj_sub1 = pickle.load(f)
with open(data_folder+'Results/final_traj_sub2.pkl', 'rb') as f:
    dict_final_traj_sub2 = pickle.load(f)
with open(data_folder+'Results/all_final_traj.pkl', 'rb') as f:
    dict_final_traj_all = pickle.load(f)
    
final_traj = [dict_final_traj[key] for key in dict_final_traj.keys()]
all_final_traj = [dict_final_traj_all[key] for key in dict_final_traj_all.keys()]
final_traj_2022 = [dict_final_traj_2022[key] for key in dict_final_traj_2022.keys()] 
final_traj_sub1 = [dict_final_traj_sub1[key] for key in dict_final_traj_sub1.keys()]
final_traj_sub2 = [dict_final_traj_sub2[key] for key in dict_final_traj_sub2.keys()]

# =============================================================================
# Forecast creation for the four test periods
# =============================================================================

h=6
df_grid_search = pd.read_csv(data_folder+'Results/df_grid_val.csv',index_col=0)
# Parameters with the lowest mean MSE at the PRIO-Grid level
best_params = df_grid_search.loc[df_grid_search['MSE'].idxmin()].iloc[:4]
thres_emd, thres_np, thres_clu, mini_mat = best_params
mini_mat = int(mini_mat) 

df_output_tot = pd.DataFrame()
df_views_tot = pd.DataFrame()
df_true_tot = pd.DataFrame()
df_t_1 = pd.DataFrame()
df_ave = pd.DataFrame()

df_output_2023 = pd.DataFrame(index=range(h),columns=df_prio.index)
gr_b_tot_2023=[]
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
            df_output_2023.loc[int(df_g_mean.iloc[i,2]),df_prio[(df_prio.iloc[:,0]==int(df_g_mean.iloc[i,0])+88) & (df_prio.iloc[:,1]==int(df_g_mean.iloc[i,1])+311)].index]=df_g_mean.iloc[i,3]
            gr_b.append(df_prio[(df_prio.iloc[:,0]==int(df_g_mean.iloc[i,0])+88) & (df_prio.iloc[:,1]==int(df_g_mean.iloc[i,1])+311)].index[0])
        except:
            pass #if prio grid does not exist 
    gr_b_tot_2023.append(pd.Series(gr_b).unique().tolist())
    
mse_ratio_pattern_2023=[]
emd_ratio_2023=[]

mae_pattern = []
emd_pattern = []

for j_t,list_pat in enumerate(gr_b_tot_2023):
    df_test = df_output_2023.loc[:,list_pat]
    df_test= df_test.fillna(0)
    df_input_test = df_input.iloc[-12:-12 + h].loc[:,list_pat] if h < 12 else df_input.iloc[-12:].loc[:,list_pat]
    df_preds_test_2_test = df_preds_views.iloc[:h].loc[:,list_pat]
    df_t_1_sub = df_input.iloc[-18:-12].loc[:,list_pat]
    df_t_1_sub.index = df_preds_test_2_test.index
    avg_row = df_input.iloc[-24:-12].loc[:, list_pat].mean()
    df_avg = pd.DataFrame([avg_row] * 6)
    df_avg.index = df_preds_test_2_test.index
    
    df_output_tot = pd.concat([df_output_tot,df_test.reset_index(drop=True)],axis=1)
    df_true_tot = pd.concat([df_true_tot,df_input_test.reset_index(drop=True)],axis=1)
    df_views_tot= pd.concat([df_views_tot,df_preds_test_2_test.reset_index(drop=True)],axis=1)
    df_t_1 = pd.concat([df_t_1,df_t_1_sub.reset_index(drop=True)],axis=1)
    df_ave = pd.concat([df_ave,df_avg.reset_index(drop=True)],axis=1)
    
    m_views=abs(df_preds_test_2_test.sum().sum() - df_input_test.sum().sum())
    m_us=abs(df_test.sum().sum()- df_input_test.sum().sum())
    m_t1=abs(df_t_1_sub.sum().sum()- df_input_test.sum().sum())
    m_ave=abs(df_avg.sum().sum()- df_input_test.sum().sum())
    mse_ratio_pattern_2023.append(np.log((m_views+1)/(m_us+1)))
    
    coor = final_traj[j_t]
    sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-24:-6]
    non_zero_indices = np.argwhere(sub_array!=0)
    coordinates = [(idx[0],idx[1],idx[2]) for idx in non_zero_indices]
    coordinates = np.array(coordinates)
    weights_1 = sub_array[sub_array != 0]
    emd_ratio_2023.append(calculate_EMD(df_test,df_input_test,df_preds_test_2_test,weights_1,coordinates,df_prio,year=2023,coor=coor))
    
    mae_pattern.append([m_us,m_views,m_t1,m_ave])
    emd_t1 = calculate_EMD_d(df_t_1_sub,df_input_test,weights_cor=weights_1,coordinates=coordinates,df_prio=df_prio,year=2023,coor=coor)
    emd_ave = calculate_EMD_d(df_avg,df_input_test,weights_1,coordinates,df_prio,year=2023,coor=coor)
    emd_us = calculate_EMD_d(df_test,df_input_test,weights_1,coordinates,df_prio,year=2023,coor=coor)
    emd_views = calculate_EMD_d(df_preds_test_2_test,df_input_test,weights_1,coordinates,df_prio,year=2023,coor=coor)
    emd_pattern.append([emd_us,emd_views,emd_t1,emd_ave])
    
    
df_metrics_2023 = pd.DataFrame([mse_ratio_pattern_2023,emd_ratio_2023]).T
df_metrics_2023.columns=['Ratio_Err','Ratio_EMD']

df_output_2022 = pd.DataFrame(index=range(h),columns=df_prio.index)
gr_b_tot_2022=[]
for enu,df_match in enumerate(dict_mat_2022):
    if len(df_match[(df_match.iloc[:,6]<thres_emd) & (df_match.iloc[:,8]<thres_np)])>mini_mat:
        df_match= df_match[(df_match.iloc[:,6]<thres_emd) & (df_match.iloc[:,8]<thres_np)]
    else:
        df_match= df_match.iloc[:mini_mat,:]
    source = dict_inp_2022[enu]
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
    df_g_mean[:,0] = df_g_mean[:,0]+final_traj_2022[enu]['xmin']
    df_g_mean[:,1] = df_g_mean[:,1]+final_traj_2022[enu]['ymin']
    df_g_mean=pd.DataFrame(df_g_mean)
    gr_b=[]
    for i in range(len(df_g_mean)):
        try:
            df_output_2022.loc[int(df_g_mean.iloc[i,2]),df_prio[(df_prio.iloc[:,0]==int(df_g_mean.iloc[i,0])+88) & (df_prio.iloc[:,1]==int(df_g_mean.iloc[i,1])+311)].index]=df_g_mean.iloc[i,3]
            gr_b.append(df_prio[(df_prio.iloc[:,0]==int(df_g_mean.iloc[i,0])+88) & (df_prio.iloc[:,1]==int(df_g_mean.iloc[i,1])+311)].index[0])
        except:
            pass #if prio grid does not exist 
    gr_b_tot_2022.append(pd.Series(gr_b).unique().tolist())
    
mse_ratio_pattern_2022=[]
emd_ratio_2022=[]
for j_t,list_pat in enumerate(gr_b_tot_2022):
    df_test = df_output_2022.loc[:,list_pat]
    df_test= df_test.fillna(0)
    df_input_test = df_input.iloc[-24:-24 + h].loc[:,list_pat]
    df_preds_test_2_test = df_preds_views_first.iloc[:h].loc[:,list_pat]
    df_t_1_sub = df_input.iloc[-30:-24].loc[:,list_pat]
    df_t_1_sub.index = df_preds_test_2_test.index
    avg_row = df_input.iloc[-36:-24].loc[:, list_pat].mean()
    df_avg = pd.DataFrame([avg_row] * 6)
    df_avg.index = df_preds_test_2_test.index
    
    df_output_tot = pd.concat([df_output_tot,df_test.reset_index(drop=True)],axis=1)
    df_true_tot = pd.concat([df_true_tot,df_input_test.reset_index(drop=True)],axis=1)
    df_views_tot= pd.concat([df_views_tot,df_preds_test_2_test.reset_index(drop=True)],axis=1)
    df_t_1 = pd.concat([df_t_1,df_t_1_sub.reset_index(drop=True)],axis=1)
    df_ave = pd.concat([df_ave,df_avg.reset_index(drop=True)],axis=1)
    
    m_views=abs(df_preds_test_2_test.sum().sum() - df_input_test.sum().sum())
    m_us=abs(df_test.sum().sum()- df_input_test.sum().sum())
    m_t1=abs(df_t_1_sub.sum().sum()- df_input_test.sum().sum())
    m_ave=abs(df_avg.sum().sum()- df_input_test.sum().sum())
    mse_ratio_pattern_2022.append(np.log((m_views+1)/(m_us+1)))
    
    coor = final_traj_2022[j_t]
    sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-36:-18]
    non_zero_indices = np.argwhere(sub_array!=0)
    coordinates = [(idx[0],idx[1],idx[2]) for idx in non_zero_indices]
    coordinates = np.array(coordinates)
    weights_1 = sub_array[sub_array != 0]
    emd_ratio_2022.append(calculate_EMD(df_test,df_input_test,df_preds_test_2_test,weights_1,coordinates,df_prio,year=2022,coor=coor))
    
    mae_pattern.append([m_us,m_views,m_t1,m_ave])
    emd_t1 = calculate_EMD_d(df_t_1_sub,df_input_test,weights_1,coordinates,df_prio,year=2022,coor=coor)
    emd_ave = calculate_EMD_d(df_avg,df_input_test,weights_1,coordinates,df_prio,year=2022,coor=coor)
    emd_us = calculate_EMD_d(df_test,df_input_test,weights_1,coordinates,df_prio,year=2022,coor=coor)
    emd_views = calculate_EMD_d(df_preds_test_2_test,df_input_test,weights_1,coordinates,df_prio,year=2022,coor=coor)
    emd_pattern.append([emd_us,emd_views,emd_t1,emd_ave])

df_metrics_2022 = pd.DataFrame([mse_ratio_pattern_2022,emd_ratio_2022]).T
df_metrics_2022.columns=['Ratio_Err','Ratio_EMD']

df_output_sub1 = pd.DataFrame(index=range(h),columns=df_prio.index)
gr_b_tot_sub1=[]
for enu,df_match in enumerate(dict_mat_sub1):
    if len(df_match[(df_match.iloc[:,6]<thres_emd) & (df_match.iloc[:,8]<thres_np)])>mini_mat:
        df_match=df_match[(df_match.iloc[:,6]<thres_emd) & (df_match.iloc[:,8]<thres_np)]
    else:
        df_match= df_match.iloc[:mini_mat,:]
    source = dict_inp_sub1[enu]
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
    df_g_mean[:,0] = df_g_mean[:,0]+final_traj_sub1[enu]['xmin']
    df_g_mean[:,1] = df_g_mean[:,1]+final_traj_sub1[enu]['ymin']
    df_g_mean=pd.DataFrame(df_g_mean)
    gr_b=[]
    for i in range(len(df_g_mean)):
        try:
            df_output_sub1.loc[int(df_g_mean.iloc[i,2]),df_prio[(df_prio.iloc[:,0]==int(df_g_mean.iloc[i,0])+88) & (df_prio.iloc[:,1]==int(df_g_mean.iloc[i,1])+311)].index]=df_g_mean.iloc[i,3]
            gr_b.append(df_prio[(df_prio.iloc[:,0]==int(df_g_mean.iloc[i,0])+88) & (df_prio.iloc[:,1]==int(df_g_mean.iloc[i,1])+311)].index[0])
        except:
            pass #if prio grid does not exist 
    gr_b_tot_sub1.append(pd.Series(gr_b).unique().tolist())
    
mse_ratio_pattern_sub1=[]
emd_ratio_sub1=[]
for j_t,list_pat in enumerate(gr_b_tot_sub1):
    df_test = df_output_sub1.loc[:,list_pat]
    df_test= df_test.fillna(0)
    df_input_test = df_input.iloc[-6:].loc[:,list_pat]
    df_preds_test_2_test = df_preds_views_half_1.iloc[:h].loc[:,list_pat]
    df_t_1_sub = df_input.iloc[-12:-6].loc[:,list_pat]
    df_t_1_sub.index = df_preds_test_2_test.index
    avg_row = df_input.iloc[-18:-6].loc[:, list_pat].mean()
    df_avg = pd.DataFrame([avg_row] * 6)
    df_avg.index = df_preds_test_2_test.index
    
    df_output_tot = pd.concat([df_output_tot,df_test.reset_index(drop=True)],axis=1)
    df_true_tot = pd.concat([df_true_tot,df_input_test.reset_index(drop=True)],axis=1)
    df_views_tot= pd.concat([df_views_tot,df_preds_test_2_test.reset_index(drop=True)],axis=1)
    df_t_1 = pd.concat([df_t_1,df_t_1_sub.reset_index(drop=True)],axis=1)
    df_ave = pd.concat([df_ave,df_avg.reset_index(drop=True)],axis=1)
    
    m_views=abs(df_preds_test_2_test.sum().sum() - df_input_test.sum().sum())
    m_us=abs(df_test.sum().sum()- df_input_test.sum().sum())
    m_t1=abs(df_t_1_sub.sum().sum()- df_input_test.sum().sum())
    m_ave=abs(df_avg.sum().sum()- df_input_test.sum().sum())
    mse_ratio_pattern_sub1.append(np.log((m_views+1)/(m_us+1)))
    
    coor = final_traj_sub1[j_t]
    sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-18:]
    non_zero_indices = np.argwhere(sub_array!=0)
    coordinates = [(idx[0],idx[1],idx[2]) for idx in non_zero_indices]
    coordinates = np.array(coordinates)
    weights_1 = sub_array[sub_array != 0]
    emd_ratio_sub1.append(calculate_EMD(df_test,df_input_test,df_preds_test_2_test,weights_1,coordinates,df_prio,year='2023_s',coor=coor))

    
    
    
    mae_pattern.append([m_us,m_views,m_t1,m_ave])
    emd_t1 = calculate_EMD_d(df_t_1_sub,df_input_test,weights_1,coordinates,df_prio,year='2023_s',coor=coor)
    emd_ave = calculate_EMD_d(df_avg,df_input_test,weights_1,coordinates,df_prio,year='2023_s',coor=coor)
    emd_us = calculate_EMD_d(df_test,df_input_test,weights_1,coordinates,df_prio,year='2023_s',coor=coor)
    emd_views = calculate_EMD_d(df_preds_test_2_test,df_input_test,weights_1,coordinates,df_prio,year='2023_s',coor=coor)
    emd_pattern.append([emd_us,emd_views,emd_t1,emd_ave])

    
df_metrics_sub1 = pd.DataFrame([mse_ratio_pattern_sub1,emd_ratio_sub1]).T
df_metrics_sub1.columns=['Ratio_Err','Ratio_EMD']

df_output_sub2 = pd.DataFrame(index=range(h),columns=df_prio.index)
gr_b_tot_sub2=[]
for enu,df_match in enumerate(dict_mat_sub2):
    if len(df_match[(df_match.iloc[:,6]<thres_emd) & (df_match.iloc[:,8]<thres_np)])>mini_mat:
        df_match=df_match[(df_match.iloc[:,6]<thres_emd) & (df_match.iloc[:,8]<thres_np)]
    else:
        df_match= df_match.iloc[:mini_mat,:]
    source = dict_inp_sub2[enu]
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
    df_g_mean[:,0] = df_g_mean[:,0]+final_traj_sub2[enu]['xmin']
    df_g_mean[:,1] = df_g_mean[:,1]+final_traj_sub2[enu]['ymin']
    df_g_mean=pd.DataFrame(df_g_mean)
    gr_b=[]
    for i in range(len(df_g_mean)):
        try:
            df_output_sub2.loc[int(df_g_mean.iloc[i,2]),df_prio[(df_prio.iloc[:,0]==int(df_g_mean.iloc[i,0])+88) & (df_prio.iloc[:,1]==int(df_g_mean.iloc[i,1])+311)].index]=df_g_mean.iloc[i,3]
            gr_b.append(df_prio[(df_prio.iloc[:,0]==int(df_g_mean.iloc[i,0])+88) & (df_prio.iloc[:,1]==int(df_g_mean.iloc[i,1])+311)].index[0])
        except:
            pass #if prio grid does not exist 
    gr_b_tot_sub2.append(pd.Series(gr_b).unique().tolist())
    
mse_ratio_pattern_sub2=[]
emd_ratio_sub2=[]
for j_t,list_pat in enumerate(gr_b_tot_sub2):
    df_test = df_output_sub2.loc[:,list_pat]
    df_test= df_test.fillna(0)
    df_input_test = df_input.iloc[-18:-12].loc[:,list_pat]
    df_preds_test_2_test = df_preds_views_half_2.iloc[:h].loc[:,list_pat]
    df_t_1_sub = df_input.iloc[-24:-18].loc[:,list_pat]
    df_t_1_sub.index= df_preds_test_2_test.index
    avg_row = df_input.iloc[-30:-18].loc[:, list_pat].mean()
    df_avg = pd.DataFrame([avg_row] * 6)
    df_avg.index = df_t_1_sub.index
    
    df_output_tot = pd.concat([df_output_tot,df_test.reset_index(drop=True)],axis=1)
    df_true_tot = pd.concat([df_true_tot,df_input_test.reset_index(drop=True)],axis=1)
    df_views_tot= pd.concat([df_views_tot,df_preds_test_2_test.reset_index(drop=True)],axis=1)
    df_t_1 = pd.concat([df_t_1,df_t_1_sub.reset_index(drop=True)],axis=1)
    df_ave = pd.concat([df_ave,df_avg.reset_index(drop=True)],axis=1)
    
    m_views=abs(df_preds_test_2_test.sum().sum() - df_input_test.sum().sum())
    m_us=abs(df_test.sum().sum()- df_input_test.sum().sum())
    m_t1=abs(df_t_1_sub.sum().sum()- df_input_test.sum().sum())
    m_ave=abs(df_avg.sum().sum()- df_input_test.sum().sum())
    mse_ratio_pattern_sub2.append(np.log((m_views+1)/(m_us+1)))

    coor = final_traj_sub2[j_t]
    sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-30:-12]
    non_zero_indices = np.argwhere(sub_array!=0)
    coordinates = [(idx[0],idx[1],idx[2]) for idx in non_zero_indices]
    coordinates = np.array(coordinates)
    weights_2 = sub_array[sub_array != 0]
    emd_ratio_sub2.append(calculate_EMD(df_test,df_input_test,df_preds_test_2_test,weights_2,coordinates,df_prio,year='2022_s',coor=coor))
    
    
    
    mae_pattern.append([m_us,m_views,m_t1,m_ave])
    emd_t1 = calculate_EMD_d(df_t_1_sub,df_input_test,weights_2,coordinates,df_prio,year='2022_s',coor=coor)
    emd_ave = calculate_EMD_d(df_avg,df_input_test,weights_2,coordinates,df_prio,year='2022_s',coor=coor)
    emd_us = calculate_EMD_d(df_test,df_input_test,weights_2,coordinates,df_prio,year='2022_s',coor=coor)
    emd_views = calculate_EMD_d(df_preds_test_2_test,df_input_test,weights_2,coordinates,df_prio,year='2022_s',coor=coor)
    emd_pattern.append([emd_us,emd_views,emd_t1,emd_ave])
    
    
df_metrics_sub2 = pd.DataFrame([mse_ratio_pattern_sub2,emd_ratio_sub2]).T
df_metrics_sub2.columns=['Ratio_Err','Ratio_EMD']
df_metric_all = pd.concat([df_metrics_2023,df_metrics_2022,df_metrics_sub1,df_metrics_sub2],axis=0)

mse_pg=[]
for i in range(len(df_output_tot.columns)):
    err_sf = mean_squared_error(df_true_tot.iloc[:,i], df_output_tot.iloc[:,i])
    err_views = mean_squared_error(df_true_tot.iloc[:,i], df_views_tot.iloc[:,i])
    mse_pg.append(np.log((err_views+1)/(err_sf+1)))

print(ttest_1samp(pd.Series(mse_pg),0))

# =============================================================================
# Figure F.1 
# =============================================================================

mean_all_mse, ci_all_mse = mean_and_ci(df_metric_all['Ratio_EMD'])
mean_all_err, ci_all_err = mean_and_ci(df_metric_all['Ratio_Err'])
plt.figure(figsize=(15,12))
sns.scatterplot(data=df_metric_all, x="Ratio_Err", y="Ratio_EMD", color='#B11B1B', marker='+', s=200)
plt.plot(mean_all_err, mean_all_mse, 'o', color='#B11B1B', markersize=16)
plt.plot([mean_all_err - ci_all_err, mean_all_err + ci_all_err],
    [mean_all_mse, mean_all_mse], color='#B11B1B', linewidth=4)
plt.plot([mean_all_err, mean_all_err],
    [mean_all_mse - ci_all_mse, mean_all_mse + ci_all_mse],
    color='#B11B1B', linewidth=4)
plt.axhline(0, linestyle='--', color='gray', alpha=0.7)
plt.axvline(0, linestyle='--', color='gray', alpha=0.7)
plt.xlabel("Fatalities Accuracy (Mean Average Error)", fontsize=30)
plt.ylabel("Spatio-temporal Accuracy (EMD)", fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xlim(-3,4)
plt.ylim(-0.5,1)
plt.savefig(fig_folder+'Figure_F1.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Figure 11 
# =============================================================================

ind_size=sum(len(sublist) for sublist in gr_b_tot_2023)
mse_pg_2023 = pd.Series(mse_pg).iloc[:ind_size]
ind_size_up=ind_size+sum(len(sublist) for sublist in gr_b_tot_2022)
mse_pg_2022 = pd.Series(mse_pg).iloc[ind_size:ind_size_up]
ind_size=ind_size_up
ind_size_up=ind_size+sum(len(sublist) for sublist in gr_b_tot_sub1)
mse_pg_sub1 = pd.Series(mse_pg).iloc[ind_size:ind_size_up]
ind_size=ind_size_up
mse_pg_sub2 = pd.Series(mse_pg).iloc[ind_size:]

ind_size=sum(len(sublist) for sublist in gr_b_tot_2023)
mse_pg_2023.index = df_output_tot.columns[:ind_size]
mse_pg_2023.drop_duplicates(inplace=True)
ind_size_up=ind_size+sum(len(sublist) for sublist in gr_b_tot_2022)
mse_pg_2022.index = df_output_tot.columns[ind_size:ind_size_up]
mse_pg_2022.drop_duplicates(inplace=True)
ind_size=ind_size_up
ind_size_up=ind_size+sum(len(sublist) for sublist in gr_b_tot_sub1)
mse_pg_sub1.index = df_output_tot.columns[ind_size:ind_size_up]
mse_pg_sub1.drop_duplicates(inplace=True)
ind_size=ind_size_up
mse_pg_sub2.index = df_output_tot.columns[ind_size:]
mse_pg_sub2.drop_duplicates(inplace=True)

df_prio_pred_obs = pd.concat([df_prio,mse_pg_2023,mse_pg_2022,mse_pg_sub1,mse_pg_sub2],axis=1)
df_prio_pred_obs.fillna(0,inplace=True)
df_prio_pred_obs['Mean'] = df_prio_pred_obs.iloc[:,-4:].mean(axis=1)
xcoord = df_prio_pred_obs.iloc[:, 2]
ycoord = df_prio_pred_obs.iloc[:, 3]

geometry = [Polygon([(x - 0.25, y - 0.25), (x + 0.25, y - 0.25), 
             (x + 0.25, y + 0.25), (x - 0.25, y + 0.25)]) for x, y in zip(xcoord, ycoord)]
gdf = gpd.GeoDataFrame(np.array(df_prio_pred_obs.iloc[:,9:]), geometry=geometry, columns=['Log_ratio'])

mean = pd.Series(mse_pg).mean()
sem = pd.Series(mse_pg).sem()
ci = 1.96 * sem  
data = pd.DataFrame({"Model": ["MSE"] * len(mse_pg), "Values": pd.Series(mse_pg)})
data['col_bin'] = 'red'
data.loc[data['Values']<0,'col_bin']='blue'

sns.set(style="ticks", rc={"figure.figsize": (12, 6)})

bins = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])  
counts, bin_edges = np.histogram(data['Values'], bins=bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
colors = ['#3B4CC0' if center < 0 else '#B40426' for center in bin_centers]

fig, axes = plt.subplots(figsize=(24,16))
vmin, vmax = -1, 1
gdf.plot(column='Log_ratio', cmap='coolwarm', norm=mcolors.Normalize(vmin=vmin, vmax=vmax), 
         ax=axes, edgecolor='lightgrey')
axes.set_axis_off()
ax_inset = fig.add_axes([0.25, 0.18, 0.15, 0.15])
ax_inset.bar(bin_centers, counts, width=np.diff(bins), color=colors, edgecolor='black', alpha=0.8)
ax_inset.axvline(x=0, color='black', linewidth=2)
ax_inset.set_xlabel("Log ratio MSE", fontsize=20)
ax_inset.set_ylabel("Frequency", fontsize=20)
ax_inset.tick_params(axis='both', which='both', labelsize=15)
sns.despine(ax=ax_inset, offset=5, trim=True)
ax_inset.tick_params(left=False)
ax_inset.text(-4,2000,f'{round((pd.Series(mse_pg)<0).sum()/len(pd.Series(mse_pg))*100)}%',color='blue',size=20)
ax_inset.text(3,2000,f'{round((pd.Series(mse_pg)>0).sum()/len(pd.Series(mse_pg))*100)}%',color='red',size=20)
ax_inset_2 = fig.add_axes([0.25, 0.38, 0.15, 0.12])
ax_inset_2.errorbar(y=0, x=mean, xerr=ci, fmt='o', color='black', capsize=10, capthick=0, 
             label='Mean ± 95% CI', markersize=15, linewidth=5, zorder=10)
ax_inset_2.axvline(x=0, color='black', linewidth=2,linestyle='--')
ax_inset_2.tick_params(axis='x', which='both', labelsize=15)
ax_inset_2.set_title('Mean ± 95% CI',fontsize=18)
sns.despine(ax=ax_inset_2, offset=5, trim=False)
ax_inset_2.set_xlim(-0.1,0.1)
ax_inset_2.yaxis.set_visible(False) 
ax_inset_2.spines['left'].set_visible(False)
ax_inset_2.text(0.02,-0.03,'P-val<0.01',color='black',size=16)
cax = fig.add_axes([0.55, 0.18, 0.15, 0.01])
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=[vmin,0, vmax])
cbar.set_label("Best Model", fontsize=20)
cbar.ax.set_xticklabels(["Views","Similar", "SF"], fontsize=15)
plt.savefig(fig_folder+'Figure_11.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Table 1 and Appendix E 
# =============================================================================

l_df_output_tot=[]
l_df_true_tot=[] 
l_df_views_tot=[]
l_df_t1_tot=[] 
l_df_ave_tot=[]

ind_size=sum(len(sublist) for sublist in gr_b_tot_2023)
l_df_output_tot.append(df_output_tot.iloc[:,:ind_size])
l_df_true_tot.append(df_true_tot.iloc[:,:ind_size])
l_df_views_tot.append(df_views_tot.iloc[:,:ind_size])
l_df_t1_tot.append(df_t_1.iloc[:,:ind_size])
l_df_ave_tot.append(df_ave.iloc[:,:ind_size])

ind_size_up=ind_size+sum(len(sublist) for sublist in gr_b_tot_2022)
l_df_output_tot.append(df_output_tot.iloc[:,ind_size:ind_size_up])
l_df_true_tot.append(df_true_tot.iloc[:,ind_size:ind_size_up])
l_df_views_tot.append(df_views_tot.iloc[:,ind_size:ind_size_up])
l_df_t1_tot.append(df_t_1.iloc[:,ind_size:ind_size_up])
l_df_ave_tot.append(df_ave.iloc[:,ind_size:ind_size_up])

ind_size=ind_size_up
ind_size_up=ind_size+sum(len(sublist) for sublist in gr_b_tot_sub1)
l_df_output_tot.append(df_output_tot.iloc[:,ind_size:ind_size_up])
l_df_true_tot.append(df_true_tot.iloc[:,ind_size:ind_size_up])
l_df_views_tot.append(df_views_tot.iloc[:,ind_size:ind_size_up])
l_df_t1_tot.append(df_t_1.iloc[:,ind_size:ind_size_up])
l_df_ave_tot.append(df_ave.iloc[:,ind_size:ind_size_up])

ind_size=ind_size_up
l_df_output_tot.append(df_output_tot.iloc[:,ind_size:])
l_df_true_tot.append(df_true_tot.iloc[:,ind_size:])
l_df_views_tot.append(df_views_tot.iloc[:,ind_size:])
l_df_t1_tot.append(df_t_1.iloc[:,ind_size:])
l_df_ave_tot.append(df_ave.iloc[:,ind_size:])

mae_results = []
mae_sf = []
mae_views = []
mae_t1 = []
mae_ave = []
period_names = ['Jan-2023', 'Jan-2022', 'July-2023', 'July-2022', 'All Periods']
for i in range(4):
    row_data = []
    row_sf = []
    row_views = []
    row_t1 = []
    row_ave = []
    for h in range(6):
        err_output = np.abs(l_df_output_tot[i].iloc[h, :] - l_df_true_tot[i].iloc[h, :])
        err_views = np.abs(l_df_views_tot[i].iloc[h, :] - l_df_true_tot[i].iloc[h, :])
        err_t1 = np.abs(l_df_t1_tot[i].iloc[h, :] - l_df_true_tot[i].iloc[h, :])
        err_ave = np.abs(l_df_ave_tot[i].iloc[h, :] - l_df_true_tot[i].iloc[h, :])
        log_ratio = np.log((err_views + 1) / (err_output + 1))
        mean_val = np.mean(log_ratio)
        se_val = np.std(log_ratio) / np.sqrt(len(log_ratio))
        row_data.append(f"{mean_val:.4f} ({se_val:.4f})")
        
        mean_sf = np.mean(np.log(err_output+1))
        se_sf = np.std(np.log(err_output+1)) / np.sqrt(len(np.log(err_output+1)))
        row_sf.append(f"{mean_sf:.4f} ({se_sf:.4f})")
        mean_views = np.mean(np.log(err_views+1))
        se_views = np.std(np.log(err_views+1)) / np.sqrt(len(np.log(err_views+1)))
        row_views.append(f"{mean_views:.4f} ({se_views:.4f})")
        
        mean_t1 = np.mean(np.log(err_t1+1))
        se_t1 = np.std(np.log(err_t1+1)) / np.sqrt(len(np.log(err_t1+1)))
        row_t1.append(f"{mean_t1:.4f} ({se_t1:.4f})")
        mean_ave = np.mean(np.log(err_ave+1))
        se_ave = np.std(np.log(err_ave+1)) / np.sqrt(len(np.log(err_ave+1)))
        row_ave.append(f"{mean_ave:.4f} ({se_ave:.4f})")
        
    err_output_all = np.abs(l_df_output_tot[i] - l_df_true_tot[i]).values.flatten()
    err_views_all = np.abs(l_df_views_tot[i] - l_df_true_tot[i]).values.flatten()
    err_t1_all = np.abs(l_df_t1_tot[i] - l_df_true_tot[i]).values.flatten()
    err_ave_all = np.abs(l_df_ave_tot[i] - l_df_true_tot[i]).values.flatten()
    log_ratio_all = np.log((err_views_all + 1) / (err_output_all + 1))
    mean_val = np.mean(log_ratio_all)
    se_val = np.std(log_ratio_all) / np.sqrt(len(log_ratio_all))
    row_data.append(f"{mean_val:.4f} ({se_val:.4f})")
    mean_sf = np.mean(np.log(err_output_all+1))
    se_sf = np.std(np.log(err_output_all+1)) / np.sqrt(len(np.log(err_output_all+1)))
    row_sf.append(f"{mean_sf:.4f} ({se_sf:.4f})")
    mean_views = np.mean(np.log(err_views_all+1))
    se_views = np.std(np.log(err_views_all+1)) / np.sqrt(len(np.log(err_views_all+1)))
    row_views.append(f"{mean_views:.4f} ({se_views:.4f})")
    mean_t1 = np.mean(np.log(err_t1_all+1))
    se_t1 = np.std(np.log(err_t1_all+1)) / np.sqrt(len(np.log(err_t1_all+1)))
    row_t1.append(f"{mean_t1:.4f} ({se_t1:.4f})")
    mean_ave = np.mean(np.log(err_ave_all+1))
    se_ave = np.std(np.log(err_ave_all+1)) / np.sqrt(len(np.log(err_ave_all+1)))
    row_ave.append(f"{mean_ave:.4f} ({se_ave:.4f})")
    
    mae_results.append(row_data)
    mae_sf.append(row_sf)
    mae_views.append(row_views)
    mae_t1.append(row_t1)
    mae_ave.append(row_ave)

row_data = []
row_sf = []
row_views = []
row_t1 = []
row_ave = []
for h in range(6):
    all_err_output = []
    all_err_views = []
    all_err_t1 = []
    all_err_ave = []
    for i in range(4):
        all_err_output.extend(np.abs(l_df_output_tot[i].iloc[h, :] - l_df_true_tot[i].iloc[h, :]).values)
        all_err_views.extend(np.abs(l_df_views_tot[i].iloc[h, :] - l_df_true_tot[i].iloc[h, :]).values)
        all_err_t1.extend(np.abs(l_df_t1_tot[i].iloc[h, :] - l_df_true_tot[i].iloc[h, :]).values)
        all_err_ave.extend(np.abs(l_df_ave_tot[i].iloc[h, :] - l_df_true_tot[i].iloc[h, :]).values)
    all_err_output = np.array(all_err_output)
    all_err_views = np.array(all_err_views)
    all_err_t1 = np.array(all_err_t1)
    all_err_ave = np.array(all_err_ave)
    log_ratio = np.log((all_err_views + 1) / (all_err_output + 1))
    mean_val = np.mean(log_ratio)
    se_val = np.std(log_ratio) / np.sqrt(len(log_ratio))
    row_data.append(f"{mean_val:.4f} ({se_val:.4f})")
    mean_sf = np.mean(np.log(all_err_output+1))
    se_sf = np.std(np.log(all_err_output+1)) / np.sqrt(len(np.log(all_err_output+1)))
    row_sf.append(f"{mean_sf:.4f} ({se_sf:.4f})")
    mean_views = np.mean(np.log(all_err_views+1))
    se_views = np.std(np.log(all_err_views+1)) / np.sqrt(len(np.log(all_err_views+1)))
    row_views.append(f"{mean_views:.4f} ({se_views:.4f})")
    
    mean_t1 = np.mean(np.log(all_err_t1+1))
    se_t1 = np.std(np.log(all_err_t1+1)) / np.sqrt(len(np.log(all_err_t1+1)))
    row_t1.append(f"{mean_t1:.4f} ({se_t1:.4f})")
    mean_ave = np.mean(np.log(all_err_ave+1))
    se_ave = np.std(np.log(all_err_ave+1)) / np.sqrt(len(np.log(all_err_ave+1)))
    row_ave.append(f"{mean_ave:.4f} ({se_ave:.4f})")

all_err_output = []
all_err_views = []
all_err_t1 = []
all_err_ave = []
for i in range(4):
    all_err_output.extend(np.abs(l_df_output_tot[i] - l_df_true_tot[i]).values.flatten())
    all_err_views.extend(np.abs(l_df_views_tot[i] - l_df_true_tot[i]).values.flatten())
    all_err_t1.extend(np.abs(l_df_t1_tot[i] - l_df_true_tot[i]).values.flatten())
    all_err_ave.extend(np.abs(l_df_ave_tot[i] - l_df_true_tot[i]).values.flatten())
all_err_output = np.array(all_err_output)
all_err_views = np.array(all_err_views)
all_err_t1 = np.array(all_err_t1)
all_err_ave = np.array(all_err_ave)
log_ratio = np.log((all_err_views + 1) / (all_err_output + 1))

mean_val = np.mean(log_ratio)
se_val = np.std(log_ratio) / np.sqrt(len(log_ratio))
row_data.append(f"{mean_val:.4f} ({se_val:.4f})")
mae_results.append(row_data)
mean_val = np.mean(np.log(all_err_output+1))
se_val = np.std(np.log(all_err_output+1)) / np.sqrt(len(np.log(all_err_output+1)))
row_sf.append(f"{mean_val:.4f} ({se_val:.4f})")
mae_sf.append(row_sf)
mean_val = np.mean(np.log(all_err_views+1))
se_val = np.std(np.log(all_err_views+1)) / np.sqrt(len(np.log(all_err_views+1)))
row_views.append(f"{mean_val:.4f} ({se_val:.4f})")
mae_views.append(row_views)

mean_t1 = np.mean(np.log(all_err_t1+1))
se_t1 = np.std(np.log(all_err_t1+1)) / np.sqrt(len(np.log(all_err_t1+1)))
row_t1.append(f"{mean_t1:.4f} ({se_t1:.4f})")
mean_ave = np.mean(np.log(all_err_ave+1))
se_ave = np.std(np.log(all_err_ave+1)) / np.sqrt(len(np.log(all_err_ave+1)))
row_ave.append(f"{mean_ave:.4f} ({se_ave:.4f})")
mae_t1.append(row_t1)
mae_ave.append(row_ave)

columns = [f'h={i+1}' for i in range(6)] + ['All Horizons']
df_mae = pd.DataFrame(mae_results, columns=columns, index=period_names)
print(df_mae.to_string())
df_mae_views = pd.DataFrame(mae_views, columns=columns, index=period_names)
print(df_mae_views.to_string())
df_mae_sf = pd.DataFrame(mae_sf, columns=columns, index=period_names)
print(df_mae_sf.to_string())
df_mae_t1 = pd.DataFrame(mae_t1, columns=columns, index=period_names)
print(df_mae_t1.to_string())
df_mae_ave = pd.DataFrame(mae_ave, columns=columns, index=period_names)
print(df_mae_ave.to_string())

mse_results = []
mse_sf = []
mse_views = []
mse_t1 = []
mse_ave = []
period_names = ['Jan-2023', 'Jan-2022', 'July-2023', 'July-2022', 'All Periods']
for i in range(4):
    row_data = []
    row_sf = []
    row_views = []
    row_t1 = []
    row_ave = []
    for h in range(6):
        err_output = (l_df_output_tot[i].iloc[h, :] - l_df_true_tot[i].iloc[h, :])**2
        err_views = (l_df_views_tot[i].iloc[h, :] - l_df_true_tot[i].iloc[h, :])**2
        err_t1 = (l_df_t1_tot[i].iloc[h, :] - l_df_true_tot[i].iloc[h, :])**2
        err_ave = (l_df_ave_tot[i].iloc[h, :] - l_df_true_tot[i].iloc[h, :])**2
        log_ratio = np.log((err_views + 1) / (err_output + 1))
        mean_val = np.mean(log_ratio)
        se_val = np.std(log_ratio) / np.sqrt(len(log_ratio))
        row_data.append(f"{mean_val:.4f} ({se_val:.4f})")
        
        mean_sf = np.mean(np.log(err_output+1))
        se_sf = np.std(np.log(err_output+1)) / np.sqrt(len(np.log(err_output+1)))
        row_sf.append(f"{mean_sf:.4f} ({se_sf:.4f})")
        mean_views = np.mean(np.log(err_views+1))
        se_views = np.std(np.log(err_views+1)) / np.sqrt(len(np.log(err_views+1)))
        row_views.append(f"{mean_views:.4f} ({se_views:.4f})")
        
        mean_t1 = np.mean(np.log(err_t1+1))
        se_t1 = np.std(np.log(err_t1+1)) / np.sqrt(len(np.log(err_t1+1)))
        row_t1.append(f"{mean_t1:.4f} ({se_t1:.4f})")
        mean_ave = np.mean(np.log(err_ave+1))
        se_ave = np.std(np.log(err_ave+1)) / np.sqrt(len(np.log(err_ave+1)))
        row_ave.append(f"{mean_ave:.4f} ({se_ave:.4f})")
        
    err_output_all = ((l_df_output_tot[i] - l_df_true_tot[i])**2).values.flatten()
    err_views_all = ((l_df_views_tot[i] - l_df_true_tot[i])**2).values.flatten()
    err_t1_all = ((l_df_t1_tot[i] - l_df_true_tot[i])**2).values.flatten()
    err_ave_all = ((l_df_ave_tot[i] - l_df_true_tot[i])**2).values.flatten()
    log_ratio_all = np.log((err_views_all + 1) / (err_output_all + 1))
    mean_val = np.mean(log_ratio_all)
    se_val = np.std(log_ratio_all) / np.sqrt(len(log_ratio_all))
    row_data.append(f"{mean_val:.4f} ({se_val:.4f})")
    mean_sf = np.mean(np.log(err_output_all+1))
    se_sf = np.std(np.log(err_output_all+1)) / np.sqrt(len(np.log(err_output_all+1)))
    row_sf.append(f"{mean_sf:.4f} ({se_sf:.4f})")
    mean_views = np.mean(np.log(err_views_all+1))
    se_views = np.std(np.log(err_views_all+1)) / np.sqrt(len(np.log(err_views_all+1)))
    row_views.append(f"{mean_views:.4f} ({se_views:.4f})")
    mean_t1 = np.mean(np.log(err_t1_all+1))
    se_t1 = np.std(np.log(err_t1_all+1)) / np.sqrt(len(np.log(err_t1_all+1)))
    row_t1.append(f"{mean_t1:.4f} ({se_t1:.4f})")
    mean_ave = np.mean(np.log(err_ave_all+1))
    se_ave = np.std(np.log(err_ave_all+1)) / np.sqrt(len(np.log(err_ave_all+1)))
    row_ave.append(f"{mean_ave:.4f} ({se_ave:.4f})")
    
    mse_results.append(row_data)
    mse_sf.append(row_sf)
    mse_views.append(row_views)
    mse_t1.append(row_t1)
    mse_ave.append(row_ave)

row_data = []
row_sf = []
row_views = []
row_t1 = []
row_ave = []
for h in range(6):
    all_err_output = []
    all_err_views = []
    all_err_t1 = []
    all_err_ave = []
    for i in range(4):
        all_err_output.extend(((l_df_output_tot[i].iloc[h, :] - l_df_true_tot[i].iloc[h, :])**2).values)
        all_err_views.extend(((l_df_views_tot[i].iloc[h, :] - l_df_true_tot[i].iloc[h, :])**2).values)
        all_err_t1.extend(((l_df_t1_tot[i].iloc[h, :] - l_df_true_tot[i].iloc[h, :])**2).values)
        all_err_ave.extend(((l_df_ave_tot[i].iloc[h, :] - l_df_true_tot[i].iloc[h, :])**2).values)
    all_err_output = np.array(all_err_output)
    all_err_views = np.array(all_err_views)
    all_err_t1 = np.array(all_err_t1)
    all_err_ave = np.array(all_err_ave)
    log_ratio = np.log((all_err_views + 1) / (all_err_output + 1))
    mean_val = np.mean(log_ratio)
    se_val = np.std(log_ratio) / np.sqrt(len(log_ratio))
    row_data.append(f"{mean_val:.4f} ({se_val:.4f})")
    mean_sf = np.mean(np.log(all_err_output+1))
    se_sf = np.std(np.log(all_err_output+1)) / np.sqrt(len(np.log(all_err_output+1)))
    row_sf.append(f"{mean_sf:.4f} ({se_sf:.4f})")
    mean_views = np.mean(np.log(all_err_views+1))
    se_views = np.std(np.log(all_err_views+1)) / np.sqrt(len(np.log(all_err_views+1)))
    row_views.append(f"{mean_views:.4f} ({se_views:.4f})")
    
    mean_t1 = np.mean(np.log(all_err_t1+1))
    se_t1 = np.std(np.log(all_err_t1+1)) / np.sqrt(len(np.log(all_err_t1+1)))
    row_t1.append(f"{mean_t1:.4f} ({se_t1:.4f})")
    mean_ave = np.mean(np.log(all_err_ave+1))
    se_ave = np.std(np.log(all_err_ave+1)) / np.sqrt(len(np.log(all_err_ave+1)))
    row_ave.append(f"{mean_ave:.4f} ({se_ave:.4f})")

all_err_output = []
all_err_views = []
all_err_t1 = []
all_err_ave = []
for i in range(4):
    all_err_output.extend(((l_df_output_tot[i] - l_df_true_tot[i])**2).values.flatten())
    all_err_views.extend(((l_df_views_tot[i] - l_df_true_tot[i])**2).values.flatten())
    all_err_t1.extend(((l_df_t1_tot[i] - l_df_true_tot[i])**2).values.flatten())
    all_err_ave.extend(((l_df_ave_tot[i] - l_df_true_tot[i])**2).values.flatten())
all_err_output = np.array(all_err_output)
all_err_views = np.array(all_err_views)
all_err_t1 = np.array(all_err_t1)
all_err_ave = np.array(all_err_ave)
log_ratio = np.log((all_err_views + 1) / (all_err_output + 1))

mean_val = np.mean(log_ratio)
se_val = np.std(log_ratio) / np.sqrt(len(log_ratio))
row_data.append(f"{mean_val:.4f} ({se_val:.4f})")
mse_results.append(row_data)
mean_val = np.mean(np.log(all_err_output+1))
se_val = np.std(np.log(all_err_output+1)) / np.sqrt(len(np.log(all_err_output+1)))
row_sf.append(f"{mean_val:.4f} ({se_val:.4f})")
mse_sf.append(row_sf)
mean_val = np.mean(np.log(all_err_views+1))
se_val = np.std(np.log(all_err_views+1)) / np.sqrt(len(np.log(all_err_views+1)))
row_views.append(f"{mean_val:.4f} ({se_val:.4f})")
mse_views.append(row_views)

mean_t1 = np.mean(np.log(all_err_t1+1))
se_t1 = np.std(np.log(all_err_t1+1)) / np.sqrt(len(np.log(all_err_t1+1)))
row_t1.append(f"{mean_t1:.4f} ({se_t1:.4f})")
mean_ave = np.mean(np.log(all_err_ave+1))
se_ave = np.std(np.log(all_err_ave+1)) / np.sqrt(len(np.log(all_err_ave+1)))
row_ave.append(f"{mean_ave:.4f} ({se_ave:.4f})")
mse_t1.append(row_t1)
mse_ave.append(row_ave)

columns = [f'h={i+1}' for i in range(6)] + ['All Horizons']
df_mse = pd.DataFrame(mse_results, columns=columns, index=period_names)
print(df_mse.to_string())
df_mse_views = pd.DataFrame(mse_views, columns=columns, index=period_names)
print(df_mse_views.to_string())
df_mse_sf = pd.DataFrame(mse_sf, columns=columns, index=period_names)
print(df_mse_sf.to_string())
df_mse_t1 = pd.DataFrame(mse_t1, columns=columns, index=period_names)
print(df_mse_t1.to_string())
df_mse_ave = pd.DataFrame(mse_ave, columns=columns, index=period_names)
print(df_mse_ave.to_string())

df_emd_pattern = pd.DataFrame(emd_pattern)
df_emd_pattern = np.log(df_emd_pattern+1)
mean_emd_pattern = df_emd_pattern.mean()
se_emd_pattern = df_emd_pattern.std() / np.sqrt(len(df_emd_pattern))

df_mae_pattern = pd.DataFrame(mae_pattern)
df_mae_pattern = np.log(df_mae_pattern+1)
mean_mae_pattern = df_mae_pattern.mean()
se_mae_pattern = df_mae_pattern.std() / np.sqrt(len(df_mae_pattern))

df_mse_pattern = pd.DataFrame(mae_pattern)**2
df_mse_pattern = np.log(df_mse_pattern+1)
mean_mse_pattern = df_mse_pattern.mean()
se_mse_pattern = df_mse_pattern.std() / np.sqrt(len(df_mse_pattern))

all_err_output = []
all_err_views = []
all_err_t1 = []
all_err_ave = []
for i in range(4):
    all_err_output.extend(np.abs(l_df_output_tot[i] - l_df_true_tot[i]).values.flatten())
    all_err_views.extend(np.abs(l_df_views_tot[i] - l_df_true_tot[i]).values.flatten())
    all_err_t1.extend(np.abs(l_df_t1_tot[i] - l_df_true_tot[i]).values.flatten())
    all_err_ave.extend(np.abs(l_df_ave_tot[i] - l_df_true_tot[i]).values.flatten())
all_err_output = np.log(np.array(all_err_output)+1)
all_err_views = np.log(np.array(all_err_views)+1)
all_err_t1 = np.log(np.array(all_err_t1)+1)
all_err_ave = np.log(np.array(all_err_ave)+1)
mean_mae_prio = pd.Series([np.mean(all_err_output),np.mean(all_err_views),np.mean(all_err_t1),np.mean(all_err_ave)])
se_mae_prio = pd.Series([np.std(all_err_output) / np.sqrt(len(all_err_output)),
                         np.std(all_err_views) / np.sqrt(len(all_err_views)),
                         np.std(all_err_t1) / np.sqrt(len(all_err_t1)),
                         np.std(all_err_ave) / np.sqrt(len(all_err_ave))])

all_err_output = []
all_err_views = []
all_err_t1 = []
all_err_ave = []
for i in range(4):
    all_err_output.extend(((l_df_output_tot[i] - l_df_true_tot[i])**2).values.flatten())
    all_err_views.extend(((l_df_views_tot[i] - l_df_true_tot[i])**2).values.flatten())
    all_err_t1.extend(((l_df_t1_tot[i] - l_df_true_tot[i])**2).values.flatten())
    all_err_ave.extend(((l_df_ave_tot[i] - l_df_true_tot[i])**2).values.flatten())
all_err_output = np.log(np.array(all_err_output)+1)
all_err_views = np.log(np.array(all_err_views)+1)
all_err_t1 = np.log(np.array(all_err_t1)+1)
all_err_ave = np.log(np.array(all_err_ave)+1)
mean_mse_prio = pd.Series([np.mean(all_err_output),np.mean(all_err_views),np.mean(all_err_t1),np.mean(all_err_ave)])
se_mse_prio = pd.Series([np.std(all_err_output) / np.sqrt(len(all_err_output)),
                         np.std(all_err_views) / np.sqrt(len(all_err_views)),
                         np.std(all_err_t1) / np.sqrt(len(all_err_t1)),
                         np.std(all_err_ave) / np.sqrt(len(all_err_ave))])

model_names = ['ShapeFinder','Views','Last Six Months','Average last year']
df_table_1 = pd.DataFrame(index=model_names)
df_table_1['Zone EMD'] = [f"{mean_emd_pattern[k]:.4f} ({se_emd_pattern[k]:.4f})" for k in range(4)]
df_table_1['Zone MAE'] = [f"{mean_mae_pattern[k]:.4f} ({se_mae_pattern[k]:.4f})" for k in range(4)]
df_table_1['Zone MSE'] = [f"{mean_mse_pattern[k]:.4f} ({se_mse_pattern[k]:.4f})" for k in range(4)]
df_table_1['Prio MAE'] = [f"{mean_mae_prio[k]:.4f} ({se_mae_prio[k]:.4f})" for k in range(4)]
df_table_1['Prio MSE'] = [f"{mean_mse_prio[k]:.4f} ({se_mse_prio[k]:.4f})" for k in range(4)]
print(df_table_1.to_string())

# 3D forecast for July-December 2023 for the Appendix I
l_df_output_tot[2].to_csv(data_folder+'Results/3d_sf.csv')

# =============================================================================
# Figure 4 
# =============================================================================

df_output_2023_no_na = df_output_2023.fillna(0)
gr_tot_2022=[j for gr_b_tot_2022_sub in gr_b_tot_2022 for j in gr_b_tot_2022_sub]
gr_tot_2023=[j for gr_b_tot_2023_sub in gr_b_tot_2023 for j in gr_b_tot_2023_sub]

df_output_2023_no_na = df_output_2023.fillna(0)
s_true = df_input.iloc[-24:-12,:].sum()
df_prio_pred_obs = pd.concat([df_prio,s_true],axis=1)
xcoord = df_prio_pred_obs.iloc[:, 2]
ycoord = df_prio_pred_obs.iloc[:, 3]
color_values = df_prio_pred_obs.iloc[:, 5].tolist()
geometry = [Polygon([(x - 0.25, y - 0.25), (x + 0.25, y - 0.25), 
             (x + 0.25, y + 0.25), (x - 0.25, y + 0.25)]) for x, y in zip(xcoord, ycoord)]
gdf = gpd.GeoDataFrame(np.array(df_prio_pred_obs.iloc[:,5]), geometry=geometry, columns=['Obs'])
gdf.index = df_prio.index

boundaries=[]
for j in range(len(gr_b_tot_2023)):
    gdf_sub = gdf.loc[gr_b_tot_2023[j]]
    merged_geometry = gdf_sub.unary_union
    if isinstance(merged_geometry, MultiPolygon):
        boundaries.append([geom.boundary for geom in merged_geometry.geoms])
    else:
        boundaries.append([merged_geometry.boundary])

vmin = 1
vmax = df_prio_pred_obs.iloc[:,5].max()
fig, axes = plt.subplots(1, 1, figsize=(24, 24))
gdf.plot(column='Obs', cmap='Reds',norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), ax=axes, edgecolor='grey')
axes.set_xlim(xcoord.min() - 1, xcoord.max() + 1)
axes.set_ylim(ycoord.min() - 1, ycoord.max() + 1)
axes.set_aspect('equal')
axes.set_axis_off()
plt.tight_layout()
plt.savefig(fig_folder+'Figure_4a.png', dpi=300, bbox_inches='tight')
plt.show()

vmin = 0.5
vmax = 1
fig, axes = plt.subplots(1, 1, figsize=(24, 24))
gdf.plot(column='Obs', cmap='Reds',norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), ax=axes, edgecolor='grey')
axes.set_xlim(xcoord.min() - 1, xcoord.max() + 1)
axes.set_ylim(ycoord.min() - 1, ycoord.max() + 1)
axes.set_aspect('equal')
axes.set_axis_off()
plt.tight_layout()
plt.savefig(fig_folder+'Figure_4b.png', dpi=300, bbox_inches='tight')
plt.show()

polygons = []
for rect in all_final_traj:
    y_min = df_prio[df_prio['row']==int(rect['xmin']+88)]['ycoord'].iloc[0]
    y_max = df_prio[df_prio['row']==int(rect['xmax']+88)]['ycoord'].iloc[0]
    x_min = df_prio[df_prio['col']==int(rect['ymin']+311)]['xcoord'].iloc[0]
    x_max = df_prio[df_prio['col']==int(rect['ymax']+311)]['xcoord'].iloc[0]
    polygons.append(Polygon([
        (x_min-0.25, y_min-0.25),
        (x_max+0.25, y_min-0.25),
        (x_max+0.25, y_max+0.25),
        (x_min-0.25, y_max+0.25)
    ]))

gdf_rect = gpd.GeoDataFrame(geometry=polygons)

vmin = 0.5
vmax = 1
fig, axes = plt.subplots(1, 1, figsize=(24, 24))
gdf_rect.boundary.plot(ax=axes, facecolor='none', edgecolor='black', linewidth=3)
gdf.plot(column='Obs', cmap='Reds',norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), ax=axes, edgecolor='grey',alpha=0.4)
axes.set_xlim(xcoord.min() - 1, xcoord.max() + 1)
axes.set_ylim(ycoord.min() - 1, ycoord.max() + 1)
axes.set_aspect('equal')
axes.set_axis_off()
plt.tight_layout()
plt.savefig(fig_folder+'Figure_4c.png', dpi=300, bbox_inches='tight')
plt.show()

vmin = 1
vmax = df_prio_pred_obs.iloc[:,5].max()
fig, axes = plt.subplots(1, 1, figsize=(24, 24))
gdf.plot(column='Obs', cmap='Reds',norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), ax=axes, edgecolor='grey')
for boundarie in boundaries:
    for boundary in boundarie:
        axes.plot(*boundary.xy, color='black', linewidth=3)
axes.set_xlim(xcoord.min() - 1, xcoord.max() + 1)
axes.set_ylim(ycoord.min() - 1, ycoord.max() + 1)
axes.set_aspect('equal')
axes.set_axis_off()
plt.tight_layout()
plt.savefig(fig_folder+'Figure_4d.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Figure 13
# =============================================================================

ratio_over=[]
ratio_input=[]
ratio_us_over=[]
max_val_sf=[]
max_val_views=[]
max_val_real=[]
tot_input_magn=[]
std_pred_v=[]
std_pred_sf=[]
size_pattern=[]
for j_t,list_pat in enumerate(gr_b_tot_2023):
    df_test = df_output_2023.loc[:,list_pat]
    size_pattern.append(len(df_test.columns))
    df_test= df_test.fillna(0)
    df_input_test = df_input.iloc[-12:-12 + h].loc[:,list_pat] if h < 12 else df_input.iloc[-12:].loc[:,list_pat]
    df_preds_test_2_test = df_preds_views.iloc[:h].loc[:,list_pat]
    ratio_over.append(((df_preds_test_2_test.sum().sum()+1)-(df_input_test.sum().sum()+1))/(df_input_test.sum().sum()+1))
    ratio_input.append(df_input_test.sum().sum()/(df_input.iloc[-24:-12].loc[:,list_pat].sum().sum()/2))
    ratio_us_over.append(((df_test.sum().sum()+1)-(df_input_test.sum().sum()+1))/(df_input_test.sum().sum()+1))
    max_val_sf.append(df_test.max().max())
    max_val_views.append(df_preds_test_2_test.max().max())
    max_val_real.append(df_input_test.max().max())
    std_pred_v.append(df_preds_test_2_test.values.std())
    std_pred_sf.append(df_test.values.std())
    tot_input_magn.append(df_input.iloc[-24:-12].loc[:,list_pat].sum().sum())
    
for j_t,list_pat in enumerate(gr_b_tot_2022):
    df_test = df_output_2022.loc[:,list_pat]
    size_pattern.append(len(df_test.columns))
    df_test= df_test.fillna(0)
    df_input_test = df_input.iloc[-24:-24 + h].loc[:,list_pat]
    df_preds_test_2_test = df_preds_views_first.iloc[:h].loc[:,list_pat]
    ratio_over.append(((df_preds_test_2_test.sum().sum()+1)-(df_input_test.sum().sum()+1))/(df_input_test.sum().sum()+1))
    ratio_input.append(df_input_test.sum().sum()/(df_input.iloc[-36:-24].loc[:,list_pat].sum().sum()/2))
    ratio_us_over.append(((df_test.sum().sum()+1)-(df_input_test.sum().sum()+1))/(df_input_test.sum().sum()+1))
    max_val_sf.append(df_test.max().max())
    max_val_views.append(df_preds_test_2_test.max().max())
    max_val_real.append(df_input_test.max().max())
    std_pred_v.append(df_preds_test_2_test.values.std())
    std_pred_sf.append(df_test.values.std())
    tot_input_magn.append(df_input.iloc[-36:-24].loc[:,list_pat].sum().sum())
    
    
for j_t,list_pat in enumerate(gr_b_tot_sub1):
    df_test = df_output_sub1.loc[:,list_pat]
    size_pattern.append(len(df_test.columns))
    df_test= df_test.fillna(0)
    df_input_test = df_input.iloc[-6:].loc[:,list_pat]
    df_preds_test_2_test = df_preds_views_half_1.iloc[:h].loc[:,list_pat]
    ratio_over.append(((df_preds_test_2_test.sum().sum()+1)-(df_input_test.sum().sum()+1))/(df_input_test.sum().sum()+1))
    ratio_input.append(df_input_test.sum().sum()/(df_input.iloc[-18:-6].loc[:,list_pat].sum().sum()/2))
    ratio_us_over.append(((df_test.sum().sum()+1)-(df_input_test.sum().sum()+1))/(df_input_test.sum().sum()+1))
    max_val_sf.append(df_test.max().max())
    max_val_views.append(df_preds_test_2_test.max().max())
    max_val_real.append(df_input_test.max().max())
    std_pred_v.append(df_preds_test_2_test.values.std())
    std_pred_sf.append(df_test.values.std())
    tot_input_magn.append(df_input.iloc[-18:-6].loc[:,list_pat].sum().sum())
    
for j_t,list_pat in enumerate(gr_b_tot_sub2):
    df_test = df_output_sub2.loc[:,list_pat]
    size_pattern.append(len(df_test.columns))
    df_test= df_test.fillna(0)
    df_input_test = df_input.iloc[-18:-12].loc[:,list_pat]
    df_preds_test_2_test = df_preds_views_half_2.iloc[:h].loc[:,list_pat]
    ratio_over.append(((df_preds_test_2_test.sum().sum()+1)-(df_input_test.sum().sum()+1))/(df_input_test.sum().sum()+1))
    ratio_input.append(df_input_test.sum().sum()/(df_input.iloc[-30:-18].loc[:,list_pat].sum().sum()/2))
    ratio_us_over.append(((df_test.sum().sum()+1)-(df_input_test.sum().sum()+1))/(df_input_test.sum().sum()+1))
    max_val_sf.append(df_test.max().max())
    max_val_views.append(df_preds_test_2_test.max().max())
    max_val_real.append(df_input_test.max().max())
    std_pred_v.append(df_preds_test_2_test.values.std())
    std_pred_sf.append(df_test.values.std())
    tot_input_magn.append(df_input.iloc[-30:-18].loc[:,list_pat].sum().sum())
    

df_metric_all['Ratio_Overpred'] = ratio_over
df_metric_all['Ratio_Input'] = ratio_input
df_metric_all['Ratio_Overpred_SF'] = ratio_us_over

df_metric_all['max_sf']= max_val_sf
df_metric_all['max_obs']= max_val_real
df_metric_all['max_views']= max_val_views

df_metric_all['SD_Views'] = std_pred_v
df_metric_all['SD_SF'] = std_pred_sf

df_metric_all['Magnitude']=tot_input_magn
df_metric_all['Dim']=size_pattern

def signed_log_transform(x):
    return np.sign(x) * np.log1p(np.abs(x))
x_ticks_input = [0, 0.5, 1, 2, 5, 10]
x_ticks_magnitude = [1, 10, 100, 1000,10000]
x_ticks_dim = [2,10,30,100,300]

fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)
axes[0].scatter(np.log(df_metric_all['Ratio_Input'] + 1), 
                signed_log_transform(df_metric_all['Ratio_Overpred_SF']),
                color='dimgrey', s=100)
axes[0].axhline(0, linestyle='--', color='black', linewidth=2)
axes[0].set_xlabel(r'$R_{inc}$', fontsize=35)
axes[0].set_ylabel('SPE - ShapeFinder', fontsize=35)
axes[0].tick_params(axis='both', labelsize=26)
axes[0].set_xticks(np.log1p(x_ticks_input))
axes[0].set_xticklabels(x_ticks_input)
axes[1].scatter(np.log(df_metric_all['Magnitude'] + 1), 
                signed_log_transform(df_metric_all['Ratio_Overpred_SF']),
                color='dimgrey', s=100)
axes[1].axhline(0, linestyle='--', color='black', linewidth=2)
axes[1].set_xlabel('Magnitude', fontsize=35)
axes[1].tick_params(axis='both', labelsize=26)
axes[1].set_xticks(np.log1p(x_ticks_magnitude))
axes[1].set_xticklabels(x_ticks_magnitude)
axes[2].scatter(np.log(df_metric_all['Dim'] + 1), 
                signed_log_transform(df_metric_all['Ratio_Overpred_SF']),
                color='dimgrey', s=100)
axes[2].axhline(0, linestyle='--', color='black', linewidth=2)
axes[2].set_xlabel('Dimension', fontsize=35)
axes[2].tick_params(axis='both', labelsize=26)
axes[2].set_xticks(np.log1p(x_ticks_dim))
axes[2].set_xticklabels(x_ticks_dim)
plt.tight_layout()
plt.savefig(fig_folder+'Figure_13.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Figure 12 
# =============================================================================

vmin, vmax = -3, 3
fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(
    np.log1p(df_metric_all['Ratio_Input']),
    signed_log_transform(df_metric_all['Ratio_Overpred']),
    c=df_metric_all['Ratio_Err'],
    cmap='coolwarm',
    s=100,
    vmin=vmin,
    vmax=vmax,
    edgecolors='lightgrey'
)
ax.axhline(0, linestyle='--', color='grey', linewidth=2)
ax.set_xlabel('$R_{inc}$', fontsize=24)
ax.set_ylabel('SPE - Views', fontsize=24)
ax.set_yticks(ax.get_yticks())
ax.set_yticklabels([f'{tick:.0f}' for tick in ax.get_yticks()], fontsize=20)
ax.set_xticks([0, 0.405, 0.693, 1.098, 1.791, 2.39789])
ax.set_xticklabels(['0', '0.5', '1', '2', '5', '10'], fontsize=20)
cax = fig.add_axes([0.55, 0.8, 0.3, 0.01])
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=[vmin, 0, vmax])
cbar.set_label("Best Model", fontsize=15)
cbar.ax.set_xticklabels(["Views", "Similar", "SF"], fontsize=12)
plt.savefig(fig_folder+'Figure_12.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Figure B.1 
# =============================================================================

df_metric_all_sub = df_metric_all[(df_metric_all['Ratio_Overpred_SF']<5) & (df_metric_all['Ratio_Overpred']<5)]
plt.figure(figsize=(10, 6))
b = sns.boxplot(data=df_metric_all[['Ratio_Overpred_SF', 'Ratio_Overpred']],
                orient="h",  
                width=0.4,
                color="white",
                linewidth=2,
                showfliers=False)
sns.stripplot(data=df_metric_all_sub[['Ratio_Overpred_SF', 'Ratio_Overpred']],
              orient="h", 
              color="darkgrey",
              linewidth=1,
              alpha=0.4)
b.set_xlabel("Signed Percentage Error", fontsize=20)
b.set_ylabel("Model", fontsize=20)
b.set_xlim(-2, 6)
b.set_yticklabels(['SF', 'VIEWS'])
b.tick_params(axis='both', which='both', labelsize=20)
b.axvline(x=0, linestyle='--', color='black', linewidth=1)
sns.despine(offset=5, trim=True)
plt.annotate("Underprediction", 
             xy=(-0.1, -0.4),  # Start point of the arrow (x, y)
             xytext=(-1.5, -0.4),  # End point for the text
             ha='center', va='center', fontsize=15,
             arrowprops=dict(arrowstyle='<-'))
plt.annotate("Overprediction", 
             xy=(0.1, -0.4),  # Start point of the arrow (x, y)
             xytext=(1.5, -0.4),  # End point for the text
             ha='center', va='center', fontsize=15,
             arrowprops=dict(arrowstyle='<-'))
plt.savefig(fig_folder+'Figure_B1.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Figure B.2 
# =============================================================================

df_metric_all['Ratio_Max_SF'] = df_metric_all['max_sf']-df_metric_all['max_obs']
df_metric_all['Ratio_Max_Views'] = df_metric_all['max_views']-df_metric_all['max_obs']
df_metric_all_sub_2 = df_metric_all[(df_metric_all['Ratio_Max_SF']<75) & (df_metric_all['Ratio_Max_Views']<75) & (df_metric_all['Ratio_Max_SF']>-125) & (df_metric_all['Ratio_Max_Views']>-125)]
plt.figure(figsize=(10, 6))
b = sns.boxplot(data=df_metric_all[['Ratio_Max_SF', 'Ratio_Max_Views']],
                orient="h",  
                width=0.4,
                color="white",
                linewidth=2,
                showfliers=False)
b=sns.stripplot(data=df_metric_all_sub_2[['Ratio_Max_SF', 'Ratio_Max_Views']],
              orient="h", 
              color="darkgrey",
              linewidth=1,
              alpha=0.4)
b.set_xlabel("Error on Maximum Value", fontsize=20)
b.set_ylabel("Model", fontsize=20)
b.set_yticklabels(['SF', 'VIEWS'])
b.axvline(x=0, linestyle='--', color='black', linewidth=1)
sns.despine(offset=5, trim=True)
plt.annotate("Underprediction", 
             xy=(-5, -0.4),  # Start point of the arrow (x, y)
             xytext=(-45, -0.4),  # End point for the text
             ha='center', va='center', fontsize=15,
             arrowprops=dict(arrowstyle='<-'))
plt.annotate("Overprediction", 
             xy=(5, -0.4),  # Start point of the arrow (x, y)
             xytext=(45, -0.4),  # End point for the text
             ha='center', va='center', fontsize=15,
             arrowprops=dict(arrowstyle='<-'))
plt.savefig(fig_folder+'Figure_B2.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Figure H.3 
# =============================================================================

h=6
ratio_zero=[]
list_pat = [item for sublist in gr_b_tot_2023 for item in sublist]
df_test = df_output_2023.loc[:, ~df_output_2023.columns.isin(list_pat)]
df_test= df_test.fillna(0)
df_input_test = df_input.iloc[-12:-12 + h].loc[:,~df_output_2023.columns.isin(list_pat)] 
df_preds_test_2_test = df_preds_views.iloc[:h].loc[:,~df_output_2023.columns.isin(list_pat)]
m_v = mean_squared_error(df_input_test, df_preds_test_2_test,multioutput='raw_values')
m_sf=mean_squared_error(df_input_test, df_test,multioutput='raw_values')
ratio_zero= ratio_zero+np.log((m_v+1)/(m_sf+1)).tolist()
ratio_zero_nz=[]
df_preds_test_2_test=df_preds_test_2_test.loc[:,df_input_test.sum()>0]
df_test=df_test.loc[:,df_input_test.sum()>0]
df_input_test=df_input_test.loc[:,df_input_test.sum()>0]
m_v = mean_squared_error(df_input_test, df_preds_test_2_test,multioutput='raw_values')
m_sf=mean_squared_error(df_input_test, df_test,multioutput='raw_values')
ratio_zero_nz= ratio_zero_nz+np.log((m_v+1)/(m_sf+1)).tolist()

list_pat = [item for sublist in gr_b_tot_2022 for item in sublist]
df_test = df_output_2022.loc[:,~df_output_2022.columns.isin(list_pat)]
df_test= df_test.fillna(0)
df_input_test = df_input.iloc[-24:-24 + h].loc[:,~df_output_2022.columns.isin(list_pat)]
df_preds_test_2_test = df_preds_views_first.iloc[:h].loc[:,~df_output_2022.columns.isin(list_pat)]
m_v = mean_squared_error(df_input_test, df_preds_test_2_test,multioutput='raw_values')
m_sf=mean_squared_error(df_input_test, df_test,multioutput='raw_values')
ratio_zero= ratio_zero+np.log((m_v+1)/(m_sf+1)).tolist()
df_preds_test_2_test=df_preds_test_2_test.loc[:,df_input_test.sum()>0]
df_test=df_test.loc[:,df_input_test.sum()>0]
df_input_test=df_input_test.loc[:,df_input_test.sum()>0]
m_v = mean_squared_error(df_input_test, df_preds_test_2_test,multioutput='raw_values')
m_sf=mean_squared_error(df_input_test, df_test,multioutput='raw_values')
ratio_zero_nz= ratio_zero_nz+np.log((m_v+1)/(m_sf+1)).tolist()

    
list_pat = [item for sublist in gr_b_tot_sub1 for item in sublist]
df_test = df_output_sub1.loc[:,~df_output_sub1.columns.isin(list_pat)]
df_test= df_test.fillna(0)
df_input_test = df_input.iloc[-6:].loc[:,~df_output_sub1.columns.isin(list_pat)]
df_preds_test_2_test = df_preds_views_half_1.iloc[:h].loc[:,~df_output_sub1.columns.isin(list_pat)]
m_v = mean_squared_error(df_input_test, df_preds_test_2_test,multioutput='raw_values')
m_sf=mean_squared_error(df_input_test, df_test,multioutput='raw_values')
ratio_zero= ratio_zero+np.log((m_v+1)/(m_sf+1)).tolist()
df_preds_test_2_test=df_preds_test_2_test.loc[:,df_input_test.sum()>0]
df_test=df_test.loc[:,df_input_test.sum()>0]
df_input_test=df_input_test.loc[:,df_input_test.sum()>0]
m_v = mean_squared_error(df_input_test, df_preds_test_2_test,multioutput='raw_values')
m_sf=mean_squared_error(df_input_test, df_test,multioutput='raw_values')
ratio_zero_nz= ratio_zero_nz+np.log((m_v+1)/(m_sf+1)).tolist()

    
list_pat= [item for sublist in gr_b_tot_sub2 for item in sublist]
df_test = df_output_sub2.loc[:,~df_output_sub2.columns.isin(list_pat)]
df_test= df_test.fillna(0)
df_input_test = df_input.iloc[-18:-12].loc[:,~df_output_sub2.columns.isin(list_pat)]
df_preds_test_2_test = df_preds_views_half_2.iloc[:h].loc[:,~df_output_sub2.columns.isin(list_pat)]
m_v = mean_squared_error(df_input_test, df_preds_test_2_test,multioutput='raw_values')
m_sf=mean_squared_error(df_input_test, df_test,multioutput='raw_values')
ratio_zero= ratio_zero+np.log((m_v+1)/(m_sf+1)).tolist()
df_preds_test_2_test=df_preds_test_2_test.loc[:,df_input_test.sum()>0]
df_test=df_test.loc[:,df_input_test.sum()>0]
df_input_test=df_input_test.loc[:,df_input_test.sum()>0]
m_v = mean_squared_error(df_input_test, df_preds_test_2_test,multioutput='raw_values')
m_sf=mean_squared_error(df_input_test, df_test,multioutput='raw_values')
ratio_zero_nz= ratio_zero_nz+np.log((m_v+1)/(m_sf+1)).tolist()

mean_all_zero, ci_all_zero = mean_and_ci(pd.Series(ratio_zero))
print(ttest_1samp(pd.Series(ratio_zero), 0))

c_z=0
c_nz=0
for i in range(len(df_input.columns)):
    if (df_input.iloc[-36:-24,i]==0).all():
        if (df_input.iloc[-24:-18,i]==0).all():
            c_z=c_z+1
        else:
            c_nz = c_nz+1
    if (df_input.iloc[-30:-18,i]==0).all():
        if (df_input.iloc[-18:-12,i]==0).all():
            c_z=c_z+1
        else:
            c_nz = c_nz+1
    if (df_input.iloc[-24:-12,i]==0).all():
        if (df_input.iloc[-12:-6,i]==0).all():
            c_z=c_z+1
        else:
            c_nz = c_nz+1
    if (df_input.iloc[-18:-6,i]==0).all():
        if (df_input.iloc[-6:,i]==0).all():
            c_z=c_z+1
        else:
            c_nz = c_nz+1
print(c_z/(c_z+c_nz))           

l_me_clu=[]
l_sd_clu=[]
for i in [mse_pg,ratio_zero,ratio_zero+mse_pg]:
    mean_cov_clu, ci_cov_clu = mean_and_ci(np.array(i))
    l_me_clu.append(mean_cov_clu)
    l_sd_clu.append(ci_cov_clu)

df_comp_clu = pd.DataFrame([l_me_clu,l_sd_clu,['Act.Zones','Non Act.Zones','All Cells']]).T
df_comp_clu.sort_values([0],inplace=True)

plt.figure(figsize=(10,8))
x_positions = np.arange(len(df_comp_clu))
plt.errorbar(x_positions, df_comp_clu[0], yerr=df_comp_clu[1], fmt='o',markersize=12,linewidth=3,color='black')
plt.axhline(0,linestyle='--',color='dimgrey')
plt.xticks(x_positions, df_comp_clu[2],fontsize=25,rotation=45,ha='right')
plt.yticks(fontsize=25)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylabel('Log Ratio of MSE',fontsize=30)
plt.xlabel('Parameters Configuration',fontsize=30)
plt.tight_layout()
plt.savefig(fig_folder+'Figure_H3.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Figure H.2 
# =============================================================================

n_onset_in = 0
n_onset_sf=0
n_onset_views=0
n_keep_zero=0
for i in range(len(df_true_tot.columns)):
    if (df_ave.iloc[:,i].sum()==0)&(df_true_tot.iloc[:,i].sum()>0):
        n_onset_in+=1
        if df_output_tot.iloc[:,i].sum()>1:
            n_onset_sf+=1
        if df_views_tot.iloc[:,i].sum()>1:
            n_onset_views+=1
    elif (df_ave.iloc[:,i].sum()==0)&(df_true_tot.iloc[:,i].sum()==0):
        n_keep_zero+=1
        
n_onset_out=0
n_ouset_out_views=0
val_onset=[]

list_pat = [item for sublist in gr_b_tot_2023 for item in sublist]
df_input_test = df_input.iloc[-12:-6].loc[:,~df_output_2023.columns.isin(list_pat)] 
df_preds_test_2_test = df_preds_views.iloc[:h].loc[:,~df_output_2023.columns.isin(list_pat)]
n_onset_out += (df_input_test.sum()>0).sum()
n_ouset_out_views += ((df_input_test.sum()>0)&(df_preds_test_2_test.sum()>1)).sum()
vale_1 = df_preds_test_2_test.sum()

list_pat = [item for sublist in gr_b_tot_2022 for item in sublist]
df_input_test = df_input.iloc[-24:-18].loc[:,~df_output_2022.columns.isin(list_pat)] 
df_preds_test_2_test = df_preds_views_first.iloc[:h].loc[:,~df_output_2022.columns.isin(list_pat)]
n_onset_out += (df_input_test.sum()>0).sum()
n_ouset_out_views += ((df_input_test.sum()>0)&(df_preds_test_2_test.sum()>1)).sum()
vale_2 = df_preds_test_2_test.sum()

list_pat = [item for sublist in gr_b_tot_sub1 for item in sublist]
df_input_test = df_input.iloc[-6:].loc[:,~df_output_sub1.columns.isin(list_pat)]
df_preds_test_2_test = df_preds_views_half_1.iloc[:h].loc[:,~df_output_sub1.columns.isin(list_pat)]
n_onset_out += (df_input_test.sum()>0).sum()
n_ouset_out_views += ((df_input_test.sum()>0)&(df_preds_test_2_test.sum()>1)).sum()
vale_3 = df_preds_test_2_test.sum()

list_pat= [item for sublist in gr_b_tot_sub2 for item in sublist]
df_input_test = df_input.iloc[-18:-12].loc[:,~df_output_sub2.columns.isin(list_pat)]
df_preds_test_2_test = df_preds_views_half_2.iloc[:h].loc[:,~df_output_sub2.columns.isin(list_pat)]
n_onset_out += (df_input_test.sum()>0).sum()
n_ouset_out_views += ((df_input_test.sum()>0)&(df_preds_test_2_test.sum()>1)).sum()
vale_4 = df_preds_test_2_test.sum()

fig, ax = plt.subplots(figsize=(7, 5))
categories = ["In Active Zones", "Out of Active Zones"]
x = np.arange(len(categories))
bar_width = 0.5
ax.bar(x, [n_onset_in, n_onset_out],
       width=bar_width, color="#B4B2A9", zorder=2)
ax.bar(x, [n_onset_views, n_ouset_out_views],
       width=bar_width, color="#2C2C2A", zorder=3)
ax.bar([x[0]], [n_onset_sf],
       width=bar_width, color="darkred", alpha=0.55,
       hatch="////", edgecolor="darkred", linewidth=1.5, zorder=4)

for i, val in enumerate([n_onset_in, n_onset_out]):
    ax.text(x[i], val + 4, str(val),
            ha="center", va="bottom", fontsize=12, color="#5F5E5A")
for i, val in enumerate([n_onset_views, n_ouset_out_views]):
    ax.text(x[i], val + 4, str(val),
            ha="center", va="bottom", fontsize=12, color="black")
for i, val in enumerate([n_onset_sf]):
    ax.text(x[i], val + 4, str(val),
            ha="center", va="bottom", fontsize=12, color="red")
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=13)
ax.set_ylabel("Number of onsets", fontsize=13)
ax.set_ylim(0, max(n_onset_in, n_onset_out) * 1.15)
ax.spines[["top", "right"]].set_visible(False)
ax.spines["left"].set_color("#B4B2A9")
ax.spines["bottom"].set_color("#B4B2A9")
ax.yaxis.grid(True, color="#D3D1C7", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
handles = [
    Patch(color="#B4B2A9", label="Total onsets"),
    Patch(color="#2C2C2A", label="Caught by VIEWS"),
    Patch(facecolor="darkred", alpha=0.55, hatch="////",
          edgecolor="darkred", label="Caught by SF")]
ax.legend(handles=handles, frameon=False, fontsize=12, loc="upper right")
plt.tight_layout()
plt.savefig(fig_folder+'Figure_H2.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Appendix H 
# =============================================================================

bx,by,bz,nz=3,3,12,6 # block size and look-ahead
orig_nan=np.isnan(array_3d) # true where out-of-area (never touched by zero→nan)
size_x,size_y,size_z=array_3d_na.shape
full_zero,also_zero,has_activity=0,0,0
for xi in range(0,size_x-bx+1,bx):
 for yi in range(0,size_y-by+1,by):
  for zi in range(0,size_z-bz+1,bz):
   b=array_3d_na[xi:xi+bx,yi:yi+by,zi:zi+bz] # current block
   if orig_nan[xi:xi+bx,yi:yi+by,zi:zi+bz].any():continue # skip out-of-area
   if np.isnan(b).all():full_zero+=1 # all zeros, no NaN → full-zero block
   else:has_activity+=1 # at least one >0, no out-of-area NaN → active block
   if not np.isnan(b).all():continue # look-ahead only for full-zero blocks
   if zi+bz+nz>size_z:continue # not enough z ahead
   nb=array_3d_na[xi:xi+bx,yi:yi+by,zi+bz:zi+bz+nz] # next 6 z-slices
   if np.nansum(nb)==0:also_zero+=1 # nansum treats NaN-as-zero and real NaN the same
print(f"Full-zero sections: {full_zero:,} | Active sections (>0, no NaN): {has_activity:,}")
print(f"% also zero next 6 months: {also_zero/full_zero*100:.2f}%" if full_zero else "No full-zero sections found")

# =============================================================================
# Figure F.2 - Examples of zone forecasts (2022)
# =============================================================================

for ind_pl in [16,19,23,25]:
    plot_map=True
    coor = final_traj_2022[ind_pl]
    sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-36:-18]
    non_zero_indices = np.argwhere(sub_array!=0)
    coordinates = [(idx[0],idx[1],idx[2]) for idx in non_zero_indices]
    coordinates = np.array(coordinates)
    weights_1 = sub_array[sub_array != 0]
    
    list_pat = gr_b_tot_2022[ind_pl]
    df_test = df_output_2022.loc[:,list_pat]
    df_test= df_test.fillna(0)
    df_input_test = df_input.iloc[-24:-24 + h].loc[:,list_pat]
    df_preds_test_2_test = df_preds_views_first.iloc[:h].loc[:,list_pat]
    df_views_coor = df_preds_test_2_test.reset_index().melt(id_vars='month_id', var_name='prio', value_name='Value')
    df_views_coor = pd.merge(df_views_coor, df_prio, left_on='prio', right_index=True, how='inner')
    df_views_coor= np.array(df_views_coor[['row', 'col','month_id','Value']])
    df_views_coor[:,0] = df_views_coor[:,0]-coor['xmin']-88
    df_views_coor[:,1] = df_views_coor[:,1]-coor['ymin']-311
    df_views_coor[:,2] = df_views_coor[:,2]-505+12
    df_views_coor= df_views_coor[df_views_coor[:,3]>0]
    
    df_test_coor = df_test.reset_index().melt(id_vars='index', var_name='prio', value_name='Value')
    df_test_coor = pd.merge(df_test_coor, df_prio, left_on='prio', right_index=True, how='inner')
    df_test_coor= np.array(df_test_coor[['row', 'col','index','Value']])
    df_test_coor[:,0] = df_test_coor[:,0]-coor['xmin']-88
    df_test_coor[:,1] = df_test_coor[:,1]-coor['ymin']-311
    df_test_coor[:,2] = df_test_coor[:,2]+12
    df_test_coor= df_test_coor[df_test_coor[:,3]>0]
    weights_train = weights_1[coordinates[:,2]<12]
    weights_test = weights_1[coordinates[:,2]>=12]
    coordinates_train = coordinates[coordinates[:,2]<12]
    coordinates_test = coordinates[coordinates[:,2]>=12]
    
    df_output_2022_no_na = df_output_2022.fillna(0)
    s_sf = df_output_2022_no_na.sum()
    s_vi = df_preds_views_first.iloc[:h].sum()
    s_true = df_input.iloc[-24:-24 + h].sum()
    df_prio_pred_obs = pd.concat([df_prio,s_sf,s_vi,s_true],axis=1)
    xcoord = df_prio_pred_obs.iloc[:, 2]
    ycoord = df_prio_pred_obs.iloc[:, 3]
    color_values = df_prio_pred_obs.iloc[:, 6].tolist()
    geometry = [Polygon([(x - 0.25, y - 0.25), (x + 0.25, y - 0.25), 
                 (x + 0.25, y + 0.25), (x - 0.25, y + 0.25)]) for x, y in zip(xcoord, ycoord)]
    gdf = gpd.GeoDataFrame(np.array(df_prio_pred_obs.iloc[:,5:]), geometry=geometry, columns=['SF','Views','Obs'])
    borders = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    col_int = gr_b_tot_2022[ind_pl]
    xcoor_min = df_prio.loc[col_int]['xcoord'].min()-7
    xcoor_max = df_prio.loc[col_int]['xcoord'].max()+7
    ycoor_min = df_prio.loc[col_int]['ycoord'].min()-7
    ycoor_max = df_prio.loc[col_int]['ycoord'].max()+7
    
    gdf.index = df_prio.index
    gdf_sub=gdf.loc[col_int]
    
    vmin = 1
    if len(weights_test)==0:
        vmax = 10
    else:
        vmax = weights_test.max()
    
    fig, axes = plt.subplots(1, 3, figsize=(72, 24))
    borders.plot(ax=axes[0], color='#F5F5F5',edgecolor='black')
    borders.plot(ax=axes[1], color='#F5F5F5',edgecolor='black')
    borders.plot(ax=axes[2], color='#F5F5F5',edgecolor='black')
    gdf.plot(column='SF', cmap='Reds',norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), ax=axes[0], edgecolor='grey',alpha=0.1)
    gdf_sub.plot(column='SF', cmap='Reds',norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), ax=axes[0], edgecolor='black',linewidth=3)
    axes[0].set_xlim(xcoor_min - 1, xcoor_max + 1)
    axes[0].set_ylim(ycoor_min - 1, ycoor_max + 1)
    axes[0].set_aspect('equal')
    axes[0].set_axis_off()
    gdf.plot(column='Views', cmap='Reds',norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), ax=axes[1], edgecolor='grey',alpha=0.1)
    gdf_sub.plot(column='Views', cmap='Reds',norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), ax=axes[1], edgecolor='black',linewidth=3)
    axes[1].set_xlim(xcoor_min - 1, xcoor_max + 1)
    axes[1].set_ylim(ycoor_min - 1, ycoor_max + 1)
    axes[1].set_aspect('equal')
    axes[1].set_axis_off()
    gdf.plot(column='Obs', cmap='Reds',norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), ax=axes[2], edgecolor='grey',alpha=0.1)
    gdf_sub.plot(column='Obs', cmap='Reds',norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), ax=axes[2], edgecolor='black',linewidth=3)
    axes[2].set_xlim(xcoor_min - 1, xcoor_max + 1)
    axes[2].set_ylim(ycoor_min - 1, ycoor_max + 1)
    axes[2].set_aspect('equal')
    axes[2].set_axis_off()
    plt.tight_layout()
    plt.savefig(fig_folder+f"Figure_F2{'abcd'[[16,19,23,25].index(ind_pl)]}.png", dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# Figure A.1 - Best and worst PRIO-Grid MSE examples
# =============================================================================

ind_size=[sum(len(sublist) for sublist in gr_b_tot_2023)]
ind_size.append(ind_size[-1]+sum(len(sublist) for sublist in gr_b_tot_2022))
ind_size.append(ind_size[-1]+sum(len(sublist) for sublist in gr_b_tot_sub1))

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(25, 10))
axes = axes.flatten()  
plot_count=0
for i in range(len(df_output_tot.columns)):
    err_sf = mean_squared_error(df_true_tot.iloc[:,i], df_output_tot.iloc[:,i])
    err_views = mean_squared_error(df_true_tot.iloc[:,i], df_views_tot.iloc[:,i])
    if (np.log((err_views+1)/(err_sf+1))>0.5) & (df_views_tot.iloc[:,i].sum()<df_output_tot.iloc[:,i].sum()):
        long,lat = df_prio.loc[df_output_tot.columns[i],['xcoord','ycoord']]
        ax = axes[plot_count]
        ax.plot(df_true_tot.iloc[:, i], label='Obs', color='black', marker='o')
        ax.plot(df_views_tot.iloc[:, i], label='Views', color='lightgrey', marker='o')
        ax.plot(df_output_tot.iloc[:, i], label='SF', color='#B11B1B', marker='o')
        ax.set_title(f'Lon: {long - 0.25} Lat: {lat - 0.25}',fontsize=20)
        ax.axis('off')
        plot_count += 1
plt.tight_layout()
plt.savefig(fig_folder+'Figure_A1_top.png', dpi=300, bbox_inches='tight')
plt.show()
        
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(25, 10))
axes = axes.flatten()  
plot_count=0
for i in range(len(df_output_tot.columns)):
    err_sf = mean_squared_error(df_true_tot.iloc[:,i], df_output_tot.iloc[:,i])
    err_views = mean_squared_error(df_true_tot.iloc[:,i], df_views_tot.iloc[:,i])
    if (np.log((err_views+1)/(err_sf+1))<-0.5) & (df_views_tot.iloc[:,i].sum()>df_output_tot.iloc[:,i].sum()):
        long,lat = df_prio.loc[df_output_tot.columns[i],['xcoord','ycoord']]
        ax = axes[plot_count]
        ax.plot(df_true_tot.iloc[:, i], label='Obs', color='black', marker='o')
        ax.plot(df_views_tot.iloc[:, i], label='Views', color='lightgrey', marker='o')
        ax.plot(df_output_tot.iloc[:, i], label='SF', color='#B11B1B', marker='o')
        ax.set_title(f'Lon: {long - 0.25} Lat: {lat - 0.25}',fontsize=20)
        ax.axis('off')
        plot_count += 1
        if plot_count == 8:
            break
plt.tight_layout()
plt.savefig(fig_folder+'Figure_A1_bottom.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Figure 2 - Kenya-Somalia input sequence and four similar cases (3D)
# =============================================================================

n_mat=2

sub_array = dict_inp[n_mat]
non_zero_indices = np.argwhere(sub_array!=0)
coordinates = [(idx[0],idx[1],idx[2]) for idx in non_zero_indices]
coordinates = np.array(coordinates)
value = sub_array[sub_array != 0]

dist_matrix = distance.cdist(coordinates, coordinates)
mst = minimum_spanning_tree(dist_matrix)
edges = np.transpose(mst.nonzero())
x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
flat_indices = value
cmap = cm.get_cmap('Reds')
norm = mcolors.LogNorm(vmin=1, vmax=value.max())
marker_colors = cmap(norm(value))
fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=10, color=marker_colors))])
for edge in edges:
    fig.add_trace(go.Scatter3d(
        x=[coordinates[edge[0], 0], coordinates[edge[1], 0]],
        y=[coordinates[edge[0], 1], coordinates[edge[1], 1]],
        z=[coordinates[edge[0], 2], coordinates[edge[1], 2]],
        mode='lines',
        line=dict(color='grey', width=2)))
fig.update_layout(scene=dict(
                    xaxis_title='Longitude',
                    yaxis_title='Latitude',
                    zaxis_title='Time'),
                    title='Cone - Mid SD',
                    showlegend=False)
fig.update_layout(scene=dict(
xaxis=dict(visible=False),
yaxis=dict(visible=False),
zaxis=dict(visible=False)
))  
plot(fig, filename=fig_folder+'Figure_2_input.html', auto_open=False)

for coor_s in range(4):
    coor = dict_mat[n_mat].sort_values([6]).iloc[coor_s,:6]
    sub_array = array_3d_zero[int(coor[0]):int(coor[1])+1, int(coor[2]):int(coor[3])+1,int(coor[4]):int(coor[5])+1]
    non_zero_indices = np.argwhere(sub_array!=0)
    coordinates = [(idx[0],idx[1],idx[2]) for idx in non_zero_indices]
    coordinates = np.array(coordinates)
    value = sub_array[sub_array != 0]
    
    dist_matrix = distance.cdist(coordinates, coordinates)
    mst = minimum_spanning_tree(dist_matrix)
    edges = np.transpose(mst.nonzero())
    x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
    flat_indices = value#np.ravel_multi_index((x, y, z), array_3d_zero.shape)
    cmap = cm.get_cmap('Reds')
    norm = mcolors.LogNorm(vmin=1, vmax=value.max())
    marker_colors = cmap(norm(value))
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=10, color=marker_colors))])
    for edge in edges:
        fig.add_trace(go.Scatter3d(
            x=[coordinates[edge[0], 0], coordinates[edge[1], 0]],
            y=[coordinates[edge[0], 1], coordinates[edge[1], 1]],
            z=[coordinates[edge[0], 2], coordinates[edge[1], 2]],
            mode='lines',
            line=dict(color='grey', width=2)))
    fig.update_layout(scene=dict(
                        xaxis_title='Longitude',
                        yaxis_title='Latitude',
                        zaxis_title='Time'),
                        title='Cone - Mid SD',
                        showlegend=False)
    fig.update_layout(scene=dict(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    zaxis=dict(visible=False)
    ))  
    plot(fig, filename=fig_folder+f'Figure_2_similar_{coor_s+1}.html', auto_open=False)

def darken_cmap(cmap_name, factor=0.9):
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, 256))
    black = np.array([0, 0, 0])
    darkened_colors = colors[:, :3] * (1 - factor) + black * factor
    return LinearSegmentedColormap.from_list(f"{cmap_name}_dark", darkened_colors)

def create_3d_mesh_with_variable_spheres_and_cylinders(coordinates, flat_indices, cylinder_radius=0.04):
    coordinates = np.array(coordinates)    
    dist_matrix = distance.cdist(coordinates, coordinates)
    mst = minimum_spanning_tree(dist_matrix)
    edges = np.transpose(mst.nonzero())
    cmap = darken_cmap('Reds', factor=0.5)
    norm_flat_indices = (flat_indices+1) / (flat_indices.max() +1)
    colors = (cmap(norm_flat_indices)[:, :3] * 255).astype(np.uint8)  
    all_vertices = []
    all_faces = []
    all_colors = []
    for i, (coord, color) in enumerate(zip(coordinates, colors)):
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.4)
        sphere.apply_translation(coord)
        all_vertices.append(sphere.vertices)
        all_faces.append(sphere.faces + len(all_vertices[-1]) * i)
        all_colors.append(np.tile(color, (len(sphere.vertices), 1)))
    vertex_offset = sum(len(v) for v in all_vertices)
    for edge in edges:
        i, j = edge
        start = coordinates[i]
        end = coordinates[j]
        cylinder = trimesh.creation.cylinder(radius=cylinder_radius, segment=[start, end])
        all_vertices.append(cylinder.vertices)
        all_faces.append(cylinder.faces + vertex_offset)
        all_colors.append(np.tile([80, 80, 80], (len(cylinder.vertices), 1))) 
        vertex_offset += len(cylinder.vertices)

    combined_vertices = np.vstack(all_vertices)
    combined_faces = np.vstack(all_faces)
    combined_colors = np.vstack(all_colors)
    full_mesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces, vertex_colors=combined_colors)
    return full_mesh

def save_mesh_as_gltf(mesh, filename):
    mesh.export(filename, file_type='glb')
    print(f"Mesh saved as {filename}")

n_mat=2
sub_array = dict_inp[n_mat]
non_zero_indices = np.argwhere(sub_array!=0)
coordinates = [(idx[0],idx[1],idx[2]) for idx in non_zero_indices]
coordinates = np.array(coordinates)
value = sub_array[sub_array != 0]
mesh_input = create_3d_mesh_with_variable_spheres_and_cylinders(coordinates, np.log(value+1))
save_mesh_as_gltf(mesh_input, fig_folder+'Figure_2_input.glb')

for coor_s in range(4):
    coor = dict_mat[n_mat].sort_values([6]).iloc[coor_s,:6]
    sub_array = array_3d_zero[int(coor[0]):int(coor[1])+1, int(coor[2]):int(coor[3])+1,int(coor[4]):int(coor[5])+1]
    non_zero_indices = np.argwhere(sub_array!=0)
    coordinates = [(idx[0],idx[1],idx[2]) for idx in non_zero_indices]
    coordinates = np.array(coordinates)
    value = sub_array[sub_array != 0]
    mesh_input = create_3d_mesh_with_variable_spheres_and_cylinders(coordinates, np.log(value+1))
    save_mesh_as_gltf(mesh_input, fig_folder+f'Figure_2_similar_{coor_s+1}.glb')

# =============================================================================
# Figure 2 - Maps of the input and similar cases
# =============================================================================

s_true = df_input.iloc[-24:-12].sum()
df_prio_pred_obs = pd.concat([df_prio,s_true],axis=1)
xcoord = df_prio_pred_obs.iloc[:, 2]
ycoord = df_prio_pred_obs.iloc[:, 3]
color_values = df_prio_pred_obs.iloc[:, 5].tolist()
geometry = [Polygon([(x - 0.25, y - 0.25), (x + 0.25, y - 0.25), 
             (x + 0.25, y + 0.25), (x - 0.25, y + 0.25)]) for x, y in zip(xcoord, ycoord)]
gdf = gpd.GeoDataFrame(np.array(df_prio_pred_obs.iloc[:,5]), geometry=geometry, columns=['Obs'])
borders = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

col_int = gr_b_tot_2023[n_mat]
xcoor_min = df_prio.loc[col_int]['xcoord'].min()-3
xcoor_max = df_prio.loc[col_int]['xcoord'].max()+3
ycoor_min = df_prio.loc[col_int]['ycoord'].min()-3
ycoor_max = df_prio.loc[col_int]['ycoord'].max()+3
gdf.index = df_prio.index
gdf_sub=gdf.loc[col_int]
coor = final_traj[n_mat]
sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-24:-12]
value = sub_array[sub_array != 0]
vmin = 1
vmax = value.max()
fig, axes = plt.subplots(1, 1, figsize=(24, 24))
borders.plot(ax=axes, color='#D3D3D3',edgecolor='#000000',linewidth=4)
gdf.plot(column='Obs', cmap='Reds',norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), ax=axes, edgecolor='grey',alpha=0.1)
gdf_sub.plot(column='Obs', cmap='Reds',norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), ax=axes, edgecolor='black',linewidth=4)
axes.set_xlim(xcoor_min - 5, xcoor_max + 5)
axes.set_ylim(ycoor_min - 5, ycoor_max + 5)
axes.set_axis_off()
plt.tight_layout()
plt.savefig(fig_folder+'Figure_2_map_input.png', dpi=300, bbox_inches='tight')
plt.show()

for coor_s in range(5):
    coor = dict_mat[n_mat].sort_values([6]).iloc[coor_s,:6]
    s_true = df_input.iloc[int(coor[4]):int(coor[5])+1].sum()
    df_prio_pred_obs = pd.concat([df_prio,s_true],axis=1)
    color_values = df_prio_pred_obs.iloc[:, 5].tolist()
    gdf = gpd.GeoDataFrame(np.array(df_prio_pred_obs.iloc[:,5]),geometry=geometry,columns=['Obs'])
    gdf.index = df_prio.index
    sub_array = array_3d_zero[int(coor[0]):int(coor[1])+1, int(coor[2]):int(coor[3])+1,int(coor[4]):int(coor[5])+1]
    non_zero_indices = np.argwhere(sub_array!=0)
    coordinates = [(idx[0],idx[1],idx[2]) for idx in non_zero_indices]
    coordinates = np.array(coordinates)
    value = sub_array[sub_array != 0]
    
    ycoor_min = df_prio[df_prio.iloc[:,0]==coor[0]+88].iloc[0,3]
    ycoor_max = df_prio[df_prio.iloc[:,0]==coor[1]+88].iloc[0,3]
    xcoor_min = df_prio[df_prio.iloc[:,1]==coor[2]+311].iloc[0,2]
    xcoor_max = df_prio[df_prio.iloc[:,1]==coor[3]+311].iloc[0,2]
    
    col_int = df_prio[(df_prio.iloc[:,3]>ycoor_min) & (df_prio.iloc[:,3]<ycoor_max) & (df_prio.iloc[:,2]>xcoor_min) & (df_prio.iloc[:,2]<xcoor_max)].index
    gdf_sub=gdf.loc[col_int]
    
    vmin = 1
    vmax = value.max()
    fig, axes = plt.subplots(1, 1, figsize=(24, 24))
    borders.plot(ax=axes, color='#D3D3D3',edgecolor='#000000',linewidth=4)
    gdf.plot(column='Obs', cmap='Reds',norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), ax=axes, edgecolor='grey',alpha=0.1)
    gdf_sub.plot(column='Obs', cmap='Reds',norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), ax=axes, edgecolor='black',linewidth=4)
    axes.set_xlim(xcoor_min - 5, xcoor_max + 5)
    axes.set_ylim(ycoor_min - 5, ycoor_max + 5)
    axes.set_axis_off()
    plt.tight_layout()
    plt.savefig(fig_folder+f"Figure_2_map_similar_{coor_s+1}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(pd.date_range('01/01/1990',periods=len(df_input),freq='M')[int(coor[4])])
    
n_mat=2
coor = final_traj[n_mat]
sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-24:-12]
value = sub_array[sub_array != 0]
sub_array_after = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-12:-6]
value_after = sub_array_after[sub_array_after != 0]
print(f'Input : {value.sum()/value_after.sum()}')
print(f'Input_count: {len(value)/len(value_after)}')

for coor_s in range(5):
    coor = dict_mat[n_mat].sort_values([6]).iloc[coor_s,:6]
    sub_array = array_3d_zero[int(coor[0]):int(coor[1])+1, int(coor[2]):int(coor[3])+1,int(coor[4]):int(coor[5])+1]
    value = sub_array[sub_array != 0]
    sub_array_after = array_3d_zero[int(coor[0]):int(coor[1])+1, int(coor[2]):int(coor[3])+1,int(coor[5])+1:int(coor[5])+7]
    value_after = sub_array_after[sub_array_after != 0]
    print(' ')
    print(f'Match {coor_s+1} : {value.sum()/value_after.sum()}')
    print(f'Match_count {coor_s+1} : {len(value)/len(value_after)}')

# =============================================================================
# Figure 9 
# =============================================================================

n_mat=2
coor = final_traj[n_mat]
sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-24:-12]
non_zero_indices = np.argwhere(sub_array!=0)
coordinates = [(idx[0],idx[1],idx[2]) for idx in non_zero_indices]
coordinates = np.array(coordinates)
value = sub_array[sub_array != 0]
dist_matrix = distance.cdist(coordinates, coordinates)
mst = minimum_spanning_tree(dist_matrix)
edges = np.transpose(mst.nonzero())
x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
flat_indices = value
cmap = cm.get_cmap('Reds')
norm = mcolors.LogNorm(vmin=1, vmax=value.max())
marker_colors = cmap(norm(value))
fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=10, color=marker_colors))])
for edge in edges:
    fig.add_trace(go.Scatter3d(
        x=[coordinates[edge[0], 0], coordinates[edge[1], 0]],
        y=[coordinates[edge[0], 1], coordinates[edge[1], 1]],
        z=[coordinates[edge[0], 2], coordinates[edge[1], 2]],
        mode='lines',
        line=dict(color='grey', width=2)))
fig.update_layout(scene=dict(
                    xaxis_title='Longitude',
                    yaxis_title='Latitude',
                    zaxis_title='Time'),
                    title='Cone - Mid SD',
                    showlegend=False)
fig.update_layout(scene=dict(
xaxis=dict(visible=False),
yaxis=dict(visible=False),
zaxis=dict(visible=False)
))  
plot(fig, filename=fig_folder+'Figure_9_input.html', auto_open=False)

coor = final_traj[n_mat]
sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-12:-6]
non_zero_indices = np.argwhere(sub_array!=0)
coordinates = [(idx[0],idx[1],idx[2]) for idx in non_zero_indices]
coordinates = np.array(coordinates)
value = sub_array[sub_array != 0]
dist_matrix = distance.cdist(coordinates, coordinates)
mst = minimum_spanning_tree(dist_matrix)
edges = np.transpose(mst.nonzero())
x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
flat_indices = value
cmap = cm.get_cmap('Reds')
norm = mcolors.LogNorm(vmin=1, vmax=value.max())
marker_colors = cmap(norm(value))
fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=10, color=marker_colors))])
for edge in edges:
    fig.add_trace(go.Scatter3d(
        x=[coordinates[edge[0], 0], coordinates[edge[1], 0]],
        y=[coordinates[edge[0], 1], coordinates[edge[1], 1]],
        z=[coordinates[edge[0], 2], coordinates[edge[1], 2]],
        mode='lines',
        line=dict(color='grey', width=2)))
fig.update_layout(scene=dict(
                    xaxis_title='Longitude',
                    yaxis_title='Latitude',
                    zaxis_title='Time'),
                    title='Cone - Mid SD',
                    showlegend=False)
fig.update_layout(scene=dict(
xaxis=dict(visible=False),
yaxis=dict(visible=False),
zaxis=dict(visible=False)
))  
plot(fig, filename=fig_folder+'Figure_9_observed.html', auto_open=False)

for coor_s in range(8):
    coor = dict_mat[n_mat].sort_values([6]).iloc[coor_s,:6]
    sub_array = array_3d_zero[int(coor[0]):int(coor[1])+1, int(coor[2]):int(coor[3])+1,int(coor[5])+1:int(coor[5])+7]
    non_zero_indices = np.argwhere(sub_array!=0)
    coordinates = [(idx[0],idx[1],idx[2]) for idx in non_zero_indices]
    coordinates = np.array(coordinates)
    value = sub_array[sub_array != 0]
    dist_matrix = distance.cdist(coordinates, coordinates)
    mst = minimum_spanning_tree(dist_matrix)
    edges = np.transpose(mst.nonzero())
    x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
    flat_indices = value
    cmap = cm.get_cmap('Reds')
    norm = mcolors.LogNorm(vmin=1, vmax=value.max())
    marker_colors = cmap(norm(value))
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=10, color=marker_colors))])
    for edge in edges:
        fig.add_trace(go.Scatter3d(
            x=[coordinates[edge[0], 0], coordinates[edge[1], 0]],
            y=[coordinates[edge[0], 1], coordinates[edge[1], 1]],
            z=[coordinates[edge[0], 2], coordinates[edge[1], 2]],
            mode='lines',
            line=dict(color='grey', width=2)))
    fig.update_layout(scene=dict(
                        xaxis_title='Longitude',
                        yaxis_title='Latitude',
                        zaxis_title='Time'),
                        title='Cone - Mid SD',
                        showlegend=False)
    fig.update_layout(scene=dict(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    zaxis=dict(visible=False)
    ))  
    plot(fig, filename=fig_folder+f'Figure_9_past_future_{coor_s+1}.html', auto_open=False)

# =============================================================================
# Figure 8 
# =============================================================================

coor = dict_mat[n_mat].sort_values([6]).iloc[0,:6]
sub_array = array_3d_zero[int(coor[0]):int(coor[1])+1, int(coor[2]):int(coor[3])+1,int(coor[5])+1:int(coor[5])+7]
non_zero_indices = np.argwhere(sub_array!=0)
coordinates = [(idx[0],idx[1],idx[2]) for idx in non_zero_indices]
coordinates_1 = np.array(coordinates)
value_1 = sub_array[sub_array != 0]    
    
coor = dict_mat[n_mat].sort_values([6]).iloc[1,:6]
sub_array = array_3d_zero[int(coor[0]):int(coor[1])+1, int(coor[2]):int(coor[3])+1,int(coor[5])+1:int(coor[5])+7]
non_zero_indices = np.argwhere(sub_array!=0)
coordinates = [(idx[0],idx[1],idx[2]) for idx in non_zero_indices]
coordinates_2 = np.array(coordinates)
value_2 = sub_array[sub_array != 0]    
    
coordinates_1=coordinates_1/coordinates_1.max(axis=0)
coordinates_2=coordinates_2/coordinates_2.max(axis=0)
value_1=value_1/value_1.max()
value_2=value_2/value_2.max()

levels = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
def quantize(coords):
    return np.array([[levels[np.abs(levels - x).argmin()] for x in point] for point in coords])
q_coords_1 = quantize(coordinates_1)
q_coords_2 = quantize(coordinates_2)
all_coords = np.vstack((q_coords_1, q_coords_2))
all_values = np.concatenate((value_1, value_2))
df = pd.DataFrame(all_coords, columns=["x", "y", "z"])
df["value"] = all_values
merged_df = df.groupby(["x", "y", "z"]).mean().reset_index()
merged_coords = merged_df[["x", "y", "z"]].to_numpy()
mean_values = merged_df["value"].to_numpy()
merged_coords, mean_values
coordinates = np.array(merged_coords)
coordinates[:,1] = coordinates[:,1]*0.6
value = mean_values
dist_matrix = distance.cdist(coordinates, coordinates)
mst = minimum_spanning_tree(dist_matrix)
edges = np.transpose(mst.nonzero())
x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
flat_indices = value
cmap = cm.get_cmap('Reds')
norm = mcolors.LogNorm(vmin=0.1, vmax=value.max())
marker_colors = cmap(norm(value))
fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=10, color=marker_colors))])
for edge in edges:
    fig.add_trace(go.Scatter3d(
        x=[coordinates[edge[0], 0], coordinates[edge[1], 0]],
        y=[coordinates[edge[0], 1], coordinates[edge[1], 1]],
        z=[coordinates[edge[0], 2], coordinates[edge[1], 2]],
        mode='lines',
        line=dict(color='grey', width=2)))
fig.update_layout(scene=dict(
                    xaxis_title='Longitude',
                    yaxis_title='Latitude',
                    zaxis_title='Time'),
                    title='Cone - Mid SD',
                    showlegend=False)
fig.update_layout(scene=dict(
xaxis=dict(visible=False),
yaxis=dict(visible=False),
zaxis=dict(visible=False)
))   
plot(fig, filename=fig_folder+'Figure_8_scenario.html', auto_open=False)

# =============================================================================
# Share of fatalities and active cells included in active zones
# =============================================================================

sum_tot_in = 0
sum_cell_in = 0
for coor in final_traj:
    sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-24:-12].sum(axis=2)
    sum_tot_in += sub_array.sum()
    sum_cell_in += (array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-24:-12].sum(axis=2)>0).sum()
for coor in final_traj_2022:
    sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-36:-24].sum(axis=2)
    sum_tot_in += sub_array.sum()
    sum_cell_in += (array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-36:-24].sum(axis=2)>0).sum()
for coor in final_traj_sub1:
    sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-18:-6].sum(axis=2)
    sum_tot_in += sub_array.sum()
    sum_cell_in += (array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-18:-6].sum(axis=2)>0).sum()
for coor in final_traj_sub2:
    sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-30:-18].sum(axis=2)
    sum_tot_in += sub_array.sum()
    sum_cell_in += (array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-30:-18].sum(axis=2)>0).sum()

sum_tot = array_3d_zero[:,:,-24:-12].sum()+array_3d_zero[:,:,-36:-24].sum()+array_3d_zero[:,:,-18:-6].sum()+array_3d_zero[:,:,-30:-18].sum()
sum_cell = (array_3d_zero[:,:,-24:-12].sum(axis=2)>0).sum()+(array_3d_zero[:,:,-36:-24].sum(axis=2)>0).sum()+(array_3d_zero[:,:,-18:-6].sum(axis=2)>0).sum()+(array_3d_zero[:,:,-30:-18].sum(axis=2)>0).sum()

print(f'Percentage of fatalities inclued : {int((sum_tot_in/sum_tot)*100)}%')
print(f'Percentage of active cell inclued : {int((sum_cell_in/sum_cell)*100)}%')

# =============================================================================
# Figure D.1 
# =============================================================================

test_diff=[]
for lat in range(2,176):
    for long in range(2,175):
        for time_i in range(0,396):
            if array_3d_zero[lat,long,time_i:time_i+12].sum()>0:
                prev=(array_3d_zero[lat-2:lat+3,long-2:long+3,time_i:time_i+12].sum(axis=2)>0).sum()
                fol=(array_3d_zero[lat-2:lat+3,long-2:long+3,time_i+12:time_i+18].sum(axis=2)>0).sum()
                exa = (array_3d_zero[lat,long,time_i+12:time_i+18].sum()>0).sum()
                test_diff.append([prev,fol,exa])
                
test_diff=pd.DataFrame(test_diff)
test_diff.columns=['N_neighbors','Diffusion','Same']

fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(24, 10), sharex=True)
axes = axes.T.flatten()
for i, n in enumerate(range(1,16)):  
    subset = test_diff[test_diff['N_neighbors'] == n]
    diffusion_counts = subset['Diffusion'].value_counts(normalize=True).sort_index() * 100  
    max_value = diffusion_counts.max()
    colors = ['red' if v == max_value else 'royalblue' for v in diffusion_counts.values]
    axes[i].bar(diffusion_counts.index, diffusion_counts.values, color=colors, alpha=0.7)
    axes[i].set_ylabel(f'N={n}', rotation=0, labelpad=30, fontsize=25, verticalalignment='center')
    axes[i].grid(axis='y', linestyle='--', alpha=0.5)
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].spines['left'].set_visible(False)
    axes[i].spines['bottom'].set_visible(False)
    axes[i].tick_params(axis='both', labelsize=15)
axes[9].set_xlabel('Diffusion Value', fontsize=30)
plt.xticks(range(16))
plt.tight_layout(rect=[0, 0, 1, 0.98]) 
plt.savefig(fig_folder+'Figure_D1.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Figure 9
# =============================================================================

ind_pl=2
plot_3d=False
coor = final_traj[ind_pl]
sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-24:-6]
non_zero_indices = np.argwhere(sub_array!=0)
coordinates = [(idx[0],idx[1],idx[2]) for idx in non_zero_indices]
coordinates = np.array(coordinates)
weights_1 = sub_array[sub_array != 0]

list_pat = gr_b_tot_2023[ind_pl]
df_test = df_output_2023.loc[:,list_pat]
df_test= df_test.fillna(0)
df_input_test = df_input.iloc[-12:-12 + h].loc[:,list_pat]
df_preds_test_2_test = df_preds_views.iloc[:h].loc[:,list_pat]
df_views_coor = df_preds_test_2_test.reset_index().melt(id_vars='month_id', var_name='prio', value_name='Value')
df_views_coor = pd.merge(df_views_coor, df_prio, left_on='prio', right_index=True, how='inner')
df_views_coor= np.array(df_views_coor[['row', 'col','month_id','Value']])
df_views_coor[:,0] = df_views_coor[:,0]-coor['xmin']-88
df_views_coor[:,1] = df_views_coor[:,1]-coor['ymin']-311
df_views_coor[:,2] = df_views_coor[:,2]-505
df_views_coor= df_views_coor[df_views_coor[:,3]>0]

df_test_coor = df_test.reset_index().melt(id_vars='index', var_name='prio', value_name='Value')
df_test_coor = pd.merge(df_test_coor, df_prio, left_on='prio', right_index=True, how='inner')
df_test_coor= np.array(df_test_coor[['row', 'col','index','Value']])
df_test_coor[:,0] = df_test_coor[:,0]-coor['xmin']-88
df_test_coor[:,1] = df_test_coor[:,1]-coor['ymin']-311
df_test_coor[:,2] = df_test_coor[:,2]+12
df_test_coor= df_test_coor[df_test_coor[:,3]>0]
weights_train = weights_1[coordinates[:,2]<12]
weights_test = weights_1[coordinates[:,2]>=12]
coordinates_train = coordinates[coordinates[:,2]<12]
coordinates_test = coordinates[coordinates[:,2]>=12]

coordinates = pd.DataFrame(df_test_coor)
coordinates.columns=['col0','col1','col2','col3']
coordinates['col3'] = coordinates['col3'].round().astype(int)
coordinates=coordinates[coordinates['col3']>1]
coordinates.loc[:,'col2']=coordinates.loc[:,'col2']-12
value = np.array(coordinates.iloc[:,3])
coordinates = np.array(coordinates.iloc[:,:3])
dist_matrix = distance.cdist(coordinates, coordinates)
mst = minimum_spanning_tree(dist_matrix)
edges = np.transpose(mst.nonzero())
x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
flat_indices = value
cmap = cm.get_cmap('Reds')
norm = mcolors.LogNorm(vmin=1, vmax=value.max())
marker_colors = cmap(norm(value))
fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=10, color=marker_colors))])
for edge in edges:
    fig.add_trace(go.Scatter3d(
        x=[coordinates[edge[0], 0], coordinates[edge[1], 0]],
        y=[coordinates[edge[0], 1], coordinates[edge[1], 1]],
        z=[coordinates[edge[0], 2], coordinates[edge[1], 2]],
        mode='lines',
        line=dict(color='grey', width=2)))
fig.update_layout(scene=dict(
                    xaxis_title='Longitude',
                    yaxis_title='Latitude',
                    zaxis_title='Time'),
                    title='Cone - Mid SD',
                    showlegend=False)
fig.update_layout(scene=dict(
xaxis=dict(visible=False),
yaxis=dict(visible=False),
zaxis=dict(visible=False)
))  
plot(fig, filename=fig_folder+'Figure_9_forecast.html', auto_open=False)

# =============================================================================
# Figure C.1
# =============================================================================

df_views_coor = pd.DataFrame(df_views_coor, columns=['col0', 'col1', 'col2', 'col3'])
df_test_coor = pd.DataFrame(df_test_coor, columns=['col0', 'col1', 'col2', 'col3'])
df_true = pd.DataFrame(np.append(coordinates_test,weights_test.reshape(len(weights_test),1),axis=1), columns=['col0', 'col1', 'col2', 'col3'])

col_na=['Longitude','Latitude','Time']
fig, axes = plt.subplots(1, 3, figsize=(24, 6))
for i in range(3):
    if i in [0, 1]:
        x_range = pd.Index(np.arange(0, max(df_views_coor[f'col{i}'].max(), df_test_coor[f'col{i}'].max(), df_true[f'col{i}'].max()) + 1))
    elif i == 2:
        x_range = pd.Index(np.arange(12, 12 + h ))
    views_sum = df_views_coor.groupby(f'col{i}')['col3'].sum().reindex(x_range, fill_value=0)
    test_sum = df_test_coor.groupby(f'col{i}')['col3'].sum().reindex(x_range, fill_value=0)
    true_sum = df_true.groupby(f'col{i}')['col3'].sum().reindex(x_range, fill_value=0)
    axes[i].plot(views_sum, color='#008000', marker='o', label='views_coor',linestyle='--')
    axes[i].plot(test_sum, color='#df2226', marker='o', label='test_coor',linestyle='--')
    axes[i].plot(true_sum, color='black', marker='o', label='true',linewidth=3)
    axes[i].set_title(col_na[i],fontsize=25)
    axes[i].tick_params(axis='both', labelsize=20)
plt.tight_layout()
plt.savefig(fig_folder+'Figure_C1.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Figure 1
# =============================================================================

s_true = df_input.iloc[-24:-12].sum()
df_prio_pred_obs = pd.concat([df_prio,s_true],axis=1)
xcoord = df_prio_pred_obs.iloc[:, 2]
ycoord = df_prio_pred_obs.iloc[:, 3]
color_values = df_prio_pred_obs.iloc[:, 5].tolist()
geometry = [Polygon([(x - 0.25, y - 0.25), (x + 0.25, y - 0.25), 
             (x + 0.25, y + 0.25), (x - 0.25, y + 0.25)]) for x, y in zip(xcoord, ycoord)]
gdf = gpd.GeoDataFrame(np.array(df_prio_pred_obs.iloc[:,5]), geometry=geometry, columns=['Obs'])
borders = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

col_int = gr_b_tot_2023[n_mat]
xcoor_min = df_prio.loc[col_int]['xcoord'].min()-3
xcoor_max = df_prio.loc[col_int]['xcoord'].max()+3
ycoor_min = df_prio.loc[col_int]['ycoord'].min()-3
ycoor_max = df_prio.loc[col_int]['ycoord'].max()+3
gdf.index = df_prio.index
gdf_sub=gdf.loc[col_int]
coor = final_traj[n_mat]
sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-24:-12]
value = sub_array[sub_array != 0]
vmin = 1
vmax = value.max()
fig, axes = plt.subplots(1, 1, figsize=(24, 24))
borders.plot(ax=axes, color='#D3D3D3',edgecolor='#000000',linewidth=4)
gdf.boundary.plot(ax=axes, edgecolor='grey',alpha=0.1)
gdf_sub.plot(column='Obs', cmap='Reds',norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), ax=axes, edgecolor='black',linewidth=4)
axes.set_xlim(xcoor_min - 2, xcoor_max + 2)
axes.set_ylim(ycoor_min - 1, ycoor_max + 1)
for i in range(9):
    axes.text(xcoor_min+2.4,ycoor_max-3.15-(0.5*i),f'{i+1}',color='black', fontsize=50)
for i in range(7):
    axes.text(xcoor_min+2.9+(0.5*i),ycoor_max-2.65,f'{i+1}',color='black', fontsize=50)
axes.set_axis_off()
plt.tight_layout()
plt.savefig(fig_folder+'Figure_1a.png', dpi=300, bbox_inches='tight')
plt.show()

s_sf = df_output_2023_no_na.sum()
s_vi = df_preds_views.iloc[:h].sum()
s_true = df_input.iloc[-12:-6].sum()
df_prio_pred_obs = pd.concat([df_prio,s_sf,s_vi,s_true],axis=1)
xcoord = df_prio_pred_obs.iloc[:, 2]
ycoord = df_prio_pred_obs.iloc[:, 3]
color_values = df_prio_pred_obs.iloc[:, 6].tolist()
geometry = [Polygon([(x - 0.25, y - 0.25), (x + 0.25, y - 0.25), 
             (x + 0.25, y + 0.25), (x - 0.25, y + 0.25)]) for x, y in zip(xcoord, ycoord)]
gdf = gpd.GeoDataFrame(np.array(df_prio_pred_obs.iloc[:,5:]), geometry=geometry, columns=['SF','Views','Obs'])
borders = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

col_int = gr_b_tot_2023[n_mat]
xcoor_min = df_prio.loc[col_int]['xcoord'].min()-3
xcoor_max = df_prio.loc[col_int]['xcoord'].max()+3
ycoor_min = df_prio.loc[col_int]['ycoord'].min()-3
ycoor_max = df_prio.loc[col_int]['ycoord'].max()+3
gdf.index = df_prio.index
gdf_sub=gdf.loc[col_int]
coor = final_traj[n_mat]
sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-24:-12]
value = sub_array[sub_array != 0]
vmin = 1
vmax = value.max()
fig, axes = plt.subplots(1, 1, figsize=(24, 24))
borders.plot(ax=axes, color='#D3D3D3',edgecolor='#000000',linewidth=4)
gdf.boundary.plot(ax=axes, edgecolor='grey',alpha=0.1)
gdf_sub.plot(column='Views', cmap='Reds',norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), ax=axes, edgecolor='black',linewidth=4)
axes.set_xlim(xcoor_min - 2, xcoor_max + 2)
axes.set_ylim(ycoor_min - 1, ycoor_max + 1)
for i in range(9):
    axes.text(xcoor_min+2.4,ycoor_max-3.15-(0.5*i),f'{i+1}',color='black', fontsize=50)
for i in range(7):
    axes.text(xcoor_min+2.9+(0.5*i),ycoor_max-2.65,f'{i+1}',color='black', fontsize=50)
axes.set_axis_off()
plt.tight_layout()
plt.savefig(fig_folder+'Figure_1b.png', dpi=300, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 1, figsize=(24, 24))
borders.plot(ax=axes, color='#D3D3D3',edgecolor='#000000',linewidth=4)
gdf.boundary.plot(ax=axes, edgecolor='grey',alpha=0.1)
gdf_sub.plot(column='SF', cmap='Reds',norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), ax=axes, edgecolor='black',linewidth=4)
axes.set_xlim(xcoor_min - 2, xcoor_max + 2)
axes.set_ylim(ycoor_min - 1, ycoor_max + 1)
for i in range(9):
    axes.text(xcoor_min+2.4,ycoor_max-3.15-(0.5*i),f'{i+1}',color='black', fontsize=50)
for i in range(7):
    axes.text(xcoor_min+2.9+(0.5*i),ycoor_max-2.65,f'{i+1}',color='black', fontsize=50)
axes.set_axis_off()
plt.tight_layout()
plt.savefig(fig_folder+'Figure_1c.png', dpi=300, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 1, figsize=(24, 24))
borders.plot(ax=axes, color='#D3D3D3',edgecolor='#000000',linewidth=4)
gdf.boundary.plot(ax=axes, edgecolor='grey',alpha=0.1)
gdf_sub.plot(column='Obs', cmap='Reds',norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), ax=axes, edgecolor='black',linewidth=4)
axes.set_xlim(xcoor_min - 2, xcoor_max + 2)
axes.set_ylim(ycoor_min - 1, ycoor_max + 1)
for i in range(9):
    axes.text(xcoor_min+2.4,ycoor_max-3.15-(0.5*i),f'{i+1}',color='black', fontsize=50)
for i in range(7):
    axes.text(xcoor_min+2.9+(0.5*i),ycoor_max-2.65,f'{i+1}',color='black', fontsize=50)
axes.set_axis_off()
plt.tight_layout()
plt.savefig(fig_folder+'Figure_1d.png', dpi=300, bbox_inches='tight')
plt.show()

print(sum(gdf_sub['Obs']>7), sum(gdf_sub['SF']>7), sum(gdf_sub['Views']>7))

# =============================================================================
# Figure 10
# =============================================================================

h=6
df_mat_pf=[]
for enu,df_match in enumerate(dict_mat):
    coor = final_traj[enu]
    sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-24:-12]
    sub_array_plus = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-12:-6]
    r_s = sub_array_plus.sum()/sub_array.sum()
    
    bound_1=np.array([[0,0,0],list(sub_array_plus.shape)])
    non_zero_indices = np.argwhere(sub_array_plus!=0)
    if len(non_zero_indices)==0: # no observed future to compare with
        continue
    coordinates_1 = [(idx[0],idx[1],idx[2]) for idx in non_zero_indices]
    coordinates_1 = np.array(coordinates_1)
    coordinates_1 = (coordinates_1 - bound_1.min(axis=0)) / (bound_1.max(axis=0)-1 - bound_1.min(axis=0))
    coordinates_1 = np.nan_to_num(coordinates_1, nan=0.5)
    if len(non_zero_indices)>1:
        weights_1 = sub_array_plus[sub_array_plus != 0]
        mass_w1 = sum((weights_1.flatten() - weights_1.flatten().min(axis=0)) / (weights_1.flatten().max(axis=0) - weights_1.flatten().min(axis=0)))
        weights_1 = weights_1 / np.sum(weights_1)
    else:
        mass_w1=1
        weights_1=np.array([1])
    for i in range(len(df_match)):
        sub_a = array_3d_zero[df_match.iloc[i,0]:df_match.iloc[i,1],df_match.iloc[i,2]:df_match.iloc[i,3],df_match.iloc[i,5]:df_match.iloc[i,5]+h]
        sub_a = np.rot90(sub_a, k=df_match.iloc[i,7], axes=(0, 1))
        matc = array_3d_zero[df_match.iloc[i,0]:df_match.iloc[i,1],df_match.iloc[i,2]:df_match.iloc[i,3],df_match.iloc[i,4]:df_match.iloc[i,5]]
        sub_b = (sub_a - matc.min()) / (matc.max()-matc.min())
        matc = (matc - matc.min()) / (matc.max()-matc.min())
        r_m = sub_b.sum()/matc.sum()
        ratio_m = abs(r_s-r_m)
        
        bound=np.array([[0,0,0],list(sub_a.shape)])
        zero_indices_2 = np.argwhere(sub_a!=0)
        if len(zero_indices_2)>1:
            coordinates_2 = [(idx[0],idx[1],idx[2]) for idx in zero_indices_2]
            coordinates_2 = np.array(coordinates_2)
            coordinates_2 = (coordinates_2 - bound.min(axis=0)) / (bound.max(axis=0)-1 - bound.min(axis=0))
            coordinates_2 = np.nan_to_num(coordinates_2, nan=0.5)
            weights_2 = sub_a[sub_a!=0]
            mass_w2 = sum((weights_2.flatten() - weights_2.flatten().min(axis=0)) / (weights_2.flatten().max(axis=0) - weights_2.flatten().min(axis=0)))
            weights_2 = weights_2 / np.sum(weights_2)
            d_met=ot.dist(coordinates_1,coordinates_2,metric='euclidean')
            d_min=ot.emd2(weights_1,weights_2,d_met)
            df_mat_pf.append([df_match.iloc[i,6],df_match.iloc[i,8],ratio_m,d_min])
        else:
            pass
        
for enu,df_match in enumerate(dict_mat_2022):
    coor = final_traj_2022[enu]
    sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-36:-24]
    sub_array_plus = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-24:-18]
    r_s = sub_array_plus.sum()/sub_array.sum()
    
    bound_1=np.array([[0,0,0],list(sub_array_plus.shape)])
    non_zero_indices = np.argwhere(sub_array_plus!=0)
    if len(non_zero_indices)==0: # no observed future to compare with
        continue
    coordinates_1 = [(idx[0],idx[1],idx[2]) for idx in non_zero_indices]
    coordinates_1 = np.array(coordinates_1)
    coordinates_1 = (coordinates_1 - bound_1.min(axis=0)) / (bound_1.max(axis=0)-1 - bound_1.min(axis=0))
    coordinates_1 = np.nan_to_num(coordinates_1, nan=0.5)
    if len(non_zero_indices)>1:
        weights_1 = sub_array_plus[sub_array_plus != 0]
        mass_w1 = sum((weights_1.flatten() - weights_1.flatten().min(axis=0)) / (weights_1.flatten().max(axis=0) - weights_1.flatten().min(axis=0)))
        weights_1 = weights_1 / np.sum(weights_1)
    else:
        mass_w1=1
        weights_1=np.array([1])
    for i in range(len(df_match)):
        sub_a = array_3d_zero[df_match.iloc[i,0]:df_match.iloc[i,1],df_match.iloc[i,2]:df_match.iloc[i,3],df_match.iloc[i,5]:df_match.iloc[i,5]+h]
        sub_a = np.rot90(sub_a, k=df_match.iloc[i,7], axes=(0, 1))
        matc = array_3d_zero[df_match.iloc[i,0]:df_match.iloc[i,1],df_match.iloc[i,2]:df_match.iloc[i,3],df_match.iloc[i,4]:df_match.iloc[i,5]]
        sub_b = (sub_a - matc.min()) / (matc.max()-matc.min())
        matc = (matc - matc.min()) / (matc.max()-matc.min())
        r_m = sub_b.sum()/matc.sum()
        ratio_m = abs(r_s-r_m)
        
        bound=np.array([[0,0,0],list(sub_a.shape)])
        zero_indices_2 = np.argwhere(sub_a!=0)
        if len(zero_indices_2)>1:
            coordinates_2 = [(idx[0],idx[1],idx[2]) for idx in zero_indices_2]
            coordinates_2 = np.array(coordinates_2)
            coordinates_2 = (coordinates_2 - bound.min(axis=0)) / (bound.max(axis=0)-1 - bound.min(axis=0))
            coordinates_2 = np.nan_to_num(coordinates_2, nan=0.5)
            weights_2 = sub_a[sub_a!=0]
            mass_w2 = sum((weights_2.flatten() - weights_2.flatten().min(axis=0)) / (weights_2.flatten().max(axis=0) - weights_2.flatten().min(axis=0)))
            weights_2 = weights_2 / np.sum(weights_2)
            d_met=ot.dist(coordinates_1,coordinates_2,metric='euclidean')
            d_min=ot.emd2(weights_1,weights_2,d_met)
            df_mat_pf.append([df_match.iloc[i,6],df_match.iloc[i,8],ratio_m,d_min])
        else:
            pass        
        
for enu,df_match in enumerate(dict_mat_sub1):
    coor = final_traj_sub1[enu]
    sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-18:-6]
    sub_array_plus = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-6:]
    r_s = sub_array_plus.sum()/sub_array.sum()
    
    bound_1=np.array([[0,0,0],list(sub_array_plus.shape)])
    non_zero_indices = np.argwhere(sub_array_plus!=0)
    if len(non_zero_indices)==0: # no observed future to compare with
        continue
    coordinates_1 = [(idx[0],idx[1],idx[2]) for idx in non_zero_indices]
    coordinates_1 = np.array(coordinates_1)
    coordinates_1 = (coordinates_1 - bound_1.min(axis=0)) / (bound_1.max(axis=0)-1 - bound_1.min(axis=0))
    coordinates_1 = np.nan_to_num(coordinates_1, nan=0.5)
    if len(non_zero_indices)>1:
        weights_1 = sub_array_plus[sub_array_plus != 0]
        mass_w1 = sum((weights_1.flatten() - weights_1.flatten().min(axis=0)) / (weights_1.flatten().max(axis=0) - weights_1.flatten().min(axis=0)))
        weights_1 = weights_1 / np.sum(weights_1)
    else:
        mass_w1=1
        weights_1=np.array([1])
    for i in range(len(df_match)):
        sub_a = array_3d_zero[df_match.iloc[i,0]:df_match.iloc[i,1],df_match.iloc[i,2]:df_match.iloc[i,3],df_match.iloc[i,5]:df_match.iloc[i,5]+h]
        sub_a = np.rot90(sub_a, k=df_match.iloc[i,7], axes=(0, 1))
        matc = array_3d_zero[df_match.iloc[i,0]:df_match.iloc[i,1],df_match.iloc[i,2]:df_match.iloc[i,3],df_match.iloc[i,4]:df_match.iloc[i,5]]
        sub_b = (sub_a - matc.min()) / (matc.max()-matc.min())
        matc = (matc - matc.min()) / (matc.max()-matc.min())
        r_m = sub_b.sum()/matc.sum()
        ratio_m = abs(r_s-r_m)
        
        bound=np.array([[0,0,0],list(sub_a.shape)])
        zero_indices_2 = np.argwhere(sub_a!=0)
        if len(zero_indices_2)>1:
            coordinates_2 = [(idx[0],idx[1],idx[2]) for idx in zero_indices_2]
            coordinates_2 = np.array(coordinates_2)
            coordinates_2 = (coordinates_2 - bound.min(axis=0)) / (bound.max(axis=0)-1 - bound.min(axis=0))
            coordinates_2 = np.nan_to_num(coordinates_2, nan=0.5)
            weights_2 = sub_a[sub_a!=0]
            mass_w2 = sum((weights_2.flatten() - weights_2.flatten().min(axis=0)) / (weights_2.flatten().max(axis=0) - weights_2.flatten().min(axis=0)))
            weights_2 = weights_2 / np.sum(weights_2)
            d_met=ot.dist(coordinates_1,coordinates_2,metric='euclidean')
            d_min=ot.emd2(weights_1,weights_2,d_met)
            df_mat_pf.append([df_match.iloc[i,6],df_match.iloc[i,8],ratio_m,d_min])
        else:
            pass   
        
for enu,df_match in enumerate(dict_mat_sub2):
    coor = final_traj_sub2[enu]
    sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-30:-18]
    sub_array_plus = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-18:-12]
    r_s = sub_array_plus.sum()/sub_array.sum()
    
    bound_1=np.array([[0,0,0],list(sub_array_plus.shape)])
    non_zero_indices = np.argwhere(sub_array_plus!=0)
    if len(non_zero_indices)==0: # no observed future to compare with
        continue
    coordinates_1 = [(idx[0],idx[1],idx[2]) for idx in non_zero_indices]
    coordinates_1 = np.array(coordinates_1)
    coordinates_1 = (coordinates_1 - bound_1.min(axis=0)) / (bound_1.max(axis=0)-1 - bound_1.min(axis=0))
    coordinates_1 = np.nan_to_num(coordinates_1, nan=0.5)
    if len(non_zero_indices)>1:
        weights_1 = sub_array_plus[sub_array_plus != 0]
        mass_w1 = sum((weights_1.flatten() - weights_1.flatten().min(axis=0)) / (weights_1.flatten().max(axis=0) - weights_1.flatten().min(axis=0)))
        weights_1 = weights_1 / np.sum(weights_1)
    else:
        mass_w1=1
        weights_1=np.array([1])
    for i in range(len(df_match)):
        sub_a = array_3d_zero[df_match.iloc[i,0]:df_match.iloc[i,1],df_match.iloc[i,2]:df_match.iloc[i,3],df_match.iloc[i,5]:df_match.iloc[i,5]+h]
        sub_a = np.rot90(sub_a, k=df_match.iloc[i,7], axes=(0, 1))
        matc = array_3d_zero[df_match.iloc[i,0]:df_match.iloc[i,1],df_match.iloc[i,2]:df_match.iloc[i,3],df_match.iloc[i,4]:df_match.iloc[i,5]]
        sub_b = (sub_a - matc.min()) / (matc.max()-matc.min())
        matc = (matc - matc.min()) / (matc.max()-matc.min())
        r_m = sub_b.sum()/matc.sum()
        ratio_m = abs(r_s-r_m)
        
        bound=np.array([[0,0,0],list(sub_a.shape)])
        zero_indices_2 = np.argwhere(sub_a!=0)
        if len(zero_indices_2)>1:
            coordinates_2 = [(idx[0],idx[1],idx[2]) for idx in zero_indices_2]
            coordinates_2 = np.array(coordinates_2)
            coordinates_2 = (coordinates_2 - bound.min(axis=0)) / (bound.max(axis=0)-1 - bound.min(axis=0))
            coordinates_2 = np.nan_to_num(coordinates_2, nan=0.5)
            weights_2 = sub_a[sub_a!=0]
            mass_w2 = sum((weights_2.flatten() - weights_2.flatten().min(axis=0)) / (weights_2.flatten().max(axis=0) - weights_2.flatten().min(axis=0)))
            weights_2 = weights_2 / np.sum(weights_2)
            d_met=ot.dist(coordinates_1,coordinates_2,metric='euclidean')
            d_min=ot.emd2(weights_1,weights_2,d_met)
            df_mat_pf.append([df_match.iloc[i,6],df_match.iloc[i,8],ratio_m,d_min])
        else:
            pass   
        

df_mat_pf = pd.DataFrame(df_mat_pf)
column = df_mat_pf.iloc[:, 2]
q1 = column.quantile(0.25)
q3 = column.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df_mat_pf = df_mat_pf[(column > lower_bound) & (column < upper_bound)]

bins = [0, 0.25, 0.3, 0.35, np.inf]
labels = ['0-0.25','0.25-0.3' ,'0.3-0.35', '0.35+']

df1 = pd.DataFrame({'x': df_mat_pf.iloc[:, 0], 'y': df_mat_pf.iloc[:, 2]})
df1['bin'] = pd.cut(df1['x'], bins=bins, labels=labels)
summary1 = df1.groupby('bin').agg(mean=('y', 'mean'),
                                  count=('y', 'count'),
                                  std=('y', 'std'))
summary1['ci95'] = (summary1['std'] / np.sqrt(summary1['count'])) * t.ppf(0.975, summary1['count'] - 1)

df2 = pd.DataFrame({'x': df_mat_pf.iloc[:, 0], 'y': df_mat_pf.iloc[:, 3]})
df2['bin'] = pd.cut(df2['x'], bins=bins, labels=labels)
summary2 = df2.groupby('bin').agg(mean=('y', 'mean'),
                                  count=('y', 'count'),
                                  std=('y', 'std'))
summary2['ci95'] = (summary2['std'] / np.sqrt(summary2['count'])) * t.ppf(0.975, summary2['count'] - 1)

fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=False)
axes[0].errorbar(summary1.index.astype(str), summary1['mean'], yerr=summary1['ci95'], fmt='o', 
                 capsize=0, color='black', markersize=15, linewidth=4)
axes[0].set_ylabel('$\Delta R_{inc}$', fontsize=25)
axes[0].tick_params(axis='both', labelsize=20)
axes[0].grid(True)
axes[0].set_xlim(-0.5,3.5)
axes[1].errorbar(summary2.index.astype(str), summary2['mean'], yerr=summary2['ci95'], fmt='o', 
                 capsize=0, color='black', markersize=15, linewidth=4)
axes[1].set_ylabel('$EMD_{pf}$', fontsize=25)
axes[1].tick_params(axis='both', labelsize=20)
axes[1].grid(True)
axes[1].set_xlim(-0.5,3.5)
fig.supxlabel("$EMD_{pattern}$", fontsize=25)
plt.tight_layout()
plt.savefig(fig_folder+'Figure_10.png', dpi=300, bbox_inches='tight')
plt.show()

