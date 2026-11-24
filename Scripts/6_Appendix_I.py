# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 00:13:43 2025

@author: thoma
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
data_folder = '../Data/'
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from shape import Shape,finder
import pickle
import ot

run_2d_sf = False # Put True to rerun the 2d model (Around 1 day)

df_input=pd.read_csv(data_folder+'df_input_pg.csv',index_col=0,parse_dates=True)
df_input.columns = list(map(int, df_input.columns))
df_input.loc[515,148759]=0 #Ethiopia
df_input = df_input.loc[:, (df_input.sum()>0)]
df_input_learn = df_input.loc[:, (df_input.sum()>100)]

# =============================================================================
# Original ShapeFinder forecast 
# =============================================================================

if run_2d_sf:
    pred_tot=[]
    for coun in range(len(df_input.columns)):
        if not (df_input.iloc[-18:-6,coun]==0).all():
            shape = Shape()
            shape.set_shape(df_input.iloc[-18:-6,coun])
            find = finder(df_input_learn.iloc[:-12,:],shape)
            find.find_patterns(min_d=0.3,metric='dtw',dtw_sel=0,select=True,min_mat=5)
            pred = find.predict(horizon=6,clu_thres=3)
            pred = pred*(df_input.iloc[-18:-6,coun].max()-df_input.iloc[-18:-6,coun].min())+df_input.iloc[-18:-6,coun].min()
        else:
            pred = pd.Series([0]*6)
        pred_tot.append(pred)
    df_pred_ori = pd.DataFrame(pred_tot).T
    df_pred_ori.to_csv(data_folder+'Results/2d_sf.csv')

# Stored forecasts of the original ShapeFinder (2d_sf.csv)
df_2d = pd.read_csv(data_folder+'Results/2d_sf.csv',index_col=0)

# =============================================================================
# Table I4  
# =============================================================================

def calculate_EMD_d(df_test,df_input_test,weights_cor,coordinates,df_prio,coor):
    # Same function as in 3_Main_Analysis.py (zone EMD between forecast and observed)
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

# All PRIO-Grid cells (zones also contain cells that never had fatalities)
df_full=pd.read_csv(data_folder+'df_input_pg.csv',index_col=0,parse_dates=True)
df_full.columns = list(map(int, df_full.columns))
df_full.loc[515,148759]=0 #Ethiopia
df_prio = pd.read_csv(data_folder+'prio.csv',index_col=0,parse_dates=True)
df_prio=df_prio.loc[df_full.columns]
df_prio_xy = df_prio.iloc[:,:2]
array_3d = np.full((df_prio_xy['row'].max(), df_prio_xy['col'].max(), len(df_full.index)), np.nan)
for pg_id, (row, col) in df_prio_xy.iterrows():
    array_3d[row-1, col-1, :] = df_full[pg_id].values
array_3d=array_3d[df_prio_xy['row'].min():,df_prio_xy['col'].min():,:]
array_3d_zero = np.nan_to_num(array_3d, nan=0)

with open(data_folder+'Results/final_traj_sub1.pkl', 'rb') as f:
    final_traj_sub1 = pickle.load(f)

# 3d_sf.csv: zone forecasts concatenated zone by zone 
df_3d_z = pd.read_csv(data_folder+'Results/3d_sf.csv',index_col=0)
ids = [int(c.split('.')[0]) for c in df_3d_z.columns]
df_3d_z.columns = range(df_3d_z.shape[1])
df_2d_z = df_2d.copy()
df_2d_z.columns = df_input.columns
df_true_z = df_full.iloc[-6:].reset_index(drop=True)

def in_zone(pg_id,coor):
    r = df_prio.loc[pg_id,'row']-88
    c = df_prio.loc[pg_id,'col']-311
    return (coor['xmin']<=r<=coor['xmax']) and (coor['ymin']<=c<=coor['ymax'])

blocks=[]
cur=[]
j=0
for k,pg_id in enumerate(ids):
    while (not in_zone(pg_id,final_traj_sub1[j])) or (pg_id in [ids[x] for x in cur]):
        blocks.append(cur)
        cur=[]
        j=j+1
    cur.append(k)
blocks.append(cur)

zone_err=[]
err_3d=[]
err_2d=[]
for j,b in enumerate(blocks):
    cells = [ids[k] for k in b]
    pred_3d = df_3d_z.iloc[:,b].copy()
    pred_3d.columns = cells
    pred_3d = pred_3d.fillna(0)
    pred_2d = pd.DataFrame({c:(df_2d_z[c].values if c in df_2d_z.columns else np.zeros(6)) for c in cells})
    obs = df_true_z[cells]
    err_3d.extend((pred_3d-obs).values.flatten())
    err_2d.extend((pred_2d-obs).values.flatten())
    coor = final_traj_sub1[j]
    sub_array = array_3d_zero[coor['xmin']:coor['xmax']+1, coor['ymin']:coor['ymax']+1,-18:]
    coordinates = np.argwhere(sub_array!=0)
    weights_1 = sub_array[sub_array != 0]
    m_3d = abs(pred_3d.values.sum()-obs.values.sum())
    m_2d = abs(pred_2d.values.sum()-obs.values.sum())
    emd_3d = calculate_EMD_d(pred_3d.copy(),obs,weights_1,coordinates,df_prio,coor)
    emd_2d = calculate_EMD_d(pred_2d.copy(),obs,weights_1,coordinates,df_prio,coor)
    zone_err.append([m_3d,m_2d,emd_3d,emd_2d])

zone_err = pd.DataFrame(zone_err,columns=['MAE_3D','MAE_2D','EMD_3D','EMD_2D'])
err_3d = np.array(err_3d)
err_2d = np.array(err_2d)

# Zone level: log(error+1), mean and standard error over zones
zone_emd_3d = np.log(zone_err['EMD_3D']+1)
zone_emd_2d = np.log(zone_err['EMD_2D']+1)
zone_mae_3d = np.log(zone_err['MAE_3D']+1)
zone_mae_2d = np.log(zone_err['MAE_2D']+1)
zone_mse_3d = np.log(zone_err['MAE_3D']**2+1)
zone_mse_2d = np.log(zone_err['MAE_2D']**2+1)
# PRIO-Grid level: log(error+1), mean and standard error over cell-months
prio_mae_3d = np.log(np.abs(err_3d)+1)
prio_mae_2d = np.log(np.abs(err_2d)+1)
prio_mse_3d = np.log(err_3d**2+1)
prio_mse_2d = np.log(err_2d**2+1)

df_table_I = pd.DataFrame(index=['3D ShapeFinder','Original ShapeFinder'])
df_table_I['Zone EMD'] = [f"{zone_emd_3d.mean():.4f} ({zone_emd_3d.std()/np.sqrt(len(zone_emd_3d)):.4f})",
                          f"{zone_emd_2d.mean():.4f} ({zone_emd_2d.std()/np.sqrt(len(zone_emd_2d)):.4f})"]
df_table_I['Zone MAE'] = [f"{zone_mae_3d.mean():.4f} ({zone_mae_3d.std()/np.sqrt(len(zone_mae_3d)):.4f})",
                          f"{zone_mae_2d.mean():.4f} ({zone_mae_2d.std()/np.sqrt(len(zone_mae_2d)):.4f})"]
df_table_I['Zone MSE'] = [f"{zone_mse_3d.mean():.4f} ({zone_mse_3d.std()/np.sqrt(len(zone_mse_3d)):.4f})",
                          f"{zone_mse_2d.mean():.4f} ({zone_mse_2d.std()/np.sqrt(len(zone_mse_2d)):.4f})"]
df_table_I['Prio MAE'] = [f"{prio_mae_3d.mean():.4f} ({np.std(prio_mae_3d)/np.sqrt(len(prio_mae_3d)):.4f})",
                          f"{prio_mae_2d.mean():.4f} ({np.std(prio_mae_2d)/np.sqrt(len(prio_mae_2d)):.4f})"]
df_table_I['Prio MSE'] = [f"{prio_mse_3d.mean():.4f} ({np.std(prio_mse_3d)/np.sqrt(len(prio_mse_3d)):.4f})",
                          f"{prio_mse_2d.mean():.4f} ({np.std(prio_mse_2d)/np.sqrt(len(prio_mse_2d)):.4f})"]
print(df_table_I.to_string())
