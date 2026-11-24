# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:56:03 2024

@author: thoma
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
data_folder = '../Data/'
import requests
import pandas as pd
import numpy as np
df_list_input={f"fatalities001_2023_00_t01/pgm/sb?page={i}":i for i in range(1,475)}
df_input_t=pd.DataFrame()
for i in range(len(df_list_input)):
    response = requests.get(f'https://api.viewsforecasting.org/{list(df_list_input.keys())[i]}')
    json_data = response.json()
    df=pd.DataFrame(json_data["data"])
    df_input_t = pd.concat([df_input_t, df])
    df_input_t=df_input_t.reset_index(drop=True)
df_input = df_input_t.pivot(index='month_id', columns='pg_id', values='sc_pgm_sb_main')
df_input = np.exp(df_input)-1
df_input = df_input.iloc[:12,:]
df_input.to_csv(data_folder+'views_pred_2pg.csv')



df_list_input={f"fatalities002_2023_06_t01/pgm/sb?page={i}":i for i in range(1,475)}
df_input_t=pd.DataFrame()
for i in range(len(df_list_input)):
    response = requests.get(f'https://api.viewsforecasting.org/{list(df_list_input.keys())[i]}')
    json_data = response.json()
    df=pd.DataFrame(json_data["data"])
    df_input_t = pd.concat([df_input_t, df])
    df_input_t=df_input_t.reset_index(drop=True)
df_input = df_input_t.pivot(index='month_id', columns='pg_id', values='main_mean')
df_input = df_input.iloc[:12,:]
df_input.to_csv(data_folder+'views_pred_06_2023.csv')



df_list_input={f"fatalities001_2022_00_t01/pgm/sb?page={i}":i for i in range(1,475)}
df_input_t=pd.DataFrame()
for i in range(len(df_list_input)):
    response = requests.get(f'https://api.viewsforecasting.org/{list(df_list_input.keys())[i]}')
    json_data = response.json()
    df=pd.DataFrame(json_data["data"])
    df_input_t = pd.concat([df_input_t, df])
    df_input_t=df_input_t.reset_index(drop=True)
df_input = df_input_t.pivot(index='month_id', columns='pg_id', values='sc_pgm_sb_main')
df_input = np.exp(df_input)-1
df_input = df_input.iloc[:12,:]
df_input.to_csv(data_folder+'views_pred_1pg.csv')



df_list_input={f"fatalities001_2022_06_t01/pgm/sb?page={i}":i for i in range(1,475)}
df_input_t=pd.DataFrame()
for i in range(len(df_list_input)):
    response = requests.get(f'https://api.viewsforecasting.org/{list(df_list_input.keys())[i]}')
    json_data = response.json()
    df=pd.DataFrame(json_data["data"])
    df_input_t = pd.concat([df_input_t, df])
    df_input_t=df_input_t.reset_index(drop=True)
df_input = df_input_t.pivot(index='month_id', columns='pg_id', values='sc_pgm_sb_main')
df_input = np.exp(df_input)-1
df_input = df_input.iloc[:12,:]
df_input.to_csv(data_folder+'views_pred_06_2022.csv')
