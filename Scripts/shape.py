# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 17:52:04 2023

@author: thoma
"""

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from dtaidistance import dtw,ed
import bisect
from scipy.cluster.hierarchy import linkage, fcluster

# =============================================================================
# ShapeFinder for Time series Autoregressive
# =============================================================================

def int_exc(seq_n, win):
    """
    Create intervals and exclude list for the given normalized sequences.

    Args:
        seq_n (list): A list of normalized sequences.
        win (int): The window size for pattern matching.

    Returns:
        tuple: A tuple containing the exclude list, intervals, and the concatenated testing sequence.
    """
    n_test = []  # List to store the concatenated testing sequence
    to = 0  # Variable to keep track of the total length of concatenated sequences
    exclude = []  # List to store the excluded indices
    interv = [0]  # List to store the intervals

    for i in seq_n:
        n_test = np.concatenate([n_test, i])  # Concatenate each normalized sequence to create the testing sequence
        to = to + len(i)  # Calculate the total length of the concatenated sequence
        exclude = exclude + [*range(to - win, to)]  # Add the excluded indices to the list
        interv.append(to)  # Add the interval (end index) for each sequence to the list

    # Return the exclude list, intervals, and the concatenated testing sequence as a tuple
    return exclude, interv, n_test

class Shape():
    """
    A class to set custom shape using a graphical interface, user-provided values or random values.

    Attributes:
        time (list): List of x-coordinates representing time.
        values (list): List of y-coordinates representing values.
        window (int): The window size for the graphical interface.
    """

    def __init__(self, time=len(range(10)), values=[0.5]*10, window=10):
        """
        Args:
            time (int): The initial number of time points.
            values (list): The initial values corresponding to each time point.
            window (int): The window size for the graphical interface.
        """
        self.time = time
        self.values = values
        self.window = window


    def set_shape(self,input_shape):
        try:
            input_shape=pd.Series(input_shape)
            input_shape=(input_shape-input_shape.min())/(input_shape.max()-input_shape.min())
            self.time=list(range(len(input_shape)))
            self.values = input_shape.tolist()
            self.window=len(input_shape.tolist())
        except: 
            print('Wrong format, please provide a compatible input.')
        

class finder():
    """
    A class to find and predict custom patterns in a given dataset using an interactive shape finder.

    Attributes:
        data (DataFrame): The dataset containing time series data.
        Shape (Shape): An instance of the Shape class used for interactive shape finding.
        sequences (list): List to store the found sequences matching the custom shape.
    """
    def __init__(self,data,Shape=Shape(),sequences=[]):
        """
        Initializes the finder object with the given dataset and Shape instance.

        Args:
            data (DataFrame): The dataset containing time series data.
            Shape (Shape, optional): An instance of the Shape class for shape finding. Defaults to Shape().
            sequences (list, optional): List to store the found sequences matching the custom shape. Defaults to [].
        """
        self.data=data
        self.Shape=Shape
        self.sequences=sequences
        
    def find_patterns(self, metric='euclidean', min_d=0.5, dtw_sel=0, select=True, min_mat=0):
        """
        Finds custom patterns in the given dataset using the interactive shape finder.
    
        Args:
            metric (str, optional): The distance metric to use for shape matching. 'euclidean' or 'dtw'. Defaults to 'euclidean'.
            min_d (float, optional): The minimum distance threshold for a matching sequence. Defaults to 0.5.
            dtw_sel (int, optional): The window size variation for dynamic time warping (Only for 'dtw' mode). Defaults to 0.
            select (bool, optional): Whether to include overlapping patterns. Defaults to True.
            min_mat (int, optional): The minimum number of matching sequences. Defaults to 0.
        """
        # Clear any previously stored sequences
        self.sequences = []
        
        # Check if dtw_sel is zero when metric is 'euclidean'
        if metric=='euclidean':
            dtw_sel=0
    
        # Extract individual columns (time series) from the data
        seq = []
        for i in range(len(self.data.columns)): 
            seq.append(self.data.iloc[:, i])
    
        # Normalize each column (time series)
        seq_n = []
        for i in seq:
            seq_n.append((i - i.mean()) / i.std())
    
        # Get exclude list, intervals, and a testing sequence for pattern matching
        exclude, interv, n_test = int_exc(seq, self.Shape.window)
    
        # Convert custom shape values to a pandas Series and normalize it
        seq1 = pd.Series(data=self.Shape.values)
        if seq1.var() != 0.0:
            seq1 = (seq1 - seq1.min()) / (seq1.max() - seq1.min())
        else :    
            seq1 = [0.5]*len(seq1)
        seq1 = np.array(seq1)
    
        # Initialize the list to store the found sequences that match the custom shape
        tot = []
    
        if dtw_sel == 0:
            # Loop through the testing sequence
            for i in range(len(n_test)):
                # Check if the current index is not in the exclude list
                if i not in exclude:
                    seq2 = n_test[i:i + self.Shape.window]
                    if seq2.var() != 0.0:
                        seq2 = (seq2 - seq2.min()) / (seq2.max() - seq2.min())
                    else:
                        seq2 = np.array([0.5]*len(seq2))
                    try:
                        if metric == 'euclidean':
                            # Calculate the Euclidean distance between the custom shape and the current window
                            dist = ed.distance(seq1, seq2)
                        elif metric == 'dtw':
                            # Calculate the Dynamic Time Warping distance between the custom shape and the current window
                            dist = dtw.distance(seq1, seq2,use_c=True)
                        tot.append([i, dist, self.Shape.window])
                    except:
                        # Ignore any exceptions (e.g., divide by zero)
                        pass
        else:
            # Loop through the range of window size variations (dtw_sel)
            for lop in range(int(-dtw_sel), int(dtw_sel) + 1):
                # Get exclude list, intervals, and a testing sequence for pattern matching with the current window size
                exclude, interv, n_test = int_exc(seq_n, self.Shape.window + lop)
                for i in range(len(n_test)):
                    # Check if the current index is not in the exclude list
                    if i not in exclude:
                        seq2 = n_test[i:i + int(self.Shape.window + lop)]
                        if seq2.var() != 0.0:
                            seq2 = (seq2 - seq2.min()) / (seq2.max() - seq2.min())
                        else:
                            seq2 = np.array([0.5]*len(seq2))
                        try:
                            # Calculate the Dynamic Time Warping distance between the custom shape and the current window
                            dist = dtw.distance(seq1, seq2)
                            tot.append([i, dist, self.Shape.window + lop])
                        except:
                            # Ignore any exceptions (e.g., divide by zero)
                            pass
    
        # Create a DataFrame from the list of sequences and distances, sort it by distance, and filter based on min_d
        tot = pd.DataFrame(tot)
        tot = tot.sort_values([1])
        tot_cut = tot[tot[1] < min_d]
        tot_cut = tot_cut.iloc[:1000,:]
        toti = tot_cut[0]
    
        if select:
            n = len(toti)
            diff_data = {f'diff{i}': toti.diff(i) for i in range(1, n + 1)}
            diff_df = pd.DataFrame(diff_data).fillna(self.Shape.window)
            diff_df = abs(diff_df)
            tot_cut = tot_cut[diff_df.min(axis=1) >= (self.Shape.window / 2)]
    
        if len(tot_cut) > min_mat:
            # If there are selected patterns, store them along with their distances in the 'sequences' list
            for c_lo in range(len(tot_cut)):
                i = tot_cut.iloc[c_lo, 0]
                win_l = int(tot_cut.iloc[c_lo, 2])
                exclude, interv, n_test = int_exc(seq_n, win_l)
                col = seq[bisect.bisect_right(interv, i) - 1].name
                index_obs = seq[bisect.bisect_right(interv, i) - 1].index[i - interv[bisect.bisect_right(interv, i) - 1]]
                obs = self.data.loc[index_obs:, col].iloc[:win_l]
                self.sequences.append([obs, tot_cut.iloc[c_lo, 1]])
        else:
            flag_end=False
            min_mat_loop=min_mat
            while flag_end==False:
                tot_cut = tot.iloc[:min_mat_loop,:]
                toti = tot_cut[0]
                if select:
                    n = len(toti)
                    diff_data = {f'diff{i}': toti.diff(i) for i in range(1, n + 1)}
                    diff_df = pd.DataFrame(diff_data).fillna(self.Shape.window)
                    diff_df = abs(diff_df)
                    tot_cut = tot_cut[diff_df.min(axis=1) >= (self.Shape.window / 2)]
                if len(tot_cut) > min_mat:
                    for c_lo in range(len(tot_cut)):
                        i = tot_cut.iloc[c_lo, 0]
                        win_l = int(tot_cut.iloc[c_lo, 2])
                        exclude, interv, n_test = int_exc(seq_n, win_l)
                        col = seq[bisect.bisect_right(interv, i) - 1].name
                        index_obs = seq[bisect.bisect_right(interv, i) - 1].index[i - interv[bisect.bisect_right(interv, i) - 1]]
                        obs = self.data.loc[index_obs:, col].iloc[:win_l]
                        self.sequences.append([obs, tot_cut.iloc[c_lo, 1]])
                    flag_end=True
                else:
                    min_mat_loop=min_mat_loop+1

        
    def predict(self, horizon=6, clu_thres=3):
        """
        Predicts future values based on historical sequences using hierarchical clustering.
    
        Args:
            horizon (int): The number of future time steps to predict.
            clu_thres (int): The threshold for clustering, affecting the number of clusters.
    
        Returns:
            pd.Series: The final predicted sequence.
        """
        # Ensure sequences exist before proceeding
        if len(self.sequences) == 0:
            raise Exception('No shape found, please fit before predict.')
        # Extract key statistics from stored sequences
        tot_seq = [
            [series.name, series.index[-1], series.min(), series.max(), series.sum()] 
            for series, weight in self.sequences]
    
        pred_seq = []
        # Generate future sequences for each stored sequence
        for col, last_date, mi, ma, somme in tot_seq:
            date = self.data.index.get_loc(last_date)  # Get index position of the last known date
            if date + horizon < len(self.data):
                # Extract future values for the given column
                seq = self.data.iloc[date + 1 : date + 1 + horizon, self.data.columns.get_loc(col)].reset_index(drop=True)
                # Normalize sequence using min-max scaling
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())

        tot_seq = pd.DataFrame(pred_seq)
        # Perform hierarchical clustering
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon / clu_thres, criterion='distance')
        # Assign clusters to the sequences
        tot_seq['Cluster'] = clusters
        # Compute mean values per cluster
        val_sce = tot_seq.groupby('Cluster').mean()
        # Determine the most frequent cluster
        pr = round(pd.Series(clusters).value_counts(normalize=True).sort_index(), 2)
        # Extract the cluster with the highest frequency
        pred_ori = val_sce.loc[pr == pr.max(), :]
        # Compute the mean prediction across sequences in the most frequent cluster
        pred_ori = pred_ori.mean(axis=0)
        # Retrieve original shape values for denormalization
        seq1 = pd.Series(data=self.Shape.values)
        # Denormalize predictions back to original scale
        preds = pred_ori * (seq1.max() - seq1.min()) + seq1.min()
    
        return preds
