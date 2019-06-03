import sys
import os
import csv

from sklearn.metrics.pairwise import pairwise_distances

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from feature_charts import *
from distances import *


def normalization_speaker(dataframe) :
    # normalizes speaker filterbanks

    #     :param dataframe : feature dataframe for a given speaker (=sound file)
    #     :param save_path : path to which the normalized df is saved
    #     :type amount dataframe : panda df
    #     :type amount save_path : string
    #     :returns: normalized filterbank coefficients
    #     :rtype: dataframe

    cols = [
        name for name in dataframe.columns
        if name.startswith('filterbank')]

    mean_filterbanks = dataframe[cols].mean()
    df_mean = pd.DataFrame(mean_filterbanks)

    std_filterbanks = dataframe[cols].std()
    df_std = pd.DataFrame(std_filterbanks)

    for i in range(25):
        dataframe['filterbank_'+str(i)] = (dataframe['filterbank_'+str(i)] - float(df_mean.loc['filterbank_'+str(i)]) / float(df_std.loc['filterbank_'+str(i)]))

    return(dataframe)


def phone_mean_feature(dataframe) :
    # computes the mean of the filterbank coefficents by phonemes

    #     :param dataframe : the table of frames outputed by the combine-time-table function (V1)
    #     :type amount: dataframe
    #     :returns: a dataframe with the following columns : "phoneme", "F0", ... "F26" with "F0"... "F26" the mean filterbank coefficients
    #     :rtype: dataframe

    cols = [
        name for name in dataframe.columns
        if name.startswith('filterbank') or name == 'phoneme'
        ]

    mean_dataframe = dataframe[cols].groupby('phoneme').mean()

    return mean_dataframe


def parse_alignment_file(path_alignment_file):
    # creates a dataframe contraining for each phoneme its label, the file, the start time stamp
    # and the end time stamp. The entry is an aligmenent file in .txt. The key is the start time
    #     :param path_alignment_file : a path to the alignment file
    #     :type amount: path
    #     :returns: a dataframe with the following columns : "file name", "start", "end", "phoneme"
    #     :rtype: a dataframe

    with open(path_alignment_file, 'r') as file:
        df_alignment = pd.DataFrame([line.split() for line in file], columns=('file_name', 'start', 'end', 'phoneme'))
        df_alignment['start'] = df_alignment['start'].astype(float)
        df_alignment['end'] = df_alignment['end'].astype(float)
        return df_alignment


def calculate_mean_per_speaker(wav_folder_path, alignment_file_path, save_path_data, save_path_norm_data, save_path_mean) :
    #     :param wav_folder_path : path of the folder containing the sound files to analyze
    #     :param alignment_file_path : path of of access to the alignment file
    #     :param save_path_align : path to which the alignment dataframe is saved in csv
    #     :param save_path_df : path to which the combine_time_tables dataframe is savec in csv
    #     :type amount wav_folder_path : string
    #     :type amount alignment_file_path : string
    #     :type amount save_path_align : string
    #     :type amount save_path_df : string
    #     :returns: normalized output of the combine_time_tables function
    #     :rtype: dataframe

    alignment = parse_alignment_file(alignment_file_path)
    #sound_file_list = [wav_folder_path + '\\' + f for f in os.listdir(wav_folder_path) if f.endswith('.wav')]
    sound_file_list = [f for f in os.listdir(wav_folder_path) if f.endswith('.wav')]
    alignment_file_list = alignment['file_name'].unique().tolist()
    file_list = set(alignment_file_list).intersection(sound_file_list)

    for sound_file in file_list:
        sound_file_path = os.path.join(wav_folder_path, sound_file)
        phoneme_time_table = alignment[alignment['file_name'] == sound_file]
        dataframe = combine_time_tables(sound_file_path, phoneme_time_table)
        dataframe.to_csv(save_path_data, index = None, header=True)
        normalized_dataframe = normalization_speaker(dataframe)
        normalized_dataframe.to_csv(save_path_norm_data, index = None, header=True)
        mean_phoneme_per_speaker = phone_mean_feature(normalized_dataframe)
        mean_phoneme_per_speaker.to_csv(save_path_mean, index = None, header=True)
        return(mean_phoneme_per_speaker)


def distance_mean_realization_per_speaker(wav_folder_path_1, wav_folder_path_2, path_alignment_file, save_path_data_1, save_path_data_2, save_path_norm_data_1, save_path_norm_data_2, save_path_mean_1, save_path_mean_2, save_path_distance_matrix):

    #     :param wav_folder_path : path of the folder containing the sound files to analyze
    #     :param alignment_file_path : path of of access to the alignment file
    #     :param save_path_align : path to which the alignment dataframe is saved in csv
    #     :param save_path_df : path to which the combine_time_tables dataframe is savec in csv
    #     :type amount wav_folder_path : string
    #     :type amount alignment_file_path : string
    #     :type amount save_path_align : string
    #     :type amount save_path_df : string
    #     :returns: normalized output of the combine_time_tables function
    #     :rtype: dataframe

    alignment = parse_alignment_file(path_alignment_file)
    data_corpus_1 = calculate_mean_per_speaker(wav_folder_path_1, alignment_file_path, save_path_data_1, save_path_norm_data_1, save_path_mean_1)
    data_corpus_2 = calculate_mean_per_speaker(wav_folder_path_2, alignment_file_path, save_path_data_2, save_path_norm_data_2, save_path_mean_2)

    distance_matrix = cdist(data_corpus_1.values, data_corpus_2.values, metric ='euclidean')
    distance_matrix = pd.DataFrame(distance_matrix, index=data_corpus_1.index, columns=data_corpus_2.index)
    distance_matrix.to_csv(save_path_distance_matrix, index = None, header=True)
    return(distance_matrix)


def save_normalized_data(wav_folder_path_1, wav_folder_path_2, path_alignment_file, save_path_norm_data_1, save_path_norm_data_2):

    alignment = parse_alignment_file(path_alignment_file)
    #sound_file_list = [wav_folder_path + '\\' + f for f in os.listdir(wav_folder_path) if f.endswith('.wav')]
    sound_file_list_1 = [
        f for f in os.listdir(wav_folder_path_1) if f.endswith('.wav')
    ]
    alignment_file_list = alignment['file_name'].unique().tolist()
    file_list_1 = set(alignment_file_list).intersection(sound_file_list_1)

    sound_file_list_2 = [
        f for f in os.listdir(wav_folder_path_2) if f.endswith('.wav')
    ]
    file_list_2 = set(alignment_file_list).intersection(sound_file_list_2)

    if os.path.isfile(save_path_norm_data_1):
        os.remove(save_path_norm_data_1)

    for i, sound_file in enumerate(file_list_1):
        midpoints = get_midpoints(
            os.path.join(wav_folder_path_1, sound_file),
            alignment[alignment['file_name'] == sound_file]
        )
        data_1 = normalization_speaker(midpoints)
        data_1.to_csv(
            save_path_norm_data_1, mode='a', header=(i == 0)
        )

    if os.path.isfile(save_path_norm_data_2):
        os.remove(save_path_norm_data_2)

    for i, sound_file in enumerate(file_list_2):
        midpoints = get_midpoints(
            os.path.join(wav_folder_path_2, sound_file),
            alignment[alignment['file_name'] == sound_file]
        )
        data_2 = normalization_speaker(midpoints)
        data_2.to_csv(
            save_path_norm_data_2, mode='a', header=(i == 0)
        )
    return(data_1, data_2)


def select_data(data_corpus_1, data_corpus_2, selected_data_corpus_1, selected_data_corpus_2, nb_of_rows) :
    df_1 = pd.read_csv(data_corpus_1, index_col=0)
    df_2 = pd.read_csv(data_corpus_2, index_col=0)

    df_1.sample(n = nb_of_rows)
    df_1.to_csv(selected_data_corpus_1, header=True)

    df_2.sample(n = nb_of_rows)
    df_2.to_csv(selected_data_corpus_2, header=True)

    return(df_1, df_2)


def compute_distances_matrix(selected_data_corpus_1, selected_data_corpus_2):

    df_1 = pd.read_csv(selected_data_corpus_1, index_col=0)
    df_2 = pd.read_csv(selected_data_corpus_2, index_col=0)

    distance_matrix = pairwise_distances(df_1.values, df_2.values, metric ='euclidean')
    distance_matrix = pd.DataFrame(
        distance_matrix, index=df_1.index, columns=df_2.index
    )

    return(distance_matrix)

def transform_distance_matrix(dist_matrix) :
    dist_matrix.index.names = ['phoneme_2']
    dist_matrix.columns.names = ['phoneme_1']
    dist_matrix = pd.melt(dist_matrix, id_vars=dist_matrix.index, value_vars=dist_matrix.columns, var_name='pair_of_phonemes')

    #mean_dist_matrix = dist_matrix.groupby('pair_of_phonemes').mean()

    return(dist_matrix)

if __name__ == '__main__' :
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute distance matrix between features extracted from corpus 1 and corpus 2')
    parser.add_argument('wav_folder_path_1', metavar='path_1', type=str,
                        help='path to wav folder containing the data of corpus 1')
    parser.add_argument('wav_folder_path_2', metavar='path_2', type=str,
                        help='path to wav folder containing the data of corpus 2')
    parser.add_argument('path_alignment_file', metavar='alignment_file', type=str,
                        help='path to the alignment file')
    parser.add_argument('save_path_norm_data_1', metavar='save_norm_1', type=str,
                        help='path to csv file where normalized feature chart of corpus 1 is saved')
    parser.add_argument('save_path_norm_data_2', metavar='save_norm_2', type=str,
                        help='path to csv file where normalized feature chart of corpus 2 is saved')

    args = parser.parse_args()

    data = save_normalized_data(args.wav_folder_path_1, args.wav_folder_path_2, args.path_alignment_file, args.save_path_norm_data_1, args.save_path_norm_data_2)
    distances_matrix = compute_distances_matrix('norm_data_1.csv', 'norm_data_2.csv')
    distances_matrix.to_csv('distances_matrix.csv', header=True)
    print(distances_matrix)
    df = transform_distance_matrix(distances_matrix)
    print(df)

    # plt.figure()
    # sns.heatmap(df, cmap='Blues')
    # plt.xlabel("phones corpus 2 (English)")
    # plt.ylabel("phones corpus 1 (French)")
    # plt.savefig('distance_matrix.pdf')
    # plt.show()

    #means_per_speaker = calculate_mean_per_speaker('toy_data', 'toy_data/toy_data_alignment.txt', 'features_data_1.csv', 'norm_data_1.csv', 'mean_data_1.csv')
    #arguments : toy_data_1 toy_data_2 alignment/toy_data_alignment.txt norm_data_1.csv norm_data_2.csv dist_matrix.csv
