import sys
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from feature_charts import *


def normalization_speaker(dataframe, save_path=None) :
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
        dataframe['filterbank_'+str(i)] = (DataFrame['filterbank_'+str(i)] - float(df_mean.loc['filterbank_'+str(i)]) / float(df_std.loc['filterbank_'+str(i)]))

    if isinstance(save_path, str) :
        dataframe = dataframe.to_csv(save_path, index = None, header=True)

    return dataframe


def phone_mean_fbank(dataframe, save_path=None) :
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

    if isinstance(save_path, str) :
        mean_dataframe = mean_dataframe.to_csv(save_path, index = None, header=True)
    
    return mean_dataframe


def Analyze_data(wav_folder_path, alignment_file_path, save_path_data, save_path_norm_data, save_path_mean) :
    #     :param wav_folder_path : path of the folder containing the sound files to analyze
    #     :param alignment_file_path : path of of access to the alignment file
    #     :param save_path_align : path to which the alignment dataframe is saved in csv
    #     :param save_path_df : path to which the combine_time_table dataframe is savec in csv
    #     :type amount wav_folder_path : string
    #     :type amount alignment_file_path : string
    #     :type amount save_path_align : string
    #     :type amount save_path_df : string
    #     :returns: normalized output of the combine_time_table function
    #     :rtype: dataframe

    alignment = parse_alignment_file(alignment_file_path)
    #sound_file_list = [wav_folder_path + '\\' + f for f in os.listdir(wav_folder_path) if f.endswith('.wav')]
    sound_file_list = [f for f in os.listdir(wav_folder_path) if f.endswith('.wav')]
    alignment = alignment[alignment['file_name'].isin(sound_file_list).copy()]

    for sound_file in sound_file_list :
        dataframe = combine_time_tables(sound_file, alignment, save_path_data)
        save_data = dataframe.to_csv(save_path_data, index = None, header=True)
        normalized_dataframe = normalization_speaker(dataframe, save_path_norm_data)
        mean_phoneme_per_speaker = phone_mean_fbank(normalized_dataframe, save_path_mean)
        return(mean_phoneme_per_speaker)



def compute_distances_comparaison(wav_folder_path_1, wav_folder_path_2, alignment_file_path, save_path_norm_data_1, save_path_norm_data_2, save_path_mean_1, save_path_mean_2, save_path_distance_matrix):

    data_corpus_1 = Analyze_data(wav_folder_path_1, alignment_file_path, save_path_norm_data_1, save_path_mean_1)
    data_corpus_2 = Analyze_data(wav_folder_path_2, alignment_file_path, save_path_norm_data_2, save_path_mean_2)

    # Get the index of the midpoint frame for each phoneme as a pd.Series.
    mid_index = (
        (data_corpus_1['start'] // .01) + (.5 * (data_corpus_1['end'] - data_corpus_1['start']) // .01)
    ).astype(int)
    # Get a pandas.Series of midpoints with phonemes as index.
    midpoints_1 = pd.DataFrame( , index=data_corpus_1['phoneme'])

    mid_index = (
        (data_corpus_2['start'] // .01) + (.5 * (data_corpus_2['end'] - data_corpus_2['start']) // .01)
    ).astype(int)
    midpoints_2 = pd.DataFrame( , index=data_corpus_2['phoneme'])


    distance_matrix = squareform(cdist(midpoints_1.values, midpoints_2.values, metric ='euclidean'))
    distance_matrix = pd.DataFrame(distances, index=midpoints_1.index, columns=midpoints_2.index)
    return(distance_matrix)



Analyze_data('c:\\users\\alain\\desktop\\cogmaster\\cogmaster_s2\\stage\\distances_features', 'c:\\users\\alain\\desktop\\cogmaster\\cogmaster_s2\\stage\\distances_features\\toy_data_alignement.txt', 'c:\\users\\alain\\desktop\\cogmaster\\cogmaster_s2\\stage\\distances_features', 'c:\\users\\alain\\desktop\\cogmaster\\cogmaster_s2\\stage\\distances_features', 'c:\\users\\alain\\desktop\\cogmaster\\cogmaster_s2\\stage\\distances_features')


#alignment path : 'c:\\users\\alain\\desktop\\cogmaster\\cogmaster_s2\\stage\\distances_features\\toy_data_alignment.txt'
# sound folder path : 'c:\\users\\alain\\desktop\\cogmaster\\cogmaster_s2\\stage')


 #(V1)
# 0) charger les données du corpus anglais et corpus français, sauvegarder les données
# 1) normalisation par speaker (1 sound file = 1 speaker) : un df par speaker
# 2) moyenne des filterbank par speaker, puis par phoneme
# 3) 1 dataframe feature_chart par speaker (=par fichier son)
# 4) Calcul de la matrice de distances : on fait une boucle pour comparer speakers français et speakers anglais
# 5) sauvegarder matrice de distance

#(V2)
# 0) charger les données, sauvegarder les données
# 1) normalisation par speaker (1 sound file = 1 speaker) : un df par speaker
# 2) moyenne des filterbank par speaker, puis par phoneme
# 3) 1 dataframe feature_chart par speaker (=par fichier son)
# 4) Calcul de la matrice de distances : on fait une boucle pour comparer speakers français et speakers anglais
# 5) moyenne des distances (V2)
# 6) sauvegarder la matrice de distance

