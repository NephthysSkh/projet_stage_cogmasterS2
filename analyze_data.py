import sys
import os
import numpy as np
import pandas as pd

from feature_charts import *

class Analyzer:
    #loads the dataset and comptues the corresponding feature chart
    # reecrire avec os.walk

    def __init__(self, alignment_file_path, wav_folder_path):
        self.alignments = parse_alignment_file(alignment_file_path)
        with open(wav_folder_path) :
            self.sound_list = [name for name in self.alignments['file_name'].unique()\
            if os.path.isfile(os.path.join(self.wav_folder, name + '.wav'))]
        self.alignments = self.alignments[
        self.alignments['file_name'].isin(self.sound_list)].copy()

    def get_data(self, sound_file_name):
    #gets the feature chart for a given sound file

    #    :param sound_file : name of a sound file
    #    :type amount: string
    #    :returns: the output of the combine_time_table function for a given sound file
    #    :rtype: dataframe

        if sound_file_name not in self.sound_list:
            raise KeyError("No alignment data for key '%s'." % sound_file_name)
        path = os.path.join(self.wav_folder, sound_file_name + '.wav')
        return combine_time_tables(path, self.alignments)


def load_corpus(alignment_file_path, wav_folder_path, save_path=None) :
    """loads the corpus 

        :param sound_file : name of a sound file
        :param alignment_file_path : path of the corresponding alignement file
        :type amount sound_file : string
        :type amount alignement_file_path : string
        :returns: the output of get_data function for the sound files in a given folder
        :rtype: dataframes """

    Analyzer(alignment_file_path, wav_folder_path)
    dictionnary = {}
    for name in Analyzer.sound_list :
        dictionnary[name] = Analyzer.get_data(name)

    return(dictionnary)

def normalization_speaker(dataframe, save_path=None) :
    """normalizes speaker filterbanks

        :param dataframe : feature dataframe for a given speaker (=sound file)
        :param save_path : path to which the normalized df is saved
        :type amount dataframe : panda df
        :type amount save_path : string
        :returns: normalized filterbank coefficients
        :rtype: dataframe """

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


def phone_mean_fbank(dataframe) :
    """computes the mean of the filterbank coefficents by phonemes

        :param dataframe : the table of frames outputed by the combine-time-table function (V1)
        :type amount: dataframe
        :returns: a dataframe with the following columns : "phoneme", "F0", ... "F26" with "F0"... "F26" the mean filterbank coefficients
        :rtype: dataframe"""

    cols = [
        name for name in dataframe.columns
        if name.startswith('filterbank') or name == 'phoneme'
        ]

    return dataframe[cols].groupby('phoneme').mean()



# if __name__ == '__main__':  
#     import sys
#     import os

#     import scipy.io.wavfile as wav
#     import numpy as np
#     import pandas as pd
#     from python_speech_features import mfcc
#     from python_speech_features import logfbank
#     from scipy.spatial.distance import pdist, squareform

"""Analyzer('c:\\users\\alain\\desktop\\cogmaster\\cogmaster_s2\\stage\\distances_features\\toy_data_alignment.txt', 'c:\\users\\alain\\desktop\\cogmaster\\cogmaster_s2\\stage')
Analyzer.get_data('animal.wav')"""

parse_alignment_file('c:\\users\\alain\\desktop\\cogmaster\\cogmaster_s2\\stage\\distances_features\\toy_data_alignement.txt')

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

