import sys
import os
import numpy as np
import pandas as pd

from distances_features.feature_charts import combine_time_tables
from distances_features.feature_charts import parse_alignment_file


def get_number_of_lines_in_file(alignment_file) : 
    #Gets the number of lines in a alignment file
    with open(alignment_file) as file :
        nrows = sum(1 for _ in file)
    return nrows


def phone_mean_fbank(dataframe) :
    cols = [
        name for name in dataframe.columns
        if name.startswith('filterbank') or name == 'phoneme'
        ]

    return dataframe[cols].groupby('phoneme').mean()


class Analyzer:

    def __init__(self, alignment_file_path, wav_folder):
        self.alignments = parse_alignment_file(alignment_file_path)
        self.wav_folder = os.path.abspath(wav_folder)
        self.sound_list = [
            name for name in self.alignments['file_name'].unique()
            if os.path.isfile(os.path.join(self.wav_folder, name + '.wav'))
        ]
        self.alignments = self.alignments[
            self.alignments['file_name'].isin(self.sound_list)
        ].copy()

    def get_data(self, sound_file_name):
        if sound_file_name not in self.sound_list:
            raise KeyError("No alignment data for key '%s'." % sound_file_name)
        path = os.path.join(self.wav_folder, sound_file_name + '.wav')
        return combine_time_tables(path, self.alignments)



def align_chart(alignment_file_path):
    alignments = parse_alignment_file(alignment_file_path)
    

#normalisation par speaker