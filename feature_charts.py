#if __name__ == "__main__":

import sys
import os

import scipy.io.wavfile as wav
import numpy as np
import pandas as pd
from python_speech_features import mfcc
from python_speech_features import logfbank
from scipy.spatial.distance import pdist, squareform


def get_filter_bank(sound_file) :
    """computes the filter bank coefficients of a sound file

        :param sound_file : sound file in format .wav
        :type amount: .wav file
        :returns: 26 filterbank coefficients per frame of 25ms every 10ms 
        :rtype: a numpy array"""

    (rate,sig) = wav.read(sound_file)
    fbank_feat = logfbank(sig,rate)
    fbank_feat = pd.DataFrame(fbank_feat)
    fbank_feat.columns = ['filterbank_%s' % i for i in range(26)]
    return(fbank_feat)


def get_number_of_windows(sound_file) : 
    """computes the number of frames for a given sound files based on the number of colums 
    of the vector stocking the filter bank coefficients 

        :param sound_file : sound file in format .wav
        :type amount: .wav file
        :returns: the number of frames of 25ms every 10ms in a sound file
        :rtype: int"""

    nb_windows = np.size(get_filter_bank(sound_file), 0)
    return(nb_windows)

def parse_alignment_file(path_alignment_file):
    """creates a dataframe contraining for each phoneme its label, the file, the start time stamp
    and the end time stamp. The entry is an aligmenent file in .txt. The key is the start time
            
        :param path_alignment_file : a path to the alignment file
        :type amount: path
        :returns: a dataframe with the following columns : "file name", "start", "end", "phoneme"
        :rtype: a dataframe"""


    assert os.path.isfile(path_alignment_file) and path_alignment_file.endswith('.txt')
    with open(path_alignment_file, 'r') as file:
        df_alignment = pd.DataFrame([line.split() for line in file], columns=('file_name', 'start', 'end', 'phoneme'))
        df_alignment['start'] = df_alignment['start'].astype(float)
        df_alignment['end'] = df_alignment['end'].astype(float)
        return df_alignment

def window_time_table(nb_windows) : 
    """creates a dataframe contraining the start time stamp and the end time stamp of 
    each frame given a number of frames.

        :param nb_windows : a number of frames
        :type amount: int
        :returns: a dataframe with the following columns : "start_frame", "end_frame"
        :rtype: a dataframe"""

    start = np.arange(0, nb_windows * .01, .01)
    window_time = pd.DataFrame({'start_frame': start, 'end_frame': start + 0.025})
    return window_time

def combine_time_tables(sound_file, path_alignment_file, save_path=None):
    """"combines the alignment file phoneme information with the filterbanks coefficients for a given sound file

     (phoneme, start time, end time) and the window time table
    (window index, start time, end time), table (phoneme, phoneme_start_time, phoneme_end_time, 
    window_index, wndow_start_time, window_end_time)

        :param sound_file : a sound file in format .wav
        :param path_alignment_file : path of the corresponding alignement chart 
        :param save_path : a path
        :type amount sound_file : .wav file
        :type amount alignments : dataframe
        :type amount save_path : path
        :returns: a dataframe with the following columns : "phoneme", "phoneme start time", "phoneme end time", 
        "window_index", "window_start_time", "window_end_time", "F0"..."F26" with "F0"... "F26" the filterbank coefficients.
        Each line corresponds to a frame.
        :rtype: dataframe"""

    # Import filterbanks.
    fbank = get_filter_bank(sound_file)
    frame_time_table = window_time_table(len(fbank))
    # Set the index to match the start frame of each phoneme.
    phoneme_time_table = parse_alignment_file(path_alignment_file)
    phoneme_time_table.index = (phoneme_time_table['start'] // .01).astype(int)
    # Drop phones that last less than the span of a frame.
    phoneme_time_table = phoneme_time_table.iloc[np.where(~phoneme_time_table.index.duplicated(keep='last'))]
    # Join both tables.
    phoneme_time_table = phoneme_time_table[phoneme_time_table['file_name'] == sound_file]
    frame_time_table = frame_time_table.join(phoneme_time_table).fillna(method='ffill')
    # Add filterbank coefficients as a single column.
    table = pd.concat([frame_time_table, fbank], axis=1)
    if isinstance(save_path, str):
        table = table.to_csv(save_path, index = None, header=True)
    return(table)


