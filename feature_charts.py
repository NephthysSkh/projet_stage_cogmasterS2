from scipy.spatial.distance import pdist, squareform
import scipy.io.wavfile as wav
import numpy as np
import pandas as pd
import sys
import os
from python_speech_features import mfcc
from python_speech_features import logfbank

def get_filter_bank(sound_file) :
    #computes the filter bank coefficients of a sound file and outputs a vector of seize nb_of_
    #windows x 26 (number of filter). Each frame is 25ms long and for a step of 10ms

    (rate,sig) = wav.read(sound_file)
    fbank_feat = logfbank(sig,rate)
    return(fbank_feat)


def get_number_of_windows(sound_file) : 
    #computes the number of frames for a given sound files based on the number of colums 
    #of the vector stocking the filter bank coefficients 

    nb_windows = np.size(get_filter_bank(sound_file), 0)
    return(nb_windows)


def get_number_of_lines_in_file(alignment_file) : 
    #Gets the number of lines in a alignment file
    with open(alignment_file) as file :
        for i, line in enumerate(file):
            pass
    return(i + 1)

def parse_alignment_file(path_alignment_file):
#creates a dictionnary contraining for each phoneme its label, the start time stamp
    #and the end time stamp. The entry is an aligmenent file in .txt. The key is the strat time

    for alignment_file in os.listdir(path_alignment_file):
        if alignment_file.endswith('.txt') :
            with open(alignment_file, 'r') as file:
                df_alignment = pd.DataFrame([line.split() for line in file], columns=('file_name', 'start', 'end', 'phoneme'))
                df_alignment['start'] = df_alignment['start'].astype(float)
                df_alignment['end'] = df_alignment['end'].astype(float)
                return(df_alignment)

    return None

def window_time_table(sound_file) : 
    #creates a dictionnary contraining the start time stamp and the end time stamp of 
    #each window in s. The entry is a sound file in .wav

    window_time = {}
    nb_windows = get_number_of_windows(sound_file)
    start = [0]
    end = [0.025]
    time_count = [0]

    for i in range(nb_windows) :
        start.insert(i+1, start[i] + 0.010)
        end.insert(i+1, start[i+1] + 0.025)
        window_time[i] = [start[i], end[i]]

    window_time = pd.DataFrame(window_time).T
    window_time.columns = ['start_frame', 'end_frame']
    return(window_time)

def get_filterbank_chart(sound_file, alignment_file, path) :
    #creates a chart based on MFCC coefficients given a sound file and an alignment_file

    df_fbank = pd.DataFrame(get_filter_bank(sound_file))
    feature_fbank_chart = df_fbank.to_csv (path, index = None, header=True)
    return(df_fbank)

def get_mfcc_chart(sound_file, path) : 
    #creates a chart based on the MFCC coefficients given a sound_file

    (rate,sig) = wav.read(sound_file)
    mfcc_feat = mfcc(sig,rate)
    df_mfcc_feat = pd.DataFrame(mfcc_feat)
    feature_mfcc_chart = df_mfcc_feat.to_csv (path, index = None, header=True)
    return(df_mfcc_feat)


def combine_time_tables(sound_file, path_alignment_file):
    """"returns the synthesis of the real time table (phoneme, start time, end time) and the window time table
    (window index, start time, end time), table (phoneme, phoneme_start_time, phoneme_end_time, 
    window_index, wndow_start_time, window_end_time)"""

    phoneme_time_table = parse_alignment_file(path_alignment_file)
    table = window_time_table(sound_file)
    # Set the index to match the start frame of each phoneme.
    phoneme_time_table.index = (phoneme_time_table['start'] // .01).astype(int)
    # Drop phones that last less than the span of a frame.
    phoneme_time_table = phoneme_time_table.iloc[np.where(~phoneme_time_table.index.duplicated(keep='last'))]
    # Join both tables.

    for index, row in phoneme_time_table.iterrows() :
        if sound_file != row['file_name'] :
            phoneme_time_table.drop(index)

    table = table.join(phoneme_time_table).fillna(method='ffill')
    # Add filterbank coefficients as a single column.
    table['filterbank'] = get_filter_bank(sound_file).tolist()

    #time_table_chart = table.to_csv (path, index = None, header=True)
    return(table)


def get_midpoints(sound_file, path_alignment_file):
    """Gets the midpoint vector for each phoneme and comptues the distance matrix"""

    phoneme_time_table = parse_alignment_file(path_alignment_file)
    filterbank = get_filter_bank(sound_file)
    # Get the index of the midpoint frame for each phoneme as a pd.Series.
    mid_index = (
        (phoneme_time_table['start'] // .01) + (.5 * (phoneme_time_table['end'] - phoneme_time_table['start']) // .01)
    ).astype(int)
    # Get a pandas.Series of midpoints with phonemes as index.
    midpoints = pd.DataFrame(filterbank[mid_index], index=phoneme_time_table['phoneme'])
    return(midpoints)

def distance_matrix(sound_file, path_alignment_file) :
    #Gets the distance matrix of distances between midpoint vectors for pairs of phonemes
    
    midpoints = get_midpoints(sound_file, path_alignment_file)
    # Get the distance matrix
    distances = squareform(pdist(midpoints.values, metric ='euclidean'))
    distances = pd.DataFrame(distances, index=midpoints.index, columns=midpoints.index)
    print(distances)
    return(distances)


#for sound_file in os.listdir(path_sound_file):
#    if sound_file.endswith('.wav') :
#        distance_matrix(sound_file, path_alignment_file)


if __name__ == '__main__':  
    combine_time_tables('animal.wav', 'C:\\Users\\alain\\Desktop\\Cogmaster\\Cogmaster_S2\\Stage\\python_speech_features-master')
    distance_matrix('animal.wav', 'C:\\Users\\alain\\Desktop\\Cogmaster\\Cogmaster_S2\\Stage\\python_speech_features-master')
    # analyze_corpus(sys.argv[1], sys.argv[2])



#get_filterbank_chart('animal.wav', 'toy_data_alignement.txt', 'C:\\Users\\alain\\Desktop\\Cogmaster\\Cogmaster_S2\\Stage\\feature_fbank_chart.csv')
#'C:\\Users\\alain\\Desktop\\Cogmaster\\Cogmaster_S2\\Stage\\python_speech_features-master'