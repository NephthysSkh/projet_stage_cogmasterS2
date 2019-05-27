import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from feature_charts import *

def get_midpoints(sound_file, path_alignment_file):
    # Gets the midpoint vector for each phoneme

    #     :param sound_file : a sound file in format .wav
    #     :param path_alignment_file : path to the corresponding aligment file
    #     :type amount sound_file : .wav sound file
    #     :type amount path_alignment_file : path
    #     :returns: a dataframe with the following columns : "phoneme", "start", "end"
    #     :rtype: a dataframe

    phoneme_time_table = parse_alignment_file(path_alignment_file)
    filterbank = get_filter_bank(sound_file)
    # Get the index of the midpoint frame for each phoneme as a pd.Series.
    mid_index = (
        (phoneme_time_table['start'] // .01) + (.5 * (phoneme_time_table['end'] - phoneme_time_table['start']) // .01)
    ).astype(int)
    # Get a pandas.Series of midpoints with phonemes as index.
    midpoints = pd.DataFrame(filterbank[mid_index], index=phoneme_time_table['phoneme'])
    return(midpoints)

def distance_matrix_pairwise(sound_file, path_alignment_file) :
    # Gets the distance matrix of distances between midpoint vectors for pairs of phonemes

    #     :param sound_file : a sound file in format .wav
    #     :param path_alignment_file : path to the corresponding aligment file
    #     :type amount sound_file : .wav sound file
    #     :type amount path_alignment_file : path
    #     :returns: the pairwise distance matrix between within speaker
    #     :rtype: a dataframe
    
    midpoints = get_midpoints(sound_file, path_alignment_file)
    # Get the distance matrix
    distances = squareform(pdist(midpoints.values, metric ='euclidean'))
    distances = pd.DataFrame(distances, index=midpoints.index, columns=midpoints.index)
    return(distances)


def distance_matrix_interspeaker(sound_file_1, sound_file_2, path_alignment_file) :
    # Gets the distance matrix of distances between midpoint vectors for pairs of phonemes

    #     :param sound_file : a sound file in format .wav
    #     :param path_alignment_file : path to the corresponding aligment file
    #     :type amount sound_file : .wav sound file
    #     :type amount path_alignment_file : path
    #     :returns: the interspeaker phonemes distance matrix
    #     :rtype: a dataframe
    
    midpoints_1 = get_midpoints(sound_file_1, path_alignment_file)
    midpoints_2 = get_midpoints(sound_file_2, path_alignment_file)
    # Get the distance matrix
    distances = squareform(cdist(midpoints_1.values, midpoints_2.values, metric ='euclidean'))
    distances = pd.DataFrame(distances, index=midpoints_1.index, columns=midpoints_2.index)
    return(distances)
