import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist, squareform

from feature_charts import *

def get_midpoints(sound_file, alignment):
    # Gets the midpoint vector for each phoneme

    #     :param sound_file : a sound file in format .wav
    #     :param path_alignment_file : path to the corresponding aligment file
    #     :type amount sound_file : .wav sound file
    #     :type amount path_alignment_file : path
    #     :returns: a dataframe with the following columns : "phoneme", "start", "end"
    #     :rtype: a dataframe

    filterbank = get_filter_bank(sound_file)
    # Get the index of the midpoint frame for each phoneme as a pd.Series.
    mid_index = (
        (alignment['start'] // .01) + (.5 * (alignment['end'] - alignment['start']) // .01)
    ).astype(int)
    # Check that the targetted indices exist - trim some off if necessary.
    mid_index = mid_index[mid_index.isin(filterbank.index)]
    # Get a pandas.Series of midpoints with phonemes as index.
    midpoints = pd.DataFrame(filterbank.loc[mid_index], index=alignment['phoneme'])
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
