import pandas as pd
from compute_features import get_filter_bank


def get_midpoints(sound_file, alignment):
    # Gets the midpoint vector for each phoneme

    #     :param sound_file : a sound file in format .wav
    #     :param path_alignment_file : path to the corresponding aligment file
    #     :type amount sound_file : .wav sound file
    #     :type amount path_alignment_file : path
    #     :returns: a dataframe with the following columns : "phoneme", "start", "end"
    #     :rtype: a dataframe

    filterbank = get_filter_bank(sound_file+'.wav')
    # Get the index of the midpoint frame for each phoneme as a pd.Series.
    mid_index = (
        (alignment['start'] // .01) + (.5 * (alignment['end'] - alignment['start']) // .01)
    ).astype(int)
    # Check that the targetted indices exist - trim some off if necessary.
    mid_index = mid_index[mid_index.isin(filterbank.index)]
    # Get a pandas.Series of midpoints with phonemes as index.
    midpoints = filterbank.loc[mid_index]
    midpoints.index = alignment['phoneme']
    return(midpoints)
