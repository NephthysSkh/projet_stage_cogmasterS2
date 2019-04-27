import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from distances_features.feature_charts import combine_time_tables
from distances_features.feature_charts import get_filter_bank
from distances_features.feature_charts import parse_alignment_file

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
    return(distances)



if __name__ == '__main__':  
    distance_matrix(sys.argv[1], sys.argv[2])

