import argparse
import operator
import os
import random

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from distances import get_midpoints

random.seed(30)

def parse_alignment_file(path_alignment_file):
    # creates a dataframe contraining for each phoneme its label, the file, the start time stamp
    # and the end time stamp. The entry is an aligmenent file in .txt. The key is the start time

    #      :param path_alignment_file : alignment file
    #      :type amount: str
    #      :returns: a dataframe with the following columns : "file name", "start", "end", "phoneme"
    #      :rtype: dataframe

    with open(path_alignment_file, 'r') as file:
        df_alignment = pd.DataFrame([line.split() for line in file])
    df_alignment.columns = ('file_name', 'start', 'end', 'phoneme')
    df_alignment['start'] = df_alignment['start'].astype(float)
    df_alignment['end'] = df_alignment['end'].astype(float)
    return df_alignment

def normalize_features(dataframe) :
    # normalizes features dataframe, column-wise

    #      :param dataframe : feature dataframe for a given speaker (=sound file)
    #      :type amount dataframe : panda df
    #      :returns: normalized coefficients
    #      :rtype: dataframe

    return (dataframe - dataframe.mean()) / dataframe.std()

def extract_corpus_features(wav_folder_path, alignment, save_path):
    # extract, normalize and save midpoint features from files in a folder

    #      :param wav_folder_path : sound folder path
    #      :type amount dataframe : str
    #      :returns: csv containing normalized coefficients
    #      :rtype: dataframe

    #ENELVER LES .WAV DES NOMS DE FICHIERS

    # list the files to process
    alignment_file_list = alignment['file_name'].unique()
    file_list = [name for name in os.listdir(wav_folder_path)]
    file_list = [os.path.splitext(file_name)[0] for file_name in file_list]
    file_list = [name for name in file_list if name in alignment_file_list] 


    # remove any pre-existing output file
    if os.path.isfile(save_path):
        os.remove(save_path)

    # iteratively compute, normalize and save file-wise features
    # note: this makes sense because each file is a single speaker
    for i, sound_file in enumerate(file_list):
        midpoints = get_midpoints(
            os.path.join(wav_folder_path, sound_file + '.wav'),
            alignment[alignment['file_name'] == sound_file]
        )
        # FIXME: optionally, we may want to only retain the mean
        #        observation for each phoneme (for each speaker)
        #        to do so, uncomment the following line:
        # midpoints = midpoints.groupby(midpoints.index).mean()
        data = normalize_features(midpoints)
        data.to_csv(save_path, mode='a', header=(i == 0))


def save_normalized_data(wav_folder_path_1, wav_folder_path_2, path_alignment_file_1, path_alignment_file_2, save_path_norm_data_1, save_path_norm_data_2):
    # saves normalized data

    #      :param wav_folder_path_1 : sound folder containing corpus 1
    #      :param wav_folder_path_2 : sound folder containing corpus 2
    #      :param path_alignment_file : alignment file
    #      :param save_path_norm_data_1 : csv file name to save corpus 1
    #      :param save_path_norm_data_2 : csv file name to save corpus 2
    #      :type amount wav_folder_1 : str
    #      :type amount wav_folder_2 : str
    #      :type amount path_alignment_file : str
    #      :type amount save_path_norm_data_1 : str
    #      :param save_path_norm_data_2 : str
    #      :returns: csv files
    #      :rtype: csv files

    # parse the alignment file
    alignment_1 = parse_alignment_file(path_alignment_file_1)
    # process the first corpus
    extract_corpus_features(
        wav_folder_path_1, alignment_1, save_path_norm_data_1
    )
    # process the second corpus
    alignment_2 = parse_alignment_file(path_alignment_file_2)
    extract_corpus_features(
        wav_folder_path_2, alignment_2, save_path_norm_data_2
    )


def select_data(data_corpus_1, data_corpus_2, selected_data_corpus_1, selected_data_corpus_2, nb_of_rows) :
    # Ramdomly samples data from normalized feature csv

    #      :param data_corpus_2 : csv containing corpus 1 normalized features
    #      :param data_corpus_1 : csv containing corpus 2 normalized features
    #      :param selected_data_corpus_1 : file name of csv where selected data of corpus 1 is saved
    #      :param selected_data_corpus_2 : file name of csv where selected data of corpus 2 is saved
    #      :param save_path_norm_data_2 : csv file name to save corpus 2
    #      :param nb_of_rows : number of rows to select
    #      :type amount data_corpus_1 : str
    #      :type amount data_corpus_2 : str
    #      :type amount selected_data_corpus_1 : str
    #      :type amount selected_data_corpus_2 : str
    #      :type amount save_path_norm_data_2 : str
    #      :type amount nb_of_rows : int 
    #      :returns: csv files
    #      :rtype: csv files

    df_1 = pd.read_csv(data_corpus_1, index_col=0)
    df_2 = pd.read_csv(data_corpus_2, index_col=0)

    df_1.sample(n = nb_of_rows)
    df_1.to_csv(selected_data_corpus_1, header=True)

    df_2.sample(n = nb_of_rows)
    df_2.to_csv(selected_data_corpus_2, header=True)

    return(df_1, df_2)


def compute_distances_matrix(selected_data_corpus_1, selected_data_corpus_2, n_repr=None):
        # compute distance matrix between corpus 1 and corpus 2 features

        #   :param selected_data_corpus_1 : csv file containing sampled normalized data of corpus 1
        #   :param selected_data_corpus_2 : csv file containing sampled normalized data of corpus 2
        #   :param save_path_norm_data_2 : csv file name to save corpus 2
        #   :param n_repr : number of clusters in get_distance_moments function (optional)
        #   :type amount selected_data_corpus_1 : str
        #   :type amount selected_data_corpus_2 : str
        #   :type amount save_path_norm_data_2 : str
        #   :type amount n_repr : int
        #   :returns: mean distance matrix
        #   :rtype:

    df_1 = pd.read_csv(selected_data_corpus_1, index_col=0)
    df_2 = pd.read_csv(selected_data_corpus_2, index_col=0)
    #FIXME: you can get rid of the stdev part if you want
            #then, adjust the get_distance_moments function
    #        and simply return metrics (get rid of follow-up code lines)
    # compute mean and standard deviation distances between phonemes.
    metrics = pd.DataFrame({
        ph_1: {
            ph_2: get_distance_moments(feats_1, feats_2, n_repr)
            for ph_2, feats_2 in df_2.groupby(df_2.index.sort_values())
        }
        for ph_1, feats_1 in df_1.groupby(df_1.index.sort_values())
    })
    # separate the previous into two matrices and return them
    mean_distances = metrics.applymap(operator.itemgetter(0))
    stdev_distances = metrics.applymap(operator.itemgetter(1))
    return mean_distances, stdev_distances


def get_distance_moments(feats_1, feats_2, n_repr=0):
        # Compute mean and stdev of distances between features collections
        #  optionally use k-means to build subsets to compare, so as to reduce computational costs (only if n_repr > 0)

        #  :param selected_data_corpus_1 : csv file containing sampled normalized data of corpus 1
        #  :param selected_data_corpus_2 : csv file containing sampled normalized data of corpus 2
        #  :param save_path_norm_data_2 : csv file name to save corpus 2
        #  :param n_repr : number of clusters in get_distance_moments function (optional)
        #  :type amount selected_data_corpus_1 : str
        #  :type amount selected_data_corpus_2 : str
        #  :type amount save_path_norm_data_2 : str
        #  :type amount n_repr : int
        #  :returns: mean distance matrix
        #  :rtype:


    # FIXME: may be overkill. It's possible to get rid of the stdev part
    # optionally cluster each features set and use the centers as feats
    if n_repr :
        if n_repr < len(feats_1):
            feats_1 = KMeans(n_clusters=n_repr).fit(feats_1).cluster_centers_
        if n_repr < len(feats_2):
            feats_2 = KMeans(n_clusters=n_repr).fit(feats_2).cluster_centers_
    # compute the pairwise distances between features collections
    distances = pairwise_distances(feats_1, feats_2, metric='euclidean')
    # return the mean and standard deviation of those distances
    return distances.mean(), distances.std()


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(
        description='Compute distance matrix between features extracted from corpus 1 and corpus 2')
    #parser.add_argument('wav_folder_path_1', metavar='path_1', type=str,
                        #help='path to wav folder containing the data of corpus 1')
    #parser.add_argument('wav_folder_path_2', metavar='path_2', type=str,
                        #help='path to wav folder containing the data of corpus 2')
    #parser.add_argument('path_alignment_file_1', metavar='alignment_file', type=str,
                        #help='path to the alignment file 1')
    #parser.add_argument('path_alignment_file_2', metavar='alignment_file', type=str,
                        #help='path to the alignment file 2')
    #parser.add_argument('save_path_norm_data_1', metavar='save_norm_1', type=str,
                        #help='path to csv file where normalized feature chart of corpus 1 is saved')
    #parser.add_argument('save_path_norm_data_2', metavar='save_norm_2', type=str,
                        #help='path to csv file where normalized feature chart of corpus 2 is saved')
    #parser.add_argument('data_corpus_1', metavar='data_corpus_1', type=str,
                        #help='csv file of normalized features of corpus 1 phones')
    #parser.add_argument('data_corpus_2', metavar='data_corpus_2', type=str,
                        #help='csv file of normalized features of corpus 2 phones')
    parser.add_argument('selected_data_corpus_1', metavar='selected_data_corpus_1', type=str,
                        help='save path for randomly selected lines of the data_corpus_1 csv files')
    parser.add_argument('selected_data_corpus_2', metavar='selected_data_corpus_2', type=str,
                        help='save path for randomly selected lines of the data_corpus_2 csv files')
    #parser.add_argument('nb_of_rows', metavar='nb_of_rows', type=int,
                        #help='number of selected rows')
    parser.add_argument('n_repr', default=0, metavar='n_repr', type=int,
                        help='optional maximum number of representatives of a phoneme to use')


    args = parser.parse_args()

    #save_normalized_data(
    #     args.wav_folder_path_1, args.wav_folder_path_2,
    #     args.path_alignment_file_1, args.path_alignment_file_2,
    #     args.save_path_norm_data_1, args.save_path_norm_data_2
    # )

    #select_data(
    #    args.data_corpus_1, args.data_corpus_2, 
    #    args.selected_data_corpus_1, args.selected_data_corpus_2, 
    #    args.nb_of_rows)

    distances_matrix, stdev_distances_matrix = compute_distances_matrix(
        args.selected_data_corpus_1, args.selected_data_corpus_2, args.n_repr
    )

    distances_matrix.to_csv('distances_matrix.csv')
    stdev_distances_matrix.to_csv('distances_matrix_stdev.csv')
    #print(distances_matrix)

    plt.figure()
    sns.heatmap(distances_matrix, cmap='Blues')
    plt.xlabel("phones corpus 2 (English)")
    plt.ylabel("phones corpus 1 (French)")
    plt.savefig('/scratch2/elannelongue/distance_matrix.pdf')
    #plt.show()

    #means_per_speaker = calculate_mean_per_speaker('toy_data', 'toy_data/toy_data_alignment.txt', 'features_data_1.csv', 'norm_data_1.csv', 'mean_data_1.csv')
    #arguments : toy_data_1 toy_data_2 alignment/toy_data_alignment_1.txt alignment/toy_data_alignment_2.txt norm_data_1.csv norm_data_2.csv


# python analyze_data.py /scratch1/projects/challenge2017/final_datasets/datasets/train/french/ /scratch1/projects/challenge2017/final_datasets/datasets/train/english/ /scratch1/projects/challenge2017/final_datasets/alignment/french/alignment_phone.txt /scratch1/projects/challenge2017/final_datasets/alignment/english/alignment_phone.txt /scratch2/elannelongue/norm_data_french.csv /scratch2/elannelongue/norm_data_english.csv
# python analyze_data.py /scratch2/elannelongue/norm_data_french.csv /scratch2/elannelongue/norm_data_english.csv /scratch2/elannelongue/selected_data_french_100lines.csv /scratch2/elannelongue/selected_data_english_100lines.csv 100
# python analyze_data.py /scratch2/elannelongue/selected_data_french_100lines.csv /scratch2/elannelongue/selected_data_english_100lines.csv