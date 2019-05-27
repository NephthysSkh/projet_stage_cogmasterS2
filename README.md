# projet_stage_cogmasterS2


# Methodology 
The aim is to compare the distances between french phonemes and english phonemes. To do that, I'm going to compute the distance between filterbanks vectors corresponding to individual phonemes. \n

The program feature_charts.py allows us to summurize the information present in .wav files. The information is divided betwween a real time table comprising the start time of each phoneme, the end time of each phoneme and its tag, a window time table comprising the start time of each window, its end time and the corresponding filterbank vector. Both tables can be synthesized, in which case every window is associated to the corresponding phoneme. \n

Distances between phonemes are computed based on the distance between filterbank vector associated with the midpoint window of each phonemes and stored in a distance matrix. \n

# Script walk throught
1) I used the python_speech_features package to compute the filterbank vectors. It needs to be downloaded before hand in order for the script to work. In signal processing, a filter bank is an array of band-pass filters that separates the input signal into multiple components, each one carrying a single frequency sub-band of the original signal. A bank of receivers can be created by performing a sequence of FFTs on overlapping segments of the input data stream. A weighting function (in this case the Hamming window function) is applied to each segment to control the shape of the frequency responses of the filters. The wider the shape, the more often the FFTs have to be done to satisfy the Nyquist sampling criteria.

def get_filter_bank(sound_file) :
    # computes the filter bank coefficients of a sound file

    #     :param sound_file : sound file in format .wav
    #     :type amount: .wav file
    #     :returns: 26 filterbank coefficients per frame of 25ms every 10ms 
    #     :rtype: dataframe

    (rate,sig) = wav.read(sound_file)
    fbank_feat = logfbank(sig,rate)
    fbank_feat = pd.DataFrame(fbank_feat)
    fbank_feat.columns = ['filterbank_%s' % i for i in range(26)]
    return(fbank_feat)

2) This function allows us to retrive the alignment file and store it in a dataframe.

def parse_alignment_file(path_alignment_file):
    # creates a dataframe contraining for each phoneme its label, the file, the start time stamp
    # and the end time stamp. The entry is an aligmenent file in .txt. The key is the start time
            
    #     :param path_alignment_file : a path to the alignment file
    #     :type amount: path
    #     :returns: a dataframe with the following columns : "file name", "start", "end", "phoneme"
    #     :rtype: a dataframe


    #assert os.path.isfile(path_alignment_file) and path_alignment_file.endswith('.txt')
    with open(path_alignment_file, 'r') as file:
        df_alignment = pd.DataFrame([line.split() for line in file], columns=('file_name', 'start', 'end', 'phoneme'))
        df_alignment['start'] = df_alignment['start'].astype(float)
        df_alignment['end'] = df_alignment['end'].astype(float)
        return df_alignment

3) Both functions allows us to combined the alignement information for each phoneme with the associated filterbank coefficients per frame. This information is stored in a dataframe with the following columns "phoneme", "phoneme start time", "phoneme end time",  "frame_index", "frame_start_time", "frame_end_time", "F0"..."F26" with "F0"... "F26" the filterbank coefficients, and each line corresponding to a frame. This table is used later on to determine the midpoint filterbank vector for each phoneme and the distances between phoneme realizations interspeaker-wise.

def frame_time_table(nb_frames) : 
    # creates a dataframe contraining the start time stamp and the end time stamp of 
    # each frame given a number of frames.

    #     :param nb_frames : a number of frames
    #     :type amount: int
    #     :returns: a dataframe with the following columns : "start_frame", "end_frame"
    #     :rtype: a dataframe

    start = np.arange(0, nb_frames * .01, .01)
    frame_time = pd.DataFrame({'start_frame': start, 'end_frame': start + 0.025})
    return frame_time

def combine_time_tables(sound_file, alignment, save_path=None):
    # combines the alignment file phoneme information with the filterbanks coefficients for a given sound file

    #  (phoneme, start time, end time) and the frame time table
    # (frame index, start time, end time), table (phoneme, phoneme_start_time, phoneme_end_time, 
    # frame_index, wndow_start_time, frame_end_time)

    #     :param sound_file : a sound file in format .wav
    #     :param path_alignment_file : path of the corresponding alignment chart 
    #     :param save_path : a path
    #     :type amount sound_file : .wav file
    #     :type amount alignments : dataframe
    #     :type amount save_path : path
    #     :returns: a dataframe with the following columns : "phoneme", "phoneme start time", "phoneme end time", 
    #     "frame_index", "frame_start_time", "frame_end_time", "F0"..."F26" with "F0"... "F26" the filterbank coefficients.
    #     Each line corresponds to a frame.
    #     :rtype: dataframe

    # Import filterbanks.
    fbank = get_filter_bank(sound_file)
    frame_table = frame_time_table(len(fbank))
    # Set the index to match the start frame of each phoneme.
    phoneme_time_table = parse_alignment_file(alignment)
    phoneme_time_table.index = (phoneme_time_table['start'] // .01).astype(int)
    # Drop phones that last less than the span of a frame.
    phoneme_time_table = phoneme_time_table.iloc[np.where(~phoneme_time_table.index.duplicated(keep='last'))]
    # Join both tables.
    phoneme_time_table = phoneme_time_table[phoneme_time_table['file_name'] == sound_file]
    frame_table = frame_table.join(phoneme_time_table).fillna(method='ffill')
    # Add filterbank coefficients as a single column.
    table = pd.concat([frame_table, fbank], axis=1)
    if isinstance(save_path, str):
        table = table.to_csv(save_path, index = None, header=True)
    return(table)


4) 
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


6) 
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


7) 
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