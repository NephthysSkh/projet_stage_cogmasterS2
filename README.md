# projet_stage_cogmasterS2


# Methodology 
The aim is to compare the distances between french phonemes and english phonemes. To do that, I'm going to compute the distance between filterbanks vectors corresponding to individual phonemes. \n

The program feature_charts.py allows us to summurize the information present in .wav files. The information is divided betwween a real time table comprising the start time of each phoneme, the end time of each phoneme and its tag, a window time table comprising the start time of each window, its end time and the corresponding filterbank vector. Both tables can be synthesized, in which case every window is associated to the corresponding phoneme. \n

Distances between phonemes are computed based on the distance between filterbank vector associated with the midpoint window of each phonemes and stored in a distance matrix. \n

