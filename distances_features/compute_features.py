import scipy.io.wavfile as wav
import pandas as pd
from python_speech_features import mfcc
from python_speech_features import logfbank


def get_filter_bank(sound_file) :
    # computes the filter bank coefficients of a sound file

    #     :param sound_file : sound file in format .wav
    #     :type amount: .wav file
    #     :returns: 26 filterbank coefficients per frame of 25ms every 10ms
    #     :rtype: a numpy array

    (rate,sig) = wav.read(sound_file)
    fbank_feat = logfbank(sig,rate)
    fbank_feat = pd.DataFrame(fbank_feat)
    fbank_feat.columns = ['filterbank_%s' % i for i in range(26)]
    return(fbank_feat)
