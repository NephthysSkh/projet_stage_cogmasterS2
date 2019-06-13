import numpy as np
import pandas as pd

from shennong.audio import Audio
from shennong.features.processor.filterbank import FilterbankProcessor
from shennong.features.processor.plp import PlpProcessor
from shennong.features.processor.rastaplp import RastaPlpProcessor
from shennong.features.processor.bottleneck import BottleneckProcessor



def get_features(sound_file, chosen_processor) :
    # computes the feature coefficients of a sound file

    #     :param sound_file : sound file in format .wav
    #     :type amount: .wav file
    #     :returns: feature coefficients per frame of 25ms every 10ms can be 'filterbank'
    #     'plp', 'rasteplp' or 'bottleneck'
    #     :rtype: a numpy array

    audio = Audio.load(sound_file)
    processors = {
        'filterbank': FilterbankProcessor(sample_rate=audio.sample_rate),
        'plp': PlpProcessor(sample_rate=audio.sample_rate),
        'rastaplp': RastaPlpProcessor(sample_rate=audio.sample_rate),
        'bottleneck': BottleneckProcessor(weights='BabelMulti')}

    features = chosen_processor.process(audio)
    features = pd.DataFrame(features)
    return(features)
