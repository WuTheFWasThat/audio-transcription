#!/usr/local/bin/python
import numpy

from pyknon.genmidi import Midi
from pyknon.music import Note, NoteSeq

from scipy.fftpack import fft
from scipy.io import wavfile

import random
import sh
import os

sh.mkdir('-p', 'cache')

# Notes in a midi file are in the range [0, 120).
kMidiMiddleC = 60
kMidiMaxNote = 120

# There are 10 families of melodic midi instruments, each with 8 variations.
# After midi instrument 80, we start on percussion.
kMidiInstrumentFamilies = 10
kMidiInstrumentFamilySize = 8

def memoize(fn):
    class memo(dict):
        def __call__(self, *args, **kwargs):
            return self[(args, frozenset(kwargs.items()))]
        def __missing__(self, key):
            (args, kwargs) = key
            self[key] = fn(*args, **dict(kwargs))
            return self[key]
    return memo()

'''
Creates a wav file of a piano playing `note` and writes it to `wav_filename`.
Returns a pair (sample_rate, data) from that wav file, where sample_rate is
the frequency in Hz of audio samples and data is a list of sample values.
'''
@memoize
def generateWavData(
    instrument, note, save_midi=False,
    save_wav=True
):
    midi_filename = 'cache/midi_i%d_n%d.mid' % (instrument, note)
    wav_filename = 'cache/wav_i%d_n%d.wav' % (instrument, note)
    cached_midi = os.path.exists(midi_filename)
    cached_wav = os.path.exists(wav_filename)
    if not cached_wav:
        if not cached_midi:
            midi = Midi(1, instrument=instrument, tempo=90)
            midi.seq_notes(NoteSeq([Note(note - kMidiMiddleC)]), track=0)
            midi.write(midi_filename)
        sh.timidity(midi_filename, '-Ow', '-o', wav_filename)

    (sample_rate, data) = wavfile.read(wav_filename)
    to_remove = []
    if not (cached_midi or save_midi):
        to_remove.append(midi_filename)
    if not (cached_wav or save_wav):
        to_remove.append(wav_filename)
    if len(to_remove):
        sh.rm(*to_remove)
    return (sample_rate, data.T[0])

'''
Plots the given sequence with matplotlib.
'''
def plotSequence(sequence):
    import matplotlib.pyplot as plot
    plot.plot(sequence)
    plot.show(block=True)

'''
Given samples from a wav file, a start location and window size, takes the fft
'''
def readSpectrum(samples, start, window):
    fragment = samples[start:start + window]
    return map(abs, fft(fragment)[:window/2])

'''
Generates a single training sample for our note-classifying network.
The result is a pair (frequencies, note), where frequencies is a
4410-dimensional feature vector (the frequencies of 0.1 seconds of a wav file,
because the default sample rate for wavs is 44100), and the output is a
120-dimensional one-hot encoding of the note.
'''
def sampleLabeledData(instrument=None, note=None, progress=None, nsecs = 0.1):
    # The distribution we're sampling from is parametrized by a note to play
    # and by progress, the time into the duration of the note from which we
    # sample the frequencies.
    if instrument is None:
        family = random.randint(0, kMidiInstrumentFamilies - 1)
        instrument = kMidiInstrumentFamilySize * family
    if note is None:
        note = random.randint(0, kMidiMaxNote - 1)
    if progress is None:
        progress = random.random()

    # Generate the actual training sample.
    sample_rate, samples = generateWavData(instrument, note)
    # only take from the first half second
    progress = progress * (sample_rate / len(samples)) * 0.5

    window = int(sample_rate * float(nsecs))

    start = int(progress * (len(samples) - window))
    assert start >= 0
    assert start <= len(samples) - window
    features = readSpectrum(samples, start, window)
    return {
        'spectrum': numpy.reshape(features, (1, -1)),
        'note': note,
        'progress': progress,
        'instrument': instrument,
    }

'''
Generates a single training sample for our note-classifying network.
The result is a pair (frequencies, note), where frequencies is a
4410-dimensional feature vector (the frequencies of 0.1 seconds of a wav file,
because the default sample rate for wavs is 44100), and the output is a
120-dimensional one-hot encoding of the note.
'''
def sampleLabeledSequentialData(instrument=None, note=None, nsecs=0.1, nsecs_overlap=0.05):
    # The distribution we're sampling from is parametrized by a note to play
    # and by progress, the time into the duration of the note from which we
    # sample the frequencies.
    if instrument is None:
        family = random.randint(0, kMidiInstrumentFamilies - 1)
        instrument = kMidiInstrumentFamilySize * family
    if note is None:
        note = random.randint(0, kMidiMaxNote - 1)

    # Generate the actual training sample.
    sample_rate, samples = generateWavData(instrument, note)

    window = int(sample_rate * float(nsecs))
    overlap = int(sample_rate * float(nsecs_overlap))
    sequential_features = []
    for start in range(0, len(samples) - window, overlap):
        features = readSpectrum(sample_rate, start, window)
        sequential_features.append(features)

    return {
        'spectrum': numpy.reshape(features, (1, -1)),
        'note': note,
        'progress': progress,
        'instrument': instrument,
    }

if __name__ == '__main__':
    print sampleLabeledData()
