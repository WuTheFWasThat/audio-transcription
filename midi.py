#!/usr/local/bin/python
import numpy

from pyknon.genmidi import Midi
from pyknon.music import Note, NoteSeq

from scipy.fftpack import fft
from scipy.io import wavfile

import random
import sh

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
def generateWavData(instrument, note):
    midi = Midi(1, instrument=instrument, tempo=90)
    midi.seq_notes(NoteSeq([Note(note - kMidiMiddleC)]), track=0)
    midi_filename = 'temp.mid'
    wav_filename = 'temp.wav'
    midi.write(midi_filename)
    sh.timidity(midi_filename, '-Ow', '-o', wav_filename)
    (sample_rate, data) = wavfile.read(wav_filename)
    sh.rm(midi_filename, wav_filename)
    return (sample_rate, data.T[0])

'''
Plots the given sequence with matplotlib.
'''
def plotSequence(sequence):
    import matplotlib.pyplot as plot
    plot.plot(sequence)
    plot.show(block=True)

'''
Reads a wav file and returns the spectrum of a 0.1-second interval that is
`progress` of the way through the file (e.g., if `progress` were 0.5, halfway
through the file.)
'''
def readSpectrum(wav_data, progress):
    (sample_rate, samples) = wav_data
    window = sample_rate / 10
    # Sample from between the first 0.1s and 0.9s of the note.
    start = (0.1 + 0.8 * progress) * min(len(samples), sample_rate) - window/2
    start = min(max(int(start), 0), len(samples) - window)
    fragment = samples[start:start + window]
    return map(abs, fft(fragment)[:window/2])

'''
Generates a single training sample for our note-classifying network.
The result is a pair (frequencies, note), where frequencies is a
4410-dimensional feature vector (the frequencies of 0.1 seconds of a wav file,
because the default sample rate for wavs is 44100), and the output is a
120-dimensional one-hot encoding of the note.
'''
def sampleLabeledData(instrument=None, note=None, progress=None):
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
    features = readSpectrum(generateWavData(instrument, note), progress)
    return (numpy.reshape(features, (1, -1)), note)

if __name__ == '__main__':
    print sampleLabeledData()
