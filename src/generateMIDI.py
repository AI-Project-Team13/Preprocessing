from config import TRACKCONFIG

import os
from pypianoroll import Multitrack, Track
import numpy as np
import muspy

SAVEDIR = '../MIDI'
TIMESTEPNUM = 10000
TRACKNUM = 17
VELOCITY = 100


def GenerateMIDI(output, metadata):
	pianorolls = output.transpose(1, 0, 2)  # shape = (TRACKNUM, TIMESTEPNUM, 128)
	bpm = metadata[3] * 150 + 50
	
	tracks = list()
	for idx in range(TRACKNUM):
		name, program = TRACKCONFIG[idx]
		is_drum = (idx == 0)
		pianoroll = pianorolls[idx]  # shape = (TIMESTEPNUM, 128)
		track = Track(name=name, program=program, is_drum=is_drum, pianoroll=pianoroll)
		tracks.append(track)
	
	tempo = [[bpm] for _ in range(TIMESTEPNUM)]
	multitrack = Multitrack(name='Sample', resolution=12, tempo=tempo, tracks=tracks)
	mus = muspy.from_pypianoroll(multitrack, VELOCITY)
	file_path = os.path.join(SAVEDIR, f'{multitrack.name}.mid')
	muspy.write_midi(file_path, mus)