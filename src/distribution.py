#%%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm # to see progress bar (not necessary)
import pypianoroll as pr


pianoroll_dir = '../pianoroll'
files = list(Path(pianoroll_dir).iterdir())
pitches = {t: [0 for _ in range(128)] for t in range(17)}




def plotbytrack(t):
    global pitches
    trackPitches = pitches[t]
    total = sum(pitches)
    trackPitches = list(map(lambda pitch: pitch / total, trackPitches))
    plt.title(f'Track{t}')
    plt.plot(trackPitches)
    plt.show()
    

for file in tqdm(files, desc='Song: '):
    multitrack = pr.load(file).set_resolution(12)
    for t, track in enumerate(multitrack.tracks):
        nonzero = np.nonzero(track.pianoroll)
        for idx in nonzero[1]:
            pitches[t][idx] += 1

for t in range(17):
    plotbytrack(t)
        