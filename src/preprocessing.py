import pypianoroll as pr
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import trange, tqdm # to see progress bar (not necessary)
from typing import Tuple, List # type anotation (not necessary)
from data import PianorollData, NpzData

pianoroll_dir = '../pianoroll'
dataset_dir = '../Dataset'
genreFile = pd.read_csv('../Metadata/genre.csv', index_col=1)
files = list(Path(pianoroll_dir).iterdir())
TIMESTEPNUM = None

# Iterate all files
for count, file in enumerate(tqdm(files, desc='Song: ')):
  dsfile = Path(dataset_dir) / f"np_{file.with_suffix('.npz').name}"
  if dsfile.exists():
    continue

  # Load the pianoroll data file
  prdata = PianorollData.load(file)

  # Init the npz data file
  npzdata = NpzData(dsfile)

  # Trim the pianoroll data
  prdata.trim(TIMESTEPNUM)

  # Init numpy array as zeros (which needs fixed length of timestep)
  npzdata.setTimestepNum(prdata.getTimestepNum())

  # get metadata of the pianoroll data
  key, genre, bpm = prdata.getMetaData()

  # set metadata of npz data
  npzdata.setMetaData(key, genre, bpm)

  # Iterate over all the tracks in the multitrack file
  for idx in enumerate(range(prdata.getTrackNum())):
    # Get some information about the current track
    instType = prdata.getTrackInst(idx)
    npzdata.pianoroll[idx] = prdata.getPianoroll(idx)

    # Iterate over all the timesteps in the pianoroll, show progress bar
    for timestep in tqdm(range(piano_roll.shape[0]), desc=f'Song{count} - Track{idx}: '):
      origin = npzdata.inst_class[instType][timestep]
      new = prdata.getInstClassStep(idx, timestep)
      npzdata.inst_class[instType][timestep] = max(origin, new)
          
  # Save the combined tracks data
  npzdata.save()