from os import PathLike
from pathlib import Path
import pypianoroll as pr
import muspy
import music21
import pandas as pd
import numpy as np
import math
from typing import Tuple, Optional
from config import INSTCONFIG, GENRECONFIG


class PianorollData:
    def __init__(self, path: PathLike) -> None:
        self.path = path
        self.name = Path(path).stem

    def __str__(self) -> str:
        return f'Data of Pianoroll {name}'
    
    @classmethod
    def load(cls, path: PathLike) -> None:
        obj = cls(path)
        obj.multitrack = pr.load(path).set_resolution(12)
        obj.timesteplen = round(len(obj.multitrack.tempo + 1) / 2) + 1
        return obj
    
    def trim(self, ts: Optional[int] = None) -> 'PianorollData':
        if ts:
            for track in self.multitrack.tracks:
                track.trim(end=ts)
                self.timesteplen = min(ts, self.timesteplen)

    def getTrackLength(self) -> int:
        return len(self.multitrack.tracks)
    
    def getTimestepLength(self, idx) -> int:
        return self.multitrack.tracks[idx].pianoroll.shape[0]
    
    def getGenre(self, genres: pd.DataFrame) -> np.ndarray:
        label = genres.loc[self.name, 'Genre']
        genre = np.zeros(10)
        genre[label] = 1
        return genre
    
    def getKey(self) -> np.ndarray:
        mus = muspy.from_pypianoroll(self.multitrack)
        score = muspy.to_music21(mus)
        k: music21.key.Key = score.analyze('key')
        mode = 1 if k.mode == 'major' else 0
        tonic = k.tonic if mode == 1 else k.relative.tonic
        theta = 30 * (music21.pitch.Pitch(tonic).ps - 60)
        rad = math.pi * theta / 180
        key = np.array([math.sin(rad), math.cos(rad), mode])
        return key
    
    def getBPM(self) -> np.ndarray:
        return np.array([round(self.multitrack.tempo[0])])
    
    def getTrackInst(self, idx) -> int:
        track = self.multitrack.tracks[idx]
        program = track.program
        isDrum = track.is_drum
        if isDrum or (program in (list(range(8, 16)) + list(range(112, 120)))):
            return 0
        elif program in range(32, 40):
            return 1
        elif program in (list(range(16, 24)) + list(range(48, 56)) + list(range(88, 96))):
            return 3
        elif program in (list(range(96, 104)) + list(range(120, 128))):
            return 4
        else:
            return 2
    
    def getInstClassStep(self, idx, timestep) -> bool:
        track = self.multitrack.tracks[idx]
        return int(track.pianoroll[timestep].any())
    
    def getPianoroll(self, idx) -> np.ndarray:
        track = self.multitrack.tracks[idx]
        track_data = track.pianoroll.T
        track_data = np.pad(track_data, ((0, 0), (0, self.timesteplen - track_data.shape[1])), mode='constant')
        return track_data


class NpzData:
    def __init__(self, path: PathLike) -> None:
        self.path = path
        self.name = Path(path).stem
    
    def __str__(self) -> str:
        return f'Npz Data of {self.name}'
            
    def setTimestepNum(self, timesteps: int) -> None:
        self.timesteps = timesteps
        self.genre = np.zeros(len(GENRECONFIG))
        self.key = np.zeros(3)
        self.bpm = np.zeros(1)
        self.instclass = np.zeros((len(INSTCONFIG), timesteps))
        self.pianoroll = np.zeros((17, 128, timesteps))

    def setMetadata(self, genre, key, bpm):
        self.genre = genre
        self.key = key
        self.bpm = bpm
    
    @classmethod
    def load(cls, path: PathLike) -> 'NpzData':
        file = np.load(path, allow_pickle=True)
        obj = cls(path)
        obj.genre = file['genre']
        obj.key = file['key']
        obj.bpm = file['bpm']
        obj.instclass = file['inst_class']
        obj.pianoroll = file['pianoroll']
        return obj
    
    def save(self) -> None:
        np.savez_compressed(
            self.path,
            pianoroll=self.pianoroll,
            inst_class=self.instclass,
            genre=self.genre,
            key=self.key,
            bpm=self.bpm
        )
        



if __name__ == '__main__':
    name = '/mnt/c/Users/Marisa/Documents/skku/4-1/인지프/Project/lpd_17/pianoroll/88221692d3aa1479174c28bcbabbaa41.npz'

    ds = PianorollData.load(Path(name))
    print(type(ds.multitrack.tracks[-1]))
    print(ds)