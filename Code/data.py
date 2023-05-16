from os import PathLike
from pathlib import Path
import pypianoroll as pr
import muspy
import pandas as pd
import numpy as np
from typing import Tuple
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
        obj.multitrack = pr.load(path)
        return obj
    
    def trim(self, ts: int) -> 'PianorollData':
        for track in self.multitrack.tracks:
            track.trim(end=ts)

    def getTrackNum(self) -> int:
        return len(self.multitrack.tracks)
    
    def getTimestepNum(self) -> int:
        return len(self.multitrack.tempo)
    
    def getGenre(self, genres: pd.DataFrame) -> np.ndarray:
        label = genres.loc[name, 'Genre']
        genre = np.zeros(10)
        genre[label] = 1
        return genre
    
    def getKey(self) -> np.ndarray:
        mus = muspy.from_pypianoroll(self.multitrack)
        score = muspy.to_music21(mus)
        key = score.analyze('key')
        mode = 1 if key.mode == 'major' else -1
        return np.array([(key.sharps + 1) * mode])
    
    def getBPM(self) -> np.ndarray:
        return np.array([round(self.multitrack.tempo[0])])
    
    def getMetaData(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.getGenre, self.getKey, self.getBPM
    
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
        return track.pianoroll.T


class NpzData:
    def __init__(self, path: PathLike) -> None:
        self.path = path
        self.name = Path(path).stem
    
    def __str__(self) -> str:
        return f'Npz Data of {self.name}'
            
    def setTimestepNum(self, timesteps: int) -> None:
        self.timesteps = timesteps
        self.genre = np.zeros(len(GENRECONFIG))
        self.key = np.zeros(1)
        self.bpm = np.zeros(1)
        self.instclass = np.zeros((len(INSTCONFIG), timesteps))
        self.pianoroll = np.zeros((17, 128, timesteps))

    def setMetadata(self, genre, key, bpm):
        self.genre = genre
        self.key = key
        self.bpm = bpm
    
    @classmethod
    def load(cls, path: PathLike) -> 'NpzData':
        file = np.load(path)
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