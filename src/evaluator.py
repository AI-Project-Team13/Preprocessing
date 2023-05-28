import numpy as np
import pandas as pd
from os import PathLike
from typing import List, Tuple, Union
from pathlib import Path
from .config import TRACK2INST, DRUMDISTRIBUTION
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, DetCurveDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

ArrayLike = Union[np.ndarray[float], List[float]]

def calculate_score(y_true: ArrayLike, y_pred: ArrayLike) -> Tuple[float, float, float, float]:
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    return accuracy, precision, recall, f1

def calculate_distribution_similarity(dist1: ArrayLike, dist2: ArrayLike) -> Tuple[float, float, float]:
    # Jensen-Shannon Divergence (JSD)
    jsd_score = jensenshannon(dist1, dist2)
    # Earth Mover's Distance (EMD)
    emd_score = wasserstein_distance(dist1, dist2)
    # Cosine Similarity
    cosine_score = cosine_similarity([dist1], [dist2])[0][0]
    return jsd_score, emd_score, cosine_score


def plot_confusion_matrix(y_true, y_pred, save_dir: PathLike=None, save=False, show=False):
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap='Blues')
    if save: plt.savefig(save_dir / 'Confusion_Matrix.png')
    if show: plt.show()

def plot_precision_recall(y_true, y_pred, save_dir: PathLike=None, save=False, show=False):
    PrecisionRecallDisplay.from_predictions(y_true, y_pred)
    plt.legend().remove()
    if save: plt.savefig(save_dir / 'Precision_Recall.png')
    if show: plt.show()

def plot_roc_curve(y_true, y_pred, save_dir: PathLike=None, save=False, show=False):
    RocCurveDisplay.from_predictions(y_true, y_pred)
    plt.legend().remove()
    if save: plt.savefig(save_dir / 'Roc_Curve.png')
    if show: plt.show()

def plot_det_curve(y_true, y_pred, save_dir: PathLike=None, save=False, show=False):
    DetCurveDisplay.from_predictions(y_true, y_pred)
    plt.legend().remove()
    if save: plt.savefig(save_dir / 'Det_Curve.png')
    if show: plt.show()

def plot_distribution(dist, save_dir: PathLike=None, save=False, show=False):
    x = np.arange(1, len(dist)+1)
    plt.bar(x, dist)
    plt.xlabel('Timestep')
    plt.ylabel('Probability')
    plt.xticks(x)
    plt.legend().remove()
    if save: plt.savefig(save_dir / 'Distribution.png')
    if show: plt.show()

def safe_plot(func):
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except ValueError:
            pass
    return wrapper

class ScoreSheet:
    def __init__(self, im_score, ir_score, dp_score, name='ScoreSheet') -> None:
        self.im_score = im_score
        self.ir_score = ir_score
        self.dp_score = dp_score
        self.name = name
        self.sheet = self.make_sheet()
    
    def __str__(self) -> str:
        return str(self.sheet)
    
    def make_sheet(self) -> pd.DataFrame:
        sheet = pd.DataFrame({
            'Test': [
                'IM Test', 'IM Test', 'IM Test', 'IM Test',
                'IR Test', 'IR Test', 'IR Test', 'IR Test',
                'DP Test', 'DP Test', 'DP Test',
            ],
            'Metric': [
                'Accuracy', 'Precision', 'Recall', 'F1 Score',
                'Accuracy', 'Precision', 'Recall', 'F1 Score',
                'Jensen-Shannon Divergence', "Earth Mover's Distance", 'Cosine Similarity',
            ],
            'Score': [
                self.im_score[0], self.im_score[1], self.im_score[2], self.im_score[3],
                self.ir_score[0], self.ir_score[1], self.ir_score[2], self.ir_score[3],
                self.dp_score[0], self.dp_score[1], self.dp_score[2],
            ]
        })
        return sheet

    def save(self, save_dir: PathLike):
        self.sheet.to_csv(Path(save_dir) / f'{self.name}.csv', index=False)


class TestTaker:
    def __init__(self, inst_class: np.ndarray, output: np.ndarray, name='TestTaker') -> None:
        self.inst_class = inst_class.T  # shape (5, timestep) -> (timestep, 5)
        output = output.transpose(2, 0, 1)
        self.output: np.ndarray = np.any(output != 0, axis=2)  # shape (timesteps, tracks)
        self.name = name
        self.timesteps = self.inst_class.shape[0]
        self.sheet: ScoreSheet = None
    
    def grade_score(self, im_score: Tuple, ir_score: Tuple, dp_score: Tuple) -> None:
        self.sheet = ScoreSheet(im_score, ir_score, dp_score, self.name)

    def print_score(self):
        print(self.sheet)
    
    def save_score(self, save_dir: PathLike):
        self.sheet.save(save_dir)


class Evaluator:
    def __init__(self, root: PathLike) -> None:
        self.root = Path(root)
        
        # Result Plot Directory
        self.im_dir = Path(root) / 'IMTest'
        self.ir_dir = Path(root) / 'IRTest'
        self.dp_dir = Path(root) / 'DPTest'
        for test_path in [self.im_dir, self.ir_dir, self.dp_dir]:
            test_path.mkdir(parents=True, exist_ok=True)
        
        # Result
        self.im_true, self.im_pred, self.ir_true, self.ir_pred = [], [], [], []
        self.dp_pred = np.zeros(12)
        self.sheet: ScoreSheet = None
    
    def __call__(self, taker: TestTaker):
        im_score = self.IMTest(taker)
        ir_score = self.IRTest(taker)
        dp_score = self.DPTest(taker)
        taker.grade_score(im_score, ir_score, dp_score)

    def IMTest(self, taker: TestTaker) -> Tuple[float, float, float, float]:
        '''
        Input Match Test
        '''
        y_true = []
        y_pred = []
        for timestep in range(taker.timesteps):
            insts = taker.inst_class[timestep]
            pred_insts = np.zeros(5)
            tracks = taker.output[timestep]
            for idx, track in enumerate(tracks):
                inst = TRACK2INST[idx]
                ori = pred_insts[inst]
                pred_insts[inst] = max(ori, int(track))
            y_true.extend(insts)
            y_pred.extend(pred_insts)
        self.im_true.extend(y_true), self.im_pred.extend(y_pred)
        im_score = calculate_score(y_true, y_pred)
        return im_score
                
    
    def IRTest(self, taker: TestTaker) -> Tuple[float, float, float, float]:
        '''
        Input Response Test
        '''
        y_true = []
        y_pred = []
        for timestep in range(taker.timesteps):
            insts = taker.inst_class[timestep]
            tracks = taker.output[timestep]
            label = np.any(insts)
            predict = np.any(tracks)
            y_true.append(label)
            y_pred.append(predict)

        self.ir_true.extend(y_true), self.ir_pred.extend(y_pred)
        ir_score = calculate_score(y_true, y_pred)
        return ir_score
            
    
    def DPTest(self, taker: TestTaker) -> Tuple[float, float, float]:
        '''
        Drum Pattern Test
        '''
        drums: np.ndarray = taker.output.T[0]
        dp_pred = np.zeros(12)
        if drums.any():
            num_iterations = taker.timesteps // 12
            for iter in range(num_iterations):
                beat = drums[iter * 12:(iter + 1) * 12]
                true_indices = np.where(beat)[0]
                dp_pred[true_indices] += 1
                total_sum = np.sum(dp_pred)
            dp_dist = dp_pred / total_sum
            dp_score = calculate_distribution_similarity(DRUMDISTRIBUTION, dp_dist)
            self.dp_pred += dp_pred
        else:
            dp_score = (0, 0, 0)
        return dp_score
    
    def calculate_total_score(self):
        im_score = calculate_score(self.im_true, self.im_pred)
        ir_score = calculate_score(self.ir_true, self.ir_pred)
        if self.dp_pred.any():
            total_sum = np.sum(self.dp_pred)
            dp_dist = self.dp_pred / total_sum
            dp_score = calculate_distribution_similarity(DRUMDISTRIBUTION, dp_dist)
        else:
            dp_score = (0, 0, 0)
        self.sheet = ScoreSheet(im_score, ir_score, dp_score, 'Total')

    def print_total_score(self):
        print(self.sheet)
    
    def save_total_score(self):
        self.sheet.save(self.root)
    
    def plot_total(self, save=False, show=False):
        self.plot_IMTest(save, show)
        self.plot_IRTest(save, show)
        self.plot_DPTest(save, show)
    
    @safe_plot
    def plot_IMTest(self, save=False, show=False):
        plot_confusion_matrix(self.im_true, self.im_pred, self.im_dir, save, show)
        plot_precision_recall(self.im_true, self.im_pred, self.im_dir, save, show)
        plot_roc_curve(self.im_true, self.im_pred, self.im_dir, save, show)
        plot_det_curve(self.im_true, self.im_pred, self.im_dir, save, show)
    
    @safe_plot
    def plot_IRTest(self, save=False, show=False):
        plot_confusion_matrix(self.ir_true, self.ir_pred, self.ir_dir, save, show)
        plot_precision_recall(self.ir_true, self.ir_pred, self.ir_dir, save, show)
        plot_roc_curve(self.ir_true, self.ir_pred, self.ir_dir, save, show)
        plot_det_curve(self.ir_true, self.ir_pred, self.ir_dir, save, show)
    
    @safe_plot
    def plot_DPTest(self, save, show):
        if self.dp_pred.any():
            total_sum = np.sum(self.dp_pred)
            dp_dist = self.dp_pred / total_sum
        else:
            dp_dist = np.zeros_like(self.dp_pred)
        plot_distribution(dp_dist, self.dp_dir, save, show)