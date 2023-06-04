import numpy as np
from numpy.typing import ArrayLike
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


def calculate_score(y_true: ArrayLike, y_pred: ArrayLike, average='macro') -> Tuple[float, float, float, float]:
    """
    Calculate accuracy, precision, recall, and F1 score.

    Args:
        y_true: Array of true labels.
        y_pred: Array of predicted labels.
        average: determines the type of averaging performed on the data

    Returns:
        Tuple of accuracy, precision, recall, and F1 score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1, average=average)
    recall = recall_score(y_true, y_pred, zero_division=1, average=average)
    f1 = f1_score(y_true, y_pred, zero_division=1, average=average)
    return accuracy, precision, recall, f1

def calculate_distribution_similarity(dist1: ArrayLike, dist2: ArrayLike) -> Tuple[float, float, float]:
    """
    Calculate distribution similarity using Jensen-Shannon Divergence (JSD), Earth Mover's Distance (EMD),
    and Cosine Similarity.

    Args:
        dist1: Array representing the first distribution.
        dist2: Array representing the second distribution.

    Returns:
        Tuple of JSD score, EMD score, and cosine similarity score.
    """
    jsd_score = jensenshannon(dist1, dist2)
    emd_score = wasserstein_distance(dist1, dist2)
    cosine_score = cosine_similarity([dist1], [dist2])[0][0]
    return jsd_score, emd_score, cosine_score

def plot_confusion_matrix(y_true, y_pred, save_dir: PathLike = None, save=False, show=False):
    """
    Plot and display the confusion matrix.

    Args:
        y_true: Array of true labels.
        y_pred: Array of predicted labels.
        save_dir: Directory to save the plot (optional).
        save: Whether to save the plot (optional).
        show: Whether to display the plot (optional).
    """
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap='Blues')
    if save: plt.savefig(Path(save_dir) / 'Confusion_Matrix.png')
    if show: plt.show()

def plot_precision_recall(y_true, y_pred, save_dir: PathLike = None, save=False, show=False):
    """
    Plot and display the precision-recall curve.

    Args:
        y_true: Array of true labels.
        y_pred: Array of predicted labels.
        save_dir: Directory to save the plot (optional).
        save: Whether to save the plot (optional).
        show: Whether to display the plot (optional).
    """
    PrecisionRecallDisplay.from_predictions(y_true, y_pred)
    plt.legend().remove()
    if save: plt.savefig(Path(save_dir) / 'Precision_Recall.png')
    if show: plt.show()

def plot_roc_curve(y_true, y_pred, save_dir: PathLike = None, save=False, show=False):
    """
    Plot and display the ROC curve.

    Args:
        y_true: Array of true labels.
        y_pred: Array of predicted labels.
        save_dir: Directory to save the plot (optional).
        save: Whether to save the plot (optional).
        show: Whether to display the plot (optional).
    """
    RocCurveDisplay.from_predictions(y_true, y_pred)
    plt.legend().remove()
    if save: plt.savefig(Path(save_dir) / 'Roc_Curve.png')
    if show: plt.show()

def plot_det_curve(y_true, y_pred, save_dir: PathLike = None, save=False, show=False):
    """
    Plot and display the DET curve.

    Args:
        y_true: Array of true labels.
        y_pred: Array of predicted labels.
        save_dir: Directory to save the plot (optional).
        save: Whether to save the plot (optional).
        show: Whether to display the plot (optional).
    """
    DetCurveDisplay.from_predictions(y_true, y_pred)
    plt.legend().remove()
    if save: plt.savefig(Path(save_dir) / 'Det_Curve.png')
    if show: plt.show()

def plot_distribution(dist, save_dir: PathLike = None, save=False, show=False):
    """
    Plot and display the distribution.

    Args:
        dist: Array representing the distribution.
        save_dir: Directory to save the plot (optional).
        save: Whether to save the plot (optional).
        show: Whether to display the plot (optional).
    """
    x = np.arange(1, len(dist) + 1)
    plt.bar(x, dist)
    plt.xlabel('Timestep')
    plt.ylabel('Probability')
    plt.xticks(x)
    plt.legend().remove()
    if save: plt.savefig(Path(save_dir) / 'Distribution.png')
    if show: plt.show()

def safe_plot(func):
    """
    Decorator to handle exceptions when plotting.

    Args:
        func: Function to be decorated.

    Returns:
        Decorated function.
    """
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
        """
        Create a score sheet DataFrame.

        Returns:
            DataFrame representing the score sheet.
        """
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
        """
        Save the score sheet as a CSV file.

        Args:
            save_dir: Directory to save the score sheet.
        """
        self.sheet.to_csv(Path(save_dir) / f'{self.name}.csv', index=False)


class TestTaker:
    def __init__(self, inst_class: np.ndarray, output: np.ndarray, name='TestTaker') -> None:
        """
        Initializes a TestTaker object.

        Args:
            inst_class: Numpy array of shape (5, #timestep) representing the class of each instrument at each timestep.
            output: Numpy array of shape (17, 128, #timestep) representing the output of the model.
            name (optional): Name of the TestTaker. Defaults to 'TestTaker'.
        """
        self.inst_class = inst_class.T  # Transpose inst_class to shape (#timestep, 5)
        output = output.transpose(2, 0, 1)  # Transpose output to shape (#timestep, 17, 128)
        
        # Determine if each track is active at each timestep
        self.output: np.ndarray = np.any(output != 0, axis=2)  # shape (#timestep, 17)
        
        self.name = name
        self.timesteps = self.inst_class.shape[0]
        self.sheet: ScoreSheet = None
    
    def grade_score(self, im_score: Tuple, ir_score: Tuple, dp_score: Tuple) -> None:
        """
        Grades the scores of the test.

        Args:
            im_score: Tuple of scores for Input Match Test.
            ir_score: Tuple of scores for Input Response Test.
            dp_score: Tuple of scores for Drum Pattern Test.
        """
        self.sheet = ScoreSheet(im_score, ir_score, dp_score, self.name)
    
    def get_score(self):
        """
        Get the scores of the test.
        """
        return self.sheet.im_score, self.sheet.ir_score, self.sheet.dp_score

    def print_score(self):
        """
        Prints the score sheet.
        """
        print(self.sheet)
    
    def save_score(self, save_dir: PathLike):
        """
        Saves the score sheet as a CSV file.

        Args:
            save_dir: Directory path to save the CSV file.
        """
        self.sheet.save(save_dir)


class Evaluator:
    def __init__(self, root: PathLike, average='macro') -> None:
        """
        Initializes an Evaluator object.

        Args:
            root: Root directory path.
            average: determines the type of averaging performed on the data when calculate score
        """
        self.root = Path(root)
        self.average = average
        
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
        """
        Evaluates a TestTaker object.

        Args:
            taker: The TestTaker object to evaluate.
        """
        im_score = self.IMTest(taker)
        ir_score = self.IRTest(taker)
        dp_score = self.DPTest(taker)
        taker.grade_score(im_score, ir_score, dp_score)

    def IMTest(self, taker: TestTaker) -> Tuple[float, float, float, float]:
        '''
        Instrument Match Test

        Args:
            taker: The TestTaker object to evaluate.

        Returns:
            Tuple of IMTest scores: (accuracy, precision, recall, f1_score).
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
            if self.average == 'binary':
                y_true.extend(insts)
                y_pred.extend(pred_insts)
            else:
                y_true.append(insts)
                y_pred.append(pred_insts)
        self.im_true.extend(y_true), self.im_pred.extend(y_pred)
        im_score = calculate_score(y_true, y_pred, self.average)
        return im_score
                
    
    def IRTest(self, taker: TestTaker) -> Tuple[float, float, float, float]:
        '''
        Input Response Test

        Args:
            taker: The TestTaker object to evaluate.

        Returns:
            Tuple of IMTest scores: (accuracy, precision, recall, f1_score).
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

        Args:
            taker: The TestTaker object to evaluate.

        Returns:
            Tuple of DPTest scores: (JSD score, EMD score, cosine similarity score).
        '''
        drums: np.ndarray = taker.output.T[0]
        y_pred = np.zeros(12)
        if drums.any():
            num_iterations = taker.timesteps // 12
            for iter in range(num_iterations):
                beat = drums[iter * 12:(iter + 1) * 12]
                true_indices = np.where(beat)[0]
                y_pred[true_indices] += 1
            total_sum = np.sum(y_pred)
            dp_dist = y_pred / total_sum
            dp_score = calculate_distribution_similarity(DRUMDISTRIBUTION, dp_dist)
            self.dp_pred += y_pred
        else:
            dp_score = (0, 0, 0)
        return dp_score
    
    def calculate_total_score(self):
        """
        Calculate the total score combining scores from different tests.

        Computes the scores for input match test (IMTest), input response test (IRTest), and drum pattern test (DPTest).
        If drum pattern predictions are available, it calculates the drum pattern score using distribution similarity.
        Creates a ScoreSheet object with the computed scores and assigns it to the `sheet` attribute.
        """
        im_score = calculate_score(self.im_true, self.im_pred, self.average)
        ir_score = calculate_score(self.ir_true, self.ir_pred)
        if self.dp_pred.any():
            total_sum = np.sum(self.dp_pred)
            dp_dist = self.dp_pred / total_sum
            dp_score = calculate_distribution_similarity(DRUMDISTRIBUTION, dp_dist)
        else:
            dp_score = (0, 0, 0)
        self.sheet = ScoreSheet(im_score, ir_score, dp_score, 'Total')
    
    def get_total_score(self):
        """
        Get the total scores of the tests.
        """
        try:
            return self.sheet.im_score, self.sheet.ir_score, self.sheet.dp_score
        except AttributeError:
            print('AttributeError: Please calculate total score first.')
    
    def print_total_score(self):
        """
        Print the total score.

        Prints the score sheet (self.sheet) containing the total score.

        Returns:
            None
        """
        print(self.sheet)
    
    def save_total_score(self):
        """
        Save the total score.

        Saves the score sheet (self.sheet) containing the total score to the root directory.

        Returns:
            None
        """
        self.sheet.save(self.root)
    
    def plot_total(self, save=False, show=False):
        """
        Plot the total scores.

        Plots the evaluation metrics for input match test (IMTest), input response test (IRTest),
        and drum pattern test (DPTest). Optionally saves the plots and/or shows them.

        Args:
            save (bool): Whether to save the plots. Default is False.
            show (bool): Whether to show the plots. Default is False.

        Returns:
            None
        """
        self.plot_IMTest(save, show)
        self.plot_IRTest(save, show)
        self.plot_DPTest(save, show)
    
    @safe_plot
    def plot_IMTest(self, save=False, show=False):
        """
        Plot the evaluation metrics for input match test (IMTest).

        Plots the confusion matrix, precision-recall curve, ROC curve, and DET curve for the IMTest.
        Optionally saves the plots and/or shows them.

        Args:
            save (bool): Whether to save the plots. Default is False.
            show (bool): Whether to show the plots. Default is False.

        Returns:
            None
        """
        plot_confusion_matrix(self.im_true, self.im_pred, self.im_dir, save, show)
        plot_precision_recall(self.im_true, self.im_pred, self.im_dir, save, show)
        plot_roc_curve(self.im_true, self.im_pred, self.im_dir, save, show)
        plot_det_curve(self.im_true, self.im_pred, self.im_dir, save, show)
    
    @safe_plot
    def plot_IRTest(self, save=False, show=False):
        """
        Plot the evaluation metrics for input response test (IRTest).

        Plots the confusion matrix, precision-recall curve, ROC curve, and DET curve for the IRTest.
        Optionally saves the plots and/or shows them.

        Args:
            save (bool): Whether to save the plots. Default is False.
            show (bool): Whether to show the plots. Default is False.

        Returns:
            None
        """
        plot_confusion_matrix(self.ir_true, self.ir_pred, self.ir_dir, save, show)
        plot_precision_recall(self.ir_true, self.ir_pred, self.ir_dir, save, show)
        plot_roc_curve(self.ir_true, self.ir_pred, self.ir_dir, save, show)
        plot_det_curve(self.ir_true, self.ir_pred, self.ir_dir, save, show)
    
    @safe_plot
    def plot_DPTest(self, save, show):
        """
        Plot the evaluation metrics for drum pattern test (DPTest).

        Plots the drum pattern distribution.
        If drum pattern predictions are available, it plots the distribution using the predicted values.
        Optionally saves the plots and/or shows them.

        Args:
            save (bool): Whether to save the plots. Default is False.
            show (bool): Whether to show the plots. Default is False.

        Returns:
            None
        """
        if self.dp_pred.any():
            total_sum = np.sum(self.dp_pred)
            dp_dist = self.dp_pred / total_sum
        else:
            dp_dist = np.zeros_like(self.dp_pred)
        plot_distribution(dp_dist, self.dp_dir, save, show)