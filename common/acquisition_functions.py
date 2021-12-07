import numpy as np
from typing import List,Union
import random


class AcquisitionFunctions():
    """ Acquisition functions
    """
    @staticmethod
    def a():
        return 1
    @staticmethod
    def pmax(probs: Union[np.ndarray, List[float]]) -> float:
        """[summary]

        Args:
            probs (Union[np.ndarray, List[float]]): [description]

        Raises:
            ValueError: [description]

        Returns:
            float: [description]
        """
        if len(probs.shape)>2:
            mean_probs = np.mean(probs, axis = 1)
        else:
            mean_probs = probs
        if any(np.abs(np.sum(mean_probs, axis = 1)-1) > 0.0001):
            raise ValueError("The sum of the array representing probabilities must sum to 1")
        else:
            # eps = np.finfo(float).eps
            return np.max(mean_probs, axis = 1)
        
    @staticmethod
    def entropy(probs: Union[np.ndarray, List[float]]) -> float:
        """[summary]

        Args:
            probs (Union[np.ndarray, List[float]]): [description]

        Raises:
            ValueError: [description]

        Returns:
            float: [description]
        """
        if len(probs.shape)>2:
            mean_probs = np.mean(probs, axis = 1)
        else:
            mean_probs = probs
        if any(np.abs(np.sum(mean_probs, axis = 1)-1) > 0.0001):
            raise ValueError("The sum of the array representing probabilities must sum to 1")
        else:
            eps = np.finfo(float).eps
            return -np.sum(np.log2(mean_probs + eps)* mean_probs, axis = 1)

    @staticmethod
    def margin(probs: Union[np.ndarray, List[float]]) -> float:
        """[summary]

        Args:
            probs (Union[np.ndarray, List[float]]): [description]

        Raises:
            ValueError: [description]

        Returns:
            float: [description]
        """
        if np.abs(np.sum(probs)- 1) > 0.0001:
            raise ValueError("The sum of the array representing probabilities \
                            must sum to 1")
        else:
            sorted_probs = np.sort(probs)
            return round(sorted_probs[-1] - sorted_probs[-2], 4)
        
    @staticmethod
    def var_ratio(ensemble_prob: Union[np.ndarray, List[np.ndarray]]) -> float:
        """[summary]

        Args:
            ensemble_prob (Union[np.ndarray, List[np.ndarray]]): [description]

        Returns:
            float: [description]
        """
        if len(ensemble_prob.shape) <=1:
            ensemble_prob = np.expand_dims(ensemble_prob, axis =1)
        nb_ens = ensemble_prob.shape[1]
        nb_classes = ensemble_prob.shape[2]
        maximums = np.argmax(ensemble_prob, axis = 2)
        res = []
        for i in range(len(maximums)):
            max_counts = np.array([np.count_nonzero(maximums[i] == c) for c in range(nb_classes)])
            res.append(1-np.max(max_counts/nb_ens))
        return np.array(res)
            
    @staticmethod
    def predictions_std(ensemble_prob: Union[np.ndarray, List[np.ndarray]]) -> float:
        """[summary]

        Args:
            ensemble_prob (Union[np.ndarray, List[np.ndarray]]): [description]

        Returns:
            float: [description]
        """
        if len(ensemble_prob.shape) <=1:
            ensemble_prob = np.expand_dims(ensemble_prob, axis =1)
        else:
            stds = np.std(ensemble_prob, axis = 1)
            return np.mean(stds, axis = 1)

    @staticmethod
    def random_acquisition(ensemble_prob: Union[np.ndarray, List[np.ndarray]]) -> float:
        """[summary]

        Args:
            ensemble_prob (Union[np.ndarray, List[np.ndarray]]): [description]

        Returns:
            float: [description]
        """
        return np.array([random.random() for i in range(len(ensemble_prob))])