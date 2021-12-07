
from typing import List, Union
import numpy as np
import pandas as pd
from progress.bar import Bar
import random
from sklearn.metrics import auc

import pickle

import copy

import pdb


from .utilities import Data
from .models import Ensemble, MLP, ModelUtilities
from .acquisition_functions  import AcquisitionFunctions

from multiprocessing import Queue


stop = pdb.set_trace

        


class PerformanceMetrics():
    """ Compute the performance metrics on the benchmark models: accuracy, separation and active learning accuracy
    """
    @staticmethod
    def modelChange(weights1: np.ndarray, weights2: np.ndarray) -> float:
        """[summary]

        Args:
            weights1 (np.ndarray): [description]
            weights2 (np.ndarray): [description]

        Returns:
            float: [description]
        """  
        def flatten(w):
            w = np.squeeze(w)
            flat_dim = 1
            for dim in w.shape:
                flat_dim *= dim
            w = np.reshape(w, (flat_dim))
            return w
        w1 = flatten(weights1)
        w2 = flatten(weights2)
        distances = []
        for i, arr in enumerate(w1):
            distances.append(np.linalg.norm(arr - w2[i]))
        return np.mean(distances)

    @staticmethod
    def errorReduction(model_t1: None, model_t2: None, test_set: pd.DataFrame) -> float:
        """[summary]

        Args:
            model_t1 (None): [description]
            model_t2 (None): [description]
            test_set (pd.DataFrame): [description]

        Raises:
            NotImplementedError: [description]

        Returns:
            float: [description]
        """
        raise NotImplementedError
        
    @staticmethod
    def recallPrecisionGain(model_t1: None, model_t2: None, test_set: pd.DataFrame) -> float:
        """[summary]

        Args:
            model_t1 (None): [description]
            model_t2 (None): [description]
            test_set (pd.DataFrame): [description]

        Raises:
            NotImplementedError: [description]

        Returns:
            float: [description]
        """
        raise NotImplementedError

    @staticmethod
    def getSeedImages(train: pd.DataFrame, n_images_per_class: int = 1, seed: int = 1, class_labels: list = ["ace", "nine", "queen", "king"]) -> pd.DataFrame:
        """[summary]

        Args:
            train (pd.DataFrame): [description]
            n_images_per_class (int, optional): [description]. Defaults to 1.
            seed (int, optional): [description]. Defaults to 1.
            class_labels (list, optional): [description]. Defaults to ["ace", "nine", "queen", "king"].

        Returns:
            pd.DataFrame: [description]
        """
        dataframe_per_class_list = []
        for c in class_labels:
            dataframe_per_class_list.append(train.loc[(train["class_label"].isin([c]))].sample(n_images_per_class))

        return pd.concat(dataframe_per_class_list)

    @staticmethod
    def accuracy(models: pd.DataFrame, z_train: np.ndarray, y_train: np.ndarray, z_test: np.ndarray, y_test: np.ndarray, verbose: bool = True) -> list:
        """ Get accuracy over the benchmark models

        Args:
            models (pd.DataFrame): Data frame of models
            z_train (np.ndarray): Training features data
            y_train (np.ndarray): Training labels
            z_test (np.ndarray): Testing features data
            y_test (np.ndarray): Testing labels

        Returns:
            list: Accuracy values for each models
        """
        mapping_emb = {"MobileNetV1": 0, "MobileNetV2": 1, "ResNet50": 2}
        res = []
        for model_name, row in models.iterrows():
            if verbose:
                print(model_name)
            model = row["model"]
            embedding = row["embedding"]
            if "Kernel" in model_name:  # Is not a classifier
                res.append(None)
            else:
                res.append(model.score(Data.select_embedding(z_test, mapping_emb[embedding]), y_test))
        return res
    
    @staticmethod
    def AUROC(uncertainty_res: dict) -> dict:
        """ Compute area  under ROC curve

        Args:
            uncertainty_res (dict): uncertainty classification scores

        Returns:
            dict: AUROC
        """
        res = {}
        for key in uncertainty_res.keys():
            res[key] = auc(uncertainty_res[key]["False Positive Rate"], uncertainty_res[key]["True Positive Rate"])
        return res
        
    
    @staticmethod
    def uncertainty_classification(models: pd.DataFrame, 
    z_positive: np.ndarray, 
    z_negative: np.ndarray, 
    acquisition_functions: dict = {"aleatoric entropy": AcquisitionFunctions.entropy, "epistemic std": AcquisitionFunctions.predictions_std}, 
    verbose = True) -> dict:
        """ Compute several metrics on the distinction between two sets of data based on uncertainty estimations

        Args:
            models (pd.DataFrame): Data frame of models
            z_positive (np.ndarray): Training features data of class positive
            z_negative (np.ndarray): Training features data of class negative
            verbose (bool, optional): Display computation progress. Defaults to True.

        Returns:
            dict: A dictionnary containing:
                - The True Positive Rates
                - The True Negative Rates
                - The False Positive Rates
                - The Precision scores
                - The Recall scores
                - The F1 scores                   
                - The different threshold values
                - The normalized threshold values
                - The uncertainties values on data points
        """
        mapping_embs = {"MobileNetV1": 0, "MobileNetV2": 1, "ResNet50": 2}
        
        # Prepare dataset
        n_positive = len(z_positive)
        n_negative = len(z_negative)
        y = np.zeros((n_positive + n_negative))
        y[:n_positive] = 1
        z = np.concatenate((z_positive, z_negative), axis = 0)
        # Result initialization
        res = {}
        for model_name, row in models.iterrows():
            if verbose:
                print(model_name)
            if "MobileNetV1" in model_name:
                embedding = "MobileNetV1"
            elif "MobileNetV2" in model_name:
                embedding = "MobileNetV2"
            elif "ResNet50" in model_name:
                embedding = "ResNet50"
            model = copy.copy(row["model"])
            # Neural Networks model needs acquisition functions to estimate uncertainty. We try several acquisition functions.
            if "DeepEnsemble" in model_name or 'MLP' in model_name:
                for strategy_name, acquisition_function in acquisition_functions.items():
                    if "DeepEnsemble" in model_name or ('MLP' in model_name and ("entropy" in strategy_name or "pmax" in strategy_name)):
                        if verbose:
                            print("\t" + strategy_name)
                        res[model_name + " " + strategy_name] = {
                            "True Positive Rate":[],
                            "True Negative Rate": [],
                            "False Positive Rate": [],
                            "precision": [],
                            "recall": [],
                            "F1": [],
                            "Threshold": [],
                            "Normalized Threshold": [],
                            "Uncertainties": None}
                        if "MLP" in model_name:
                            scores = acquisition_function(model.predict(Data.select_embedding(z, mapping_embs[embedding])))
                        else:
                            scores = acquisition_function(model.predict_ensemble(Data.select_embedding(z, mapping_embs[embedding])))
                        if "pmax" in strategy_name:
                            scores = 1-scores
                    if "DeepEnsemble" in model_name or ("MLP" in model_name and ("entropy" in strategy_name or "pmax" in strategy_name)):
                        res[model_name + " " + strategy_name]["Uncertainties"] = scores
                        h_bins = 50
                        h = np.histogram(scores, bins=h_bins)
                        min_local_thresh = h[1][np.where(h[0] > 5)[0][0]]
                        max_local_thresh = h[1][np.where(h[0] > 5)[0][-1]]
                        min_thresh = np.min(scores)
                        max_thresh = np.max(scores)
                        thresh_points = 10000
                        local_thresh_points = 100
                        thresholds = np.concatenate((np.linspace(min_local_thresh, max_local_thresh, local_thresh_points), np.linspace(min_thresh, max_thresh, thresh_points)), axis = 0)
                        thresholds = np.sort(thresholds)
                        for t in thresholds:
                            tp = np.where(y == 1)[0]
                            tn = np.where(y == 0)[0]
                            pd = np.where(scores < t)[0]
                            nd = np.where(scores >= t)[0]
                            true_positive = len(set(tp) & set(pd)) 
                            true_negative = len(set(tn) & set(nd))
                            false_positive = len(set(tn) & set(pd))
                            false_negative = len(set(tp) & set(nd))
                            tpr = true_positive / len(tp) if len(tp) else 0
                            tnr = true_negative / len(tn) if len(tn) else 0
                            fpr = false_positive / (false_positive + true_negative) if (false_positive + true_negative) else 0
                            if true_positive == 0:
                                precision = np.NaN
                                recall = np.NaN
                                F1 = np.NaN
                            elif false_negative == 0:
                                recall = 1
                                precision = true_positive / (true_positive + false_positive)
                                F1 = 2/((1/recall)+(1/precision))
                            elif false_positive == 0:
                                precision = 1
                                recall = true_positive / (true_positive + false_negative)
                                F1 = 2/((1/recall)+(1/precision))
                            else:
                                recall = true_positive / (true_positive + false_negative)
                                precision = precision = true_positive / (true_positive + false_positive)
                                F1 = 2/((1/recall)+(1/precision))
                            if "DeepEnsemble" in model_name or ("MLP" in model_name and ("entropy" in strategy_name or "pmax" in strategy_name)):
                                res[model_name + " " + strategy_name]["True Positive Rate"].append(tpr)
                                res[model_name + " " + strategy_name]["True Negative Rate"].append(tnr)
                                res[model_name + " " + strategy_name]["False Positive Rate"].append(fpr)
                                res[model_name + " " + strategy_name]["precision"].append(precision)
                                res[model_name + " " + strategy_name]["recall"].append(recall)
                                res[model_name + " " + strategy_name]["F1"].append(F1)
                                res[model_name + " " + strategy_name]["Threshold"].append(t)
                        res[model_name + " " + strategy_name]["Normalized Threshold"] = (thresholds - min(thresholds))/max(thresholds - min(thresholds))
            # Density based models use likelihood estimations
            elif model_name[0] == "G":
                scores = model.distance(Data.select_embedding(z, mapping_embs[embedding]))

                res[model_name] = {"True Positive Rate":[],
                                "True Negative Rate": [],
                                "False Positive Rate": [],
                                "precision": [],
                                "recall": [],
                                "F1": [],
                                "Threshold": [],
                                "Normalized Threshold": [],
                                "Uncertainties": None}
                
                res[model_name]["Uncertainties"] = scores

                h_bins = 50 if model_name != "GDA" else 500
                h = np.histogram(scores, bins=h_bins)


                min_local_thresh = h[1][np.where(h[0] > 5)[0][0]]
                max_local_thresh = h[1][np.where(h[0] > 5)[0][-1]]

                min_thresh = np.min(scores)
                max_thresh = np.max(scores)

                thresh_points = 10000

                local_thresh_points = 100
                
                thresholds = np.concatenate((np.linspace(min_local_thresh, max_local_thresh, local_thresh_points), np.linspace(min_thresh, max_thresh, thresh_points)), axis = 0)
                thresholds = np.sort(thresholds)
                res[model_name]["Normalized Threshold"] = (thresholds - min(thresholds))/max(thresholds - min(thresholds))
                for t in thresholds:
                    tp = np.where(y == 1)[0]
                    tn = np.where(y == 0)[0]
                    pd = np.where(scores < t)[0]
                    nd = np.where(scores >= t)[0]

                    true_positive = len(set(tp) & set(pd)) 
                    true_negative = len(set(tn) & set(nd))

                    false_positive = len(set(tn) & set(pd))
                    false_negative = len(set(tp) & set(nd))

                    tpr = true_positive / len(tp) if len(tp) else 0
                    tnr = true_negative / len(tn) if len(tn) else 0
                    fpr = false_positive / (false_positive + true_negative) if (false_positive + true_negative) else 0

                    if true_positive == 0:
                        precision = np.NaN
                        recall = np.NaN
                        F1 = np.NaN
                    elif false_negative == 0:
                        recall = 1
                        precision = true_positive / (true_positive + false_positive)
                        F1 = 2/((1/recall)+(1/precision))
                    elif false_positive == 0:
                        precision = 1
                        recall = true_positive / (true_positive + false_negative)
                        F1 = 2/((1/recall)+(1/precision))
                    else:
                        recall = true_positive / (true_positive + false_negative)
                        precision = precision = true_positive / (true_positive + false_positive)
                        F1 = 2/((1/recall)+(1/precision))
                        
                    res[model_name]["True Positive Rate"].append(tpr)
                    res[model_name]["True Negative Rate"].append(tnr)
                    res[model_name]["False Positive Rate"].append(fpr)
                    res[model_name]["precision"].append(precision)
                    res[model_name]["recall"].append(recall)
                    res[model_name]["F1"].append(F1)
                    res[model_name]["Threshold"].append(t)


            else:
                scores = None
                continue
            
                
        
        return res
    
    @staticmethod
    def curriculum(models: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, uncertain: pd.DataFrame,
                                nb_replication: int = 10, budget: Union[list, np.array] = np.arange(120),
                                seed_size: list = [1],
                                classes : list = ["nine", "queen", "king"],
                                output_name: str = "curricula.pkl",
                                acquisition_functions: dict = {"aleatoric entropy": AcquisitionFunctions.entropy, "epistemic std":AcquisitionFunctions.predictions_std, "random":  AcquisitionFunctions.random_acquisition}) -> pd.DataFrame:
        """ Perform a curriculum with the given parameters and acquisition function

        Args:
            models (pd.DataFrame): Untrained models
            train (pd.DataFrame): Training set and pool of data
            test (pd.DataFrame): Testing set
            uncertainty (pd.DataFrame): Uncertain data
            nb_replication (int, optional): Number of iteration of the curriculum. Defaults to 10.
            budget (int, optional): Final size of the pool. Defaults to 60.
            step (list or int, optional): Increment of how much instances are added.  Can be a list of the number of data added at each steps. Defaults to [].
            seed_size (list, optional): Initial size of the training set. Defaults to [1].
            model_params (dict, optional): Model parameters. Defaults to {"classes": ["ace", "nine", "queen", "king"],  "embeddings": ["MobileNetV1", "MobileNetV2", "ResNet50"],  "nb_mlp" : [12], "architecture": [(64,64,32)], "epochs": [60], "batch_sizes": [16]}.
            acquisition_functions (dict, optional): Acquisition functions. Defaults to {"aleatoric entropy": AcquisitionFunctions.entropy, "epistemic std":AcquisitionFunctions.predictions_std, "random": AcquisitionFunctions.random_acquisition}.

        Returns:
            pd.DataFrame: [description] TODO
        """
        def reshape_z(z):
            stop()
            res = np.array()
            for i in len(z):
                res = np.concatenate((res, z[i][0]), axis = 1)
            return res
                
                
        mapping_emb = {"MobileNetV1": 0, "MobileNetV2": 1, "ResNet50": 2}
        acquisition_functions_mlp = dict(acquisition_functions)
        acquisition_functions_mlp.pop("epistemic std")
        acquisition_functions_mlp.pop("random")
        print(acquisition_functions_mlp)
        res = {}
        for model_name, row in models.iterrows():
            if "DeepEnsemble" in model_name:
                for strategy_name, acquisition_function in acquisition_functions.items():
                    res[model_name + " " + strategy_name] = pd.DataFrame(columns=["seed_size", "epochs", "architecture mlp", "accuracy", "model change", "tpr", "fpr", "uncertainty",  "card added", "correlation_matrix"])
            elif 'MLP' in model_name:
                for strategy_name, acquisition_function in acquisition_functions_mlp.items():
                    res[model_name + " " + strategy_name] = pd.DataFrame(columns=["seed_size", "epochs", "architecture mlp", "accuracy", "model change", "tpr", "fpr", "uncertainty",  "card added", "correlation_matrix"])
            else:
                res[model_name] = pd.DataFrame(columns=["seed_size", "epochs", "architecture mlp", "accuracy", "model change", "auroc", "uncertainty",  "card added", "correlation_matrix"])
        print(res)
        # Prepare train dataset
        (z_train, y_train), (z_test, y_test), z_uncertain = Data.dataframe2arrays(train, test, uncertain)
        
        for f in res.keys():
            for s in seed_size:
                accuracy = np.zeros((nb_replication, len(budget)))
                model_change = np.zeros((nb_replication, len(budget)))
                uncertainty = np.zeros((nb_replication, len(budget)))
                auroc = np.zeros((nb_replication, len(budget)))
                # tpr = np.zeros((nb_replication, len(budget)))
                # fpr = np.zeros((nb_replication, len(budget)))
                card_added = [[None] * len(budget)]*nb_replication
                    
                for i in range(nb_replication):
                    print("{0}: Replication {1}/{2}, seed size: {3},".format(f, i+1, nb_replication, s))
                    
                    curriculum = pd.DataFrame(columns=["seed_size", "epochs", "architecture mlp", "accuracy", "model change", "auroc", "uncertainty",  "card added"])

                    # Training set initialization
                    seed = PerformanceMetrics.getSeedImages(train, n_images_per_class = s, class_labels = classes)
                    pool = pd.DataFrame(train).drop(seed.index)
                    # uncertainty_pool = pd.DataFrame(uncertain)                    


                    for nn, n in enumerate(budget):
                        print("{0}/{1}".format(n, max(budget)))
                        
                        (z, y), (z_test, y_test), _ = Data.dataframe2arrays(seed, test, uncertain)
                        (z_pool, _), (z_test, y_test), z_uncertain = Data.dataframe2arrays(pool, test, uncertain)
                        
                        # Model initialization
                        if "DeepEnsemble" in f or "MLP" in f:
                            model_name = f.split(" ")[0]
                            # performance_model = copy.deepcopy(models.loc[model_name].model)
                        else:
                            model_name = f
                        performance_model = Ensemble(num_classes = len(classes), n_models = 3, embedding = models.loc[model_name].embedding, archi=MLP, hidden_layer_sizes = (64,32), epochs = 20, batch_size = models.loc[model_name].batch_size)
                        
                        perf_weights_before = performance_model.get_weights()
                        
                        # Fit models
                        performance_model.fit(Data.select_embedding(z, mapping_emb[models.loc[model_name].embedding]) ,y)
                        
                        perf_weights_after = performance_model.get_weights()
                        
                        # Compute model change
                        model_change[i,nn] = PerformanceMetrics.modelChange(perf_weights_before, perf_weights_after) 
                        # Compute model accuracy
                        accuracy[i,nn] = performance_model.score(Data.select_embedding(z_test, mapping_emb[models.loc[model_name].embedding]), y_test)
                        
                        single_model = pd.DataFrame([], columns = list(models.keys())).append(models.loc[model_name])
                        single_model.model.values[0]  = copy.deepcopy(models.loc[model_name].model)
                        stop()
                        ModelUtilities.fit_models(single_model, z, y, verbose=False)
                        if "MLP" in f:
                            auroc[i,nn] = PerformanceMetrics.AUROC(PerformanceMetrics.uncertainty_classification(single_model, z_test, z_uncertain, acquisition_functions = acquisition_functions_mlp, verbose= False))[f]
                        else:
                            auroc[i,nn] = PerformanceMetrics.AUROC(PerformanceMetrics.uncertainty_classification(single_model, z_test, z_uncertain, acquisition_functions = acquisition_functions, verbose= False))[f]

                        uncertainties = acquisition_function(performance_model.predict(Data.select_embedding(z_pool, mapping_emb[models.loc[model_name].embedding])))
                        uncertainty[i,nn] = np.max(uncertainties)
                        most_uncertain_index = np.argmax(uncertainties)
                        most_uncertain = pool.iloc[most_uncertain_index]
                        seed = seed.append(most_uncertain)
                        pool = pool.drop(pool[pool['name'] == most_uncertain["name"]].index)
                        card_added[i][nn] = most_uncertain["X_PIL"]
                        
                    tmp = {
                        "seed_size": s,
                        "epochs": models.loc[model_name]['training_epochs'],
                        "architecture mlp": models.loc[model_name]['architecture'],
                        "accuracy": [accuracy],
                        "model change": [model_change],
                        "auroc": [auroc],
                        "uncertainty": [uncertainty],
                        "card added":  [card_added],
                        }
                    curriculum = curriculum.append(tmp, ignore_index = True)
            
                res[f] = res[f].append(curriculum, ignore_index = True)
                pickle.dump(res, open("./" + output_name, 'wb'))    
