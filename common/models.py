#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import scipy
from scipy.stats import multivariate_normal
import pandas as pd
import numpy as np
from typing import List
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
import tensorflow as tf
import json
import pickle

from .acquisition_functions import AcquisitionFunctions
from .utilities import Data

import pdb
stop = pdb.set_trace

class Mukhoti():
    """[summary]

    Raises:
        AttributeError: [description]
        RuntimeError: [description]
        RuntimeError: [description]
        RuntimeError: [description]
        RuntimeError: [description]

    Returns:
        [type]: [description]
    """
    def __init__(self, num_classes : int = 4,
                 seed: int = 1,
                 model: str = "resnet18",
                 spectral_normalization: bool = True,
                 archi_modif: bool = True,
                 coeff_sn: float = 3.0,
                 cuda: bool = True):
        self.dataset_num_classes = {"cifar10": 10, "cifar100": 100, "svhn": 10, "dirty_mnist": 10}
        self.dataset_loader = {
            "cifar10": cifar10,
            "cifar100": cifar100,
            "svhn": svhn,
            "dirty_mnist": dirty_mnist,
        }
        self.models = {
            "lenet": lenet,
            "resnet18": resnet18,
            "resnet50": resnet50,
            "wide_resnet": wrn,
            "vgg16": vgg16,
        }
        torch.manual_seed(seed)
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")
        print("CUDA set: " + str(cuda))
        self.num_classes = num_classes
        self.net = self.models[model](
        spectral_normalization=spectral_normalization,
            mod=archi_modif,
            coeff=coeff_sn,
            num_classes=num_classes,
            mnist=None,
        )
        

class MLP():
    """ Multi-Layer Perceptron create with Keras Sequential models
    """
    def __init__(self, embedding = "MobileNetV1", num_classes = 4,hidden_layer_sizes = (64, 64, 32), epochs = 60, batch_size = 16):
        """ Initialize model parameters

        Args:
            embedding (str, optional): Name of the embedding used. Defaults to "MobileNetV1".
            num_classes (int, optional): Number of classes. Defaults to 4.
            hidden_layer_sizes (tuple, optional): Architecture of the MLP (number of layers and neurons per layers). Defaults to (64, 64, 32).
            epochs (int, optional): Number of gradient descent epochs. Defaults to 60.
            batch_size (int, optional): Batch size. Defaults to 16.
        """
        if embedding == "MobileNetV1":
            input_shape = (1024)
        elif embedding == "MobileNetV2":
            input_shape = (1280)
        elif embedding == "ResNet50":
            input_shape = (2048)
        else:
            raise AttributeError("The embedding name do not correpond to any supported ones. Please use MobileNetV1, MobileNetV2 or ResNet50.")
        self.model =  tf.keras.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        for nb_neurons in hidden_layer_sizes:
            self.model.add(tf.keras.layers.Dense(nb_neurons, activation="relu"))
        self.model.add(tf.keras.layers.Dense(num_classes, activation = "softmax"))
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.mapping_emb = {"MobileNetV1": 0, "MobileNetV2": 1, "ResNet50": 2}
        self.embedding = embedding
        self.fitted = False
        self.epochs = epochs
        self.batch_size = batch_size
            
    def fit(self, z: np.ndarray, y: np.array):
        """ Fit MLP

        Args:
            z (np.ndarray): Training data
            y (np.array): Training labels
        """
        self.fitted = True
        self.model.fit(z,y, epochs = self.epochs, batch_size = self.batch_size, verbose = False)    

    def predict(self, z: np.ndarray) -> np.ndarray:
        """ Predict softmax probabilities

        Args:
            z (np.ndarray): Feature arrays

        Returns:
            np.ndarray: Arrays of instances class probabilities 
        """
        return self.model.predict(z)
    
    def score(self,z: np.ndarray, y: np.array) -> float:
        """ Compute accuracy

        Args:
            z (np.ndarray): Feature arrays
            y (np.array): Corresponding labels

        Returns:
            float: Return an accuracy score between 0 and 1
        """
        predictions = self.predict(z)
        y_pred = np.argmax(predictions, axis = 1)
        return np.count_nonzero(y_pred == y)/len(y)
    
    def get_weights(self) -> np.ndarray:
        """ Get weights of the MLP

        Returns:
            np.ndarray: Weights of the MLP
        """
        return self.model.get_weights()

class GaussianDensity():
    """ Gaussian Density estimation with fixed bandwith
    """
    def __init__(self, scaler_path: str):
        """Initialize model parameters
        Args:
            scaler_path (str): path to the serialised scaler.
        """
        self.model = KernelDensity(kernel = "gaussian", bandwidth = 0.2)
        self.fitted = False
        try:
            self.scaler = pickle.load(open(scaler_path, "rb"))
            self.loaded = True
        except:
            self.scaler = MinMaxScaler()
            self.loaded = False
 
    def fit(self, z: np.ndarray, y: np.array = None):
        """[summary]

        Args:
            z (np.ndarray): Training datra
            y (np.array, optional): Labels are not used but it was added in parameters for homogeneity with other models. Defaults to None.
        """
        self.model.fit(z)
        self.fitted = True
        

    def export_scaler(self, z):
        self.scaler.fit(self.model.score_samples(z).reshape(-1,1))
        pickle.dump(self.scaler, open('./scaler_params/scaler_GaussianKernel.pkl', "wb"))
        

    def distance(self, z: np.ndarray) -> np.ndarray:
        """ Compute the log density model on the data.

        Args:
            z (np.ndarray): Feature arrays

        Raises:
            RuntimeError: Error if the model is not fitted

        Returns:
            np.ndarray: Array of the log density across the data
        """
        if self.fitted:
            # self.scaler.partial_fit(self.model.score_samples(z).reshape(-1, 1))
            if self.loaded:
                return 1-self.scaler.transform(self.model.score_samples(z).reshape(-1, 1))[:, 0]
            else:
                return -self.model.score_samples(z)
        else:
            raise RuntimeError("The Gaussian Kernel is not fitted")

    def get_scaler_params(self):
        params = self.scaler.get_params(deep=True)
        params['data_range'] = [self.scaler.data_min_.tolist(), self.scaler.data_max_.tolist()]
        return params
class GMM():
    """ Gaussian Mixture Model
    """
    def __init__(self, nb_classes: int = 4, covariance_type: str = "diag", random_state: int = 1):
        """ Initialize parameters

        Args:
            nb_classes (int, optiosnal): Number of classes. Defaults to 4.
            covariance_type (str, optional): [description]. Defaults to "diag".
            random_state (int, optional): [description]. Defaults to 1.
        """
        self.nb_classes = nb_classes
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.fitted = False
        try:
            self.scaler = pickle.load(open('scaler_params/scaler_GMM.pkl', 'rb'))
            self.loaded = True
        except:
            self.scaler = MinMaxScaler()
            self.loaded = False
        
    def fit(self, z: np.ndarray, y: np.array):
        """ Fit GMM

        Args:
            z (np.ndarray): 
            y (np.array): [description]
        """
        self.classes = sorted(np.unique(y))
        self.model = GaussianMixture(
            n_components=len(self.classes),
            covariance_type=self.covariance_type, 
            means_init=np.array([z[y == i].mean(axis=0) for i in self.classes]),
            reg_covar=1e-4,
            random_state=self.random_state)
        self.model.fit(z)
        self.fitted = True
        
    def get_scaler_params(self):
        params = self.scaler.get_params(deep=True)
        params['data_range'] = [self.scaler.data_min_.tolist(), self.scaler.data_max_.tolist()]
        return params

    def export_scaler(self, z):
        self.scaler.fit(self.model.score_samples(z).reshape(-1, 1))
        pickle.dump(self.scaler, open('./scaler_params/scaler_GMM.pkl', "wb"))
        
    def predict(self, z: np.ndarray) -> np.ndarray:
        """ Predict likelihood 

        Args:
            z (np.ndarray): Training data 

        Returns:
            np.ndarray : Likelihood on samples
        """
        return self.model.predict_proba(z)
    
    def score(self,z: np.ndarray, y: np.array) -> float:
        """ Compute accuracy

        Args:
            z (np.ndarray): Feature arrays
            y (np.array): Labels

        Returns:
            float: accuracy between 0 and 1
        """
        y_pred = self.model.predict(z)
        return np.count_nonzero(y_pred == y)/len(y)
    
    def distance(self,z: np.ndarray) -> np.array:
        """ Compute Likelihoods

        Args:
            z (np.ndarray): Feature arrays

        Raises:
            RuntimeError: Raise error if GMM is not fitted

        Returns:
            np.array: Likelihoods on samples
        """
        if self.fitted: 
            
            if self.loaded:
                return self.scaler.transform(self.model.score_samples(z).reshape(-1, 1))
            else:
                return 1 - self.model.score_samples(z)
        else:
            raise RuntimeError("The GMM is not fitted")
        
        
class GmmPerClass():
    """ Ensemble ofGaussian Mixture Models (per class)
    """
    def __init__(self, nb_classes: int = 4, nb_components_per_class: int = 5, covariance_type: str = "diag", random_state: int = 1):
        """ Initialize parameters

        Args:
            nb_classes (int, optiosnal): Number of classes = number of GMM. Defaults to 4.
            nb_components_per_class (int, optional): Number of components per GMM
            covariance_type (str, optional): [description]. Defaults to "diag".
            random_state (int, optional): [description]. Defaults to 1.
        """
        self.nb_classes = nb_classes
        self.nb_components = nb_components_per_class
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.fitted = False
        self.models = [GMM(nb_classes= self.nb_components, covariance_type = covariance_type, random_state = random_state) for _ in range(self.nb_classes)]
        
    def fit(self, z: np.ndarray, y: np.array):
        """ Fit GMM

        Args:
            z (np.ndarray): 
            y (np.array): [description]
        """
        for i in range(self.nb_classes):
            self.models[i].fit(z[y==i], np.random.randint(self.nb_components, size=(len(y[y==i]))))
        self.fitted = True
        
    def predict(self, z: np.ndarray) -> np.ndarray:
        """ Predict likelihood 

        Args:
            z (np.ndarray): Training data 

        Returns:
            np.ndarray : Likelihood on samples
        """
        likelihoods = []
        for gmm in self.models:
            likelihoods.append(gmm.predict(z))
        likelihoods = np.swapaxes(np.array(likelihoods), 0, 1)
        return np.mean(likelihoods, axis = 1)
    
    def score(self,z: np.ndarray, y: np.array) -> float:
        """ Compute accuracy

        Args:
            z (np.ndarray): Feature arrays
            y (np.array): Labels

        Returns:
            float: accuracy between 0 and 1
        """
        predictions = self.distance(z)
        y_pred = np.argmax(predictions)
        return np.count_nonzero(y_pred == y)/len(y)
    
    def distances(self,z: np.ndarray) -> np.array:
        """ Compute likelihoods

        Args:
            z (np.ndarray): Feature arrays

        Returns:
            np.array: Likelihoods on samples
        """
        if self.fitted:
            res = []
            for gmm in self.models:
                res.append(gmm.distance(z))
            return np.swapaxes(np.array(res), 0, 1)
        else:
            raise RuntimeError("The GMM is not fitted")
        
    def distance(self,z: np.ndarray) -> np.array:
        """ Compute Likelihoods

        Args:
            z (np.ndarray): Feature arrays

        Raises:
            RuntimeError: 

        Returns:
            np.array: Likelihoods on samples
        """
        if self.fitted:
            return np.mean(self.distances(z),  axis =1)
            
        else:
            raise RuntimeError("The GMM is not fitted")
class Ensemble():
    """ Ensemble of MLP
    """
    def __init__(self, num_classes = 4, n_models = 12, embedding = "MobileNetV2", archi=MLP, hidden_layer_sizes = (64,64,32), epochs = 60, batch_size = 16):

        self.fitted = False
        self.mapping_emb = {"MobileNetV1": 0, "MobileNetV2": 1, "ResNet50": 2}
        self.embedding = embedding
        self.ensemble = []
        self.params = {"num_classes": num_classes,
                        "embedding": embedding,
                        "hidden_layer_sizes":hidden_layer_sizes,
                        "epochs": epochs,
                        "batch_size": batch_size}

        self.ensemble = [archi(**self.params) for _ in range(n_models)]
            
    def reshape_array(self, array: np.array) -> np.ndarray:
        return np.array([np.squeeze(array, 0) for arr in array])
    
    def fit(self, z: np.ndarray, y: np.array) -> None:
        """ Fit ensemble

        Args:
            z (np.array): Feature data
            y (np.array): Corresponding labels
        """
        for model in self.ensemble:
            model.fit(z,y)
        self.fitted = True
    
    def predict_ensemble(self, z: np.ndarray) -> np.ndarray:
        """ Probability predictions on the features over the MLP of the ensemble

        Args:
            z (np.ndarray): Feature data

        Returns:
            np.ndarray: Ensemble of probabilities. Shape: (nb_instances, nb_mlp, nb_class)
        """
        predictions_ensemble = []
        for model in self.ensemble:
            predictions_ensemble.append(model.predict(z))
        return np.swapaxes(np.array(predictions_ensemble), 0, 1)
    
    def predict(self, z: np.ndarray) -> np.ndarray:
        """ Predict average probabilities over the MLP ensemble

        Args:
            z (np.ndarray): Features data

        Returns:
            np.ndarray: Probabilities. Shape: (nb_instances, nb_class)
        """
        predictions_ensemble = self.predict_ensemble(z)
        return np.mean(predictions_ensemble, axis = 1)

    def score(self, z: np.ndarray, y: np.array) -> float:
        """ Compute accuracy over the features

        Args:
            z (np.ndarray): Features data
            y (np.array): Corresponding labels

        Returns:
            float: Accuracy between 0 and 1
        """
        predictions = self.predict(z)
        y_pred = np.argmax(predictions, axis = 1)
        return np.count_nonzero(y_pred == y)/len(y)
        
    def get_weights(self) -> np.ndarray:
        """ Get weights from the ensemble

        Returns:
            np.ndarray: Weights from all MLP in the ensemble
        """
        weights = []
        for model in self.ensemble:
            weights.append(model.get_weights())
        return np.array(weights, dtype=object)
    
        
class ModelUtilities():
    """ Fucntions to 

    Returns:
        [type]: [description]
    """

    @staticmethod
    def get_models_param(classes: list = ["ace", "nine", "queen", "king"],
                    embeddings: list = ["MobileNetV1", "MobileNetV2", "ResNet50"],
                    nb_mlp : list = [12],
                    architecture: list = [(64,64,32)],
                    epochs: list = [60],
                    batch_sizes: list = [16],
                    
                    ) -> pd.DataFrame:
        """ Generate a benchmark (data frame) that combined all the the model parameters to use.

        Args:
            classes (list, optional): List of the class labels. Defaults to ["ace", "nine", "queen", "king"].
            embeddings (list, optional): List of the embeddings to use. Defaults to ["MobileNetV1", "MobileNetV2", "ResNet50"].
            nb_mlp (list, optional): List of the number of Multi-Layer Perceptron to add in the ensemble. Defaults to [12].
            architecture (list, optional): List of the neural network architectures to use. Defaults to [(64,64,32)].
            epochs (list, optional): List of the number of gradient descent epochs to perform on the neural networks. Defaults to [60].
            batch_sizes (list, optional): List of the number of instances in a batch. Defaults to [16].

        Returns:
            pd.DataFrame: Data frame containing all combinations of the list of parameters chosen.
        """
        models = pd.DataFrame(columns=["name",
                            "embedding",
                            "n_tot_MLP",
                            "architecture",
                            "training_epochs",
                            "batch_size",
                            "classifier",
                            "uncertainty_model",
                            "model",
                            "params",
                            "acquisition_function"
                            ])
        for embedding in embeddings:             
            
            model = GMM
            models = models.append({"name": "GMM-{0}".format(embedding),
                                    "embedding": embedding,
                                    "n_tot_MLP": 0,
                                    "architecture": None,
                                    "training_epochs": None,
                                    "batch_size": None,
                                    "classifier": True,
                                    "uncertainty_model": True,
                                    "model": model,
                                    "params":  {"nb_classes": len(classes)},
                                    "acquisition_function": [model.distance]},
                                    ignore_index=True)
            
            # model = GmmPerClass
            # models = models.append({"name": "GmmPerClass-{0}".format(embedding),
            #                         "embedding": embedding,
            #                         "n_tot_MLP": 0,
            #                         "architecture": None,
            #                         "training_epochs": None,
            #                         "batch_size": None,
            #                         "classifier": True,
            #                         "uncertainty_model": True,
            #                         "model": model,
            #                         "params":  {"nb_classes": len(classes)},
            #                         "acquisition_function": [model.distance]},
            #                         ignore_index=True)
            
            model = GaussianDensity
            models = models.append({"name": "GaussianKernel-{0}".format(embedding),
                                    "embedding": embedding,
                                    "n_tot_MLP": 0,
                                    "architecture": None,
                                    "training_epochs": None,
                                    "batch_size": None,
                                    "classifier": False,
                                    "uncertainty_model": True,
                                    "model": model,
                                    "params":  {"scaler_path":  "./scaler/GaussianKernel_scaler.pkl"},
                                    "acquisition_function": [model.distance]},
                                    ignore_index=True)
        
        # Add Neural Nets models
        for n in nb_mlp:
            for a in architecture:
                for e in epochs:
                    for b in batch_sizes:
                        # Add Deep Ensemble models
                        for embedding in embeddings:
                            model = Ensemble
                            models = models.append({"name": "DeepEnsemble-{0}-epochs={1}".format(embedding, e),
                                                    "embedding": embedding,
                                                    "n_tot_MLP": n,
                                                    "architecture": a,
                                                    "training_epochs": e,
                                                    "batch_size": b,
                                                    "classifier": True,
                                                    "uncertainty_model": True,
                                                    "model": model,
                                                    "params": {"n_models": n, "num_classes": len(classes), "embedding": embedding, 'archi' : MLP, "hidden_layer_sizes": a, "epochs": e, "batch_size": b},
                                                    "acquisition_function": [AcquisitionFunctions.entropy, 
                                                    AcquisitionFunctions.predictions_std, 
                                                    AcquisitionFunctions.var_ratio]},
                                                    ignore_index=True)
                            
                            # Add simple Neural Network
                            models = models.append({"name": "MLP-{0}-epochs={1}".format(embedding, e),
                                                    "embedding": embedding,
                                                    "n_tot_MLP": 1,
                                                    "architecture": a,
                                                    "training_epochs":e,
                                                    "batch_size": b,
                                                    "classifier": True,
                                                    "uncertainty_model": False,
                                                    "model": MLP,
                                                    "params":  {"embedding": embedding, 'num_classes': len(classes), "hidden_layer_sizes": a, "epochs": e, "batch_size": b},
                                                    "acquisition_function": None},
                                                    ignore_index=True)
        return models.set_index("name")
        
    @classmethod
    def get_models(cls, params: pd.DataFrame = None) -> pd.DataFrame:
        """ Get a Data Frame containing the models instantiated with the function get_model_parameters.
        Args:
            params (pd.DataFrame, optional): Model parameters generated with get_model_. Defaults to get_models_param().

        Returns:
            pd.DataFrame: Data frame containing the parameters and models defined
        """
        if params is None:
            params = cls.get_models_param()
        res = pd.DataFrame(columns=[
                            "name",
                            "embedding",
                            "n_tot_MLP",
                            "architecture",
                            "training_epochs",
                            "batch_size",
                            "classifier",
                            "model",
                            ])
        
        for name, row in params.iterrows():
            
            model = row["model"](**row["params"])
            res = res.append({"name": name,
                            "embedding": row["embedding"],
                            "n_tot_MLP": row["n_tot_MLP"],
                            "architecture": row["architecture"],
                            "training_epochs": row["training_epochs"],
                            "batch_size": row["batch_size"],
                            "classifier": row["classifier"],
                            "model": model,
                            }, ignore_index = True)
        
        return res.set_index("name")

    @staticmethod
    def fit_models(models_df: pd.DataFrame, z_train: np.ndarray, y_train: np.ndarray, verbose = True) -> None:
        """ Fit all models in the data frame generated with get_models
        Args:
            models (pd.DataFrame): Data frame summurizing parameters and models used
            z_train (np.ndarray): Features vector of the training instances
            y_train (np.ndarray): Labesls vector of the training instances
            verbose (bool, optional): Display training progress. Defaults to True.
        """
        for model_name, row in models_df.iterrows():
            if verbose:
                print(model_name)
            model = row["model"]
            if "MobileNetV1" in model_name:
                model.fit(Data.select_embedding(z_train, 0), y_train)
            elif "MobileNetV2" in model_name:
                model.fit(Data.select_embedding(z_train, 1), y_train)
            else:
                model.fit(Data.select_embedding(z_train, 2), y_train)
                
    @staticmethod
    def reset_selected_models(classes = ["ace", "nine", "queen","king"], embeddings: list = ["MobileNetV1"], selected_strats: list = ["GaussianKernel-MobileNetV1", "DeepEnsemble-MobileNetV1-epochs=60"]):
        model_params = ModelUtilities.get_models_param(embeddings = embeddings,
                        nb_mlp = [12],
                        architecture = [(64,64,32)],
                        epochs = [60],
                        batch_sizes = [16],
                        classes = classes
                        )
        models = ModelUtilities.get_models(model_params)
        
        models_selected = pd.DataFrame()
        for strat in selected_strats:
            models_selected = models_selected.append(models.loc[strat])
        # models_selected = models_selected.append(models.loc['DeepEnsemble-MobileNetV2-epochs=60'])
        return models_selected
    
if __name__ == '__main__':
    test = Mukhoti()
    stop()