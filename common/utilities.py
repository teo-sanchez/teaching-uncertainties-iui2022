#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from math import e
import tensorflow as tf
import numpy as np
import os
import json
import glob
import itertools
from types import SimpleNamespace
import random
import pandas as pd
import PIL
from PIL import Image
import io
import base64
import scipy.ndimage


from IPython.display import display
import ipywidgets as widgets
from keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler
import ddu_dirty_mnist
from pretty_simple_namespace import pprint
import pickle
import gzip

import pdb
stop = pdb.set_trace

class Mapping:
    @staticmethod
    def condition2uncertainty(c: str):
        mapping = {"A": "aleatoric", "B": "epistemic"}
        return mapping[c]
    
    @staticmethod
    def uncertainty2condition(u: str):
        mapping = {"aleatoric": "A", "epistemic":"B"}
        return mapping[u]

    @staticmethod
    def real2analysis(p: int) -> int:
        mapping = { 3:1,
            15:2,
            28:3,
            14:4,
            23:5,
            32:6,
            30:7,
            25:8,
            10:9,
            11:10,
            1: 11,
            18:12,
            17:13,
            6: 14,
            2: 15,
            21:16
        }
        return mapping[p]
    
    @staticmethod
    def analysis2real(i : int) -> int:
        inverse_mapping = {1: 3,
           2: 15,
           3: 28,
           4: 14,
           5: 23,
           6: 32,
           7: 30,
           8:  25,
           9: 10,
           10: 11,
           11: 1,
           12: 18,
           13: 17,
           14: 6,
           15: 2,
           16: 21}
        return inverse_mapping[i]

    @staticmethod
    def suit2number(v: str) -> int:
        mapping = {"Neuf": 0, "Dame": 1, "Roi": 2}
        return mapping[v]

class Study:
    @staticmethod
    def  import_demography():
        res = pd.DataFrame(columns = ["Genre", "Age", "Field", "Expertise-CS", "Expertise-ML"])
        for i in range(1, 17):
            res.loc[i] = [np.NaN] * 5
        res["Genre"] = ["F", "F", "F", "M", "F", "M", "M", "M", "F", "F", "F", "F", "F", "M", "F", "F"]
        res["Age"] = [30, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18]
        res["Field"] = ["Sociology", "Phisolophy", "Linguistic", "Math", "Design", "Design", "Design", "Design", "Biology", "Economics", "Biology", "Biology", "Biology","Chemistry", "Biology", "Biology"]
        res["Expertise-CS"] = ["full-novice", "novice", "full-novice", "informed", "novice", "novice", "informed", "full-novice", "full-novice", "novice", "full-novice", "full-novice", "full-novice", "novice", "novice", "full-novice"]
        res["Expertise-ML"] = ["novice", "novice", "novice", "full-novice", "novice", "full-novice", "novice", "novice", "novice", "novice", "full-novice", "full-novice", "novice", "full-novice", "full-novice", "novice"]
        return res
    @staticmethod
    def import_results(log_path: "str" = "./log_pilot",  participants: list = [3, 4], conditions: list = ["epistemic", "aleatoric"], study_phase: list = ["survey", "instances"], verbose: bool = True) -> dict:
        """ Import log data from user study

        Args:
            log_path (str, optional): Folder path in which the log data are stored. Defaults to "./log_pilot".
            participants (list, optional): Participant ID to load. Defaults to [3, 4].
            conditions (list, optional): Conditions to load. Defaults to ["epistemic", "aleatoric"].
            study_phase (list, optional): Phase of the study to load. Defaults to ["survey", "instances"].

        Returns:
            dict: A nested dictionnary (dict[participant][condition][phase] containing pd.DataFrames
        """
        mapping = {"aleatoric": "a", "epistemic": "b"}
        res = {}
        for p in participants:
            res[p] = {}
            for c in conditions:
                res[p][c] = {}
                for d in study_phase:
                    if "accuracy" in d:
                        res[p][c]["test_accuracy"] = pd.DataFrame(
                            columns=["images", "truth", "predicted", "answer", "result"])
                    elif "uncertainty" in d:
                        res[p][c]["test_uncertainty"] = pd.DataFrame(
                            columns=["images", "predicted", "answer"])
                    elif "instances" in d:
                        res[p][c]["instances"] = pd.DataFrame(
                            columns=["images", "id", "features", "labels", "timestamp", "deleted"])
                    elif "survey" in d:
                        res[p][c]["survey"] = pd.DataFrame(
                            columns=["question",  "answer", "answer_digits"])
        if verbose:
            print("Loading...")
        cpt_answers = {"acc_eval": 0,
                       "uncert_usefulness": 0,
                       "uncert_relatable": 0}

        for p in participants:
            for c in conditions:
                for d in study_phase:
                    mapping = {"aleatoric":  "with_ambiguity",
                               "epistemic": "with_novelty"}
                    mapping = {"aleatoric": "a", "epistemic": "b"}
                    # TODO: rajouter condition
                    path = "{0}/P{1}/{2}-{1}-{3}.db".format(
                        log_path, p, d, mapping[c])
                    if verbose:
                        # print("P{0} - {1} - {2}".format(p, c, d))
                        print(path)

                    try:
                        file = glob.glob(path)[0]
                        # , object_hook = lambda x : x)
                        if d != "instances":
                            raw_data = JSON.load_file_multiple(file, "r")
                        else:
                            count = 0
                            raw_data = []
                            with open(os.getcwd()+"/"+path, 'r') as f:
                                while True:
                                    count += 1
                                    line = f.readline()
                                    try:
                                        raw_data.append(JSON.load_data(line))
                                    except:
                                        break
                            
                            # test = json.load(open(full_path, "r"))
                            
                        if "accuracy" in d:
                            res[p][c]["test_accuracy"]["truth"] = pd.Series(
                                [d.labels.truth for d in raw_data.content.data])
                            res[p][c]["test_accuracy"]["predicted"] = pd.Series(
                                [d.labels.predicted for d in raw_data.content.data])
                            res[p][c]["test_accuracy"]["answer"] = pd.Series(
                                [d.answer.value for d in raw_data.content.data])
                            res[p][c]["test_accuracy"]["result"] = pd.Series(
                                [d.result for d in raw_data.content.data])
                        elif "uncertainty" in d:
                            if "aleatoric" == c:
                                res[p][c]["test_uncertainty"]["predicted"] = pd.Series(
                                    [d.truth.aleatoric for d in raw_data.content.data])
                                res[p][c]["test_uncertainty"]["answer"] = pd.Series(
                                    [d.answer.aleatoric for d in raw_data.content.data])
                            elif "epistemic" == c:
                                res[p][c]["test_uncertainty"]["predicted"] = pd.Series(
                                    [d.truth.epistemic for d in raw_data.content.data])
                                res[p][c]["test_uncertainty"]["answer"] = pd.Series(
                                    [d.answer.epistemic for d in raw_data.content.data])
                        elif "survey" in d:
                            res[p][c]["survey"]["question"] = pd.Series(["Dans l’ensemble, mon classifieur est capable de reconnaître les cartes que je lui ai apprises.",
                                                                         "La mesure d’incertitude m’a aidé à identifier les exemples que mon classifieur sait reconnaître",
                                                                         "La mesure d’incertitude m’a aidé à identifier les exemples que mon classifieur ne sait pas reconnaître",
                                                                         "La mesure d’incertitude m’a aidé à identifier les exemples ambigus pour mon classifieur",
                                                                         "La mesure d'incertitude a eu globalement un comportement prévisible."])  # pd.Series([d.q.text for d in raw_data.content.data][:3])
                            answers_digits = {d.id: d.a.data.level for d in raw_data.content.data if (True in [e == d.id for e in [
                                                                                                      "acc_eval", "usefulness_knows", "usefulness_unknows", "usefulness_ambiguous", "uncert_relatable"]] and d.q.format == "likert")}
                            answers = {d.id: d.q.options.likert.levels[d.a.data.level] for d in raw_data.content.data if (True in [e == d.id for e in [
                                                                                                                          "acc_eval", "usefulness_knows", "usefulness_unknows", "usefulness_ambiguous", "uncert_relatable"]] and d.q.format == "likert")}
                            res_d = []
                            res_a = []
                            for key in ["acc_eval", "usefulness_knows", "usefulness_unknows", "usefulness_ambiguous", "uncert_relatable"]:
                                if key in answers_digits.keys():
                                    res_d.append(answers_digits[key])
                                    res_a.append(answers[key])
                            res[p][c]["survey"]["answer_digits"] = pd.Series(
                                res_d)
                            # pd.Series([d.q.options.likert.levels[d.a.data.level] for  d in raw_data.content.data][:3])
                            res[p][c]["survey"]["answer"] = pd.Series(res_a)
                        elif "instances" in d:
                            images = []
                            features = []
                            labels = []
                            timestamp = []
                            id = []
                            deleted = []
                            timestamp_ref = Preprocessing.timestamp2seconds(
                                raw_data[0].timestamp)
                            for i, instance in enumerate(raw_data):
                                try:
                                    images.append(
                                        Preprocessing.imagedata2PIL(instance.data))
                                    features.append(instance.x)
                                    labels.append(instance.y)
                                    timestamp.append(Preprocessing.timestamp2seconds(
                                        instance.timestamp) - timestamp_ref)
                                    id.append(instance._id)
                                    deleted.append(False)
                                except AttributeError:
                                    index = id.index(instance._id)
                                    id.append(instance._id)
                                    images.append(images[index])
                                    features.append(features[index])
                                    labels.append(labels[index])
                                    timestamp.append(timestamp[index])
                                    deleted.append(True)
                            res[p][c]["instances"]["images"] = pd.Series(
                                images)
                            res[p][c]["instances"]["id"] = pd.Series(id)
                            res[p][c]["instances"]["features"] = pd.Series(
                                features)
                            res[p][c]["instances"]["labels"] = pd.Series(
                                labels)
                            res[p][c]["instances"]["timestamp"] = pd.Series(
                                timestamp)
                            res[p][c]["instances"]["deleted"] = pd.Series(
                                deleted)
                    except IndexError:
                        print("No log file found with options: participant={0}, condition={1}, phase:{2}".format(
                            p, c, d))

        return res
    @staticmethod
    def processStudyData(data: dict) -> pd.DataFrame:
        """ Convert the data into as single dataframe

        Args:
            res (dict): [description]

        Returns:
            pd.DataFrame: [description]
        """
        condition_mapping = {"aleatoric": "A", "epistemic": "B"}
        res = pd.DataFrame(columns=["strat", "first", "acc_eval", "usefulness_knows", "usefulness_unknows", "usefulness_ambiguous", "uncert_relatable",  "test_acc", "test_uncert", "training_size"])
        for i, p in enumerate(data.keys()):
            for j,c in enumerate(["aleatoric","epistemic"]):
                index = str(Mapping.real2analysis(p)) + condition_mapping[c]
                line = []
                line.append(1 if j else 0)
                line.append(((p+j)%2+1))
                for k, phase in enumerate(["survey", "test_accuracy", "test_uncertainty", "instances"]):
                    if phase == "survey":
                        line += list(data[p][c][phase]["answer_digits"].values)
                    elif phase == "test_accuracy":
                        result = data[p][c]["test_accuracy"]["result"]
                        score = np.sum([1 if val == "right" else (
                            0 if val == "dont_know" else -1) for val in result])
                        # score should be between 0 and 20
                        # score = 10 * score / len(result) + 10
                        line.append(score)
                    elif phase == "test_uncertainty":
                        score = 1-np.mean(np.abs(np.clip(
                            data[p][c]["test_uncertainty"]["predicted"].values, 0, 1)-data[p][c]["test_uncertainty"]["answer"].values))
                        line.append(score)
                    else:
                        line.append(int(list(data[p][c][phase]["deleted"].values).count(False)))
                res.loc[index] = line
        return res
    
    @staticmethod
    def getStrategiesStudy() -> pd.DataFrame:
        """[summary]
        """
        strategies = pd.DataFrame(columns = ["uncertainty_as_guidance", "systematic"])
        for p in range(1,17):
            for c in ["A","B"]:
                row_index = str(p)+c
                strategies.loc[row_index] = [0.] * 2
        
        for p in ["2A", "7B"]:
            strategies.loc[p]["uncertainty_as_guidance"] = 1.
        # for p in ["4A", "7B", "15A"]:
        #     strategies.loc[p]["systematic"] = 1.
        # for p in ["4B", "9B"]:
        #     strategies.loc[p]["disorganized"] = 1.
        # for p in ["6A", "7B", "8A","8B", "9A", "12A", "13B", "14B", "15B", "16A"]:
        for p in ["4B", "6A", "6B", "7B", "8A"]:
            strategies.loc[p]["systematic"] = 1.
        # for p in ["2A", "7B", "8B", "9B", "15A", "15B"]:
        #     strategies.loc[p]["disorganized"] = 1.
        # for p in ["6A", "7B", "8A","8B", "9A", "12A", "13B", "14B", "15B", "16A"]:
        
        #     strategies.loc[p]["exhaustive"] = 1.
        # for p in ["6A", "6B", "15A"]:
        #     strategies.loc[p]["exclusive"] = 1.
        return strategies
    
    @staticmethod
    def computeSystematicity(data):
        
        res = pd.DataFrame(columns= ["pattern_count", "pattern"])
        patterns = list(itertools.permutations(["Neuf","Dame","Roi"]))
        for pattern in list(patterns):
            patterns.append((pattern[0], pattern[0], pattern[1], pattern[1], pattern[2], pattern[2]))
        
        for p in range(1,17):
            for c in ["A", "B"]:
                classes = data[Mapping.analysis2real(p)][Mapping.condition2uncertainty(c)]["instances"]["labels"]
                max_count = 0
                best_pattern = None
                for pattern in patterns:
                    count = 0
                    for i in range(len(classes)-len(pattern)):
                        if all(pattern == classes[i:i+len(pattern)].values):
                            count += 1
                    if count > max_count:
                        max_count = float(count)
                        best_pattern = pattern
                res.loc[str(p)+c] = [max_count, best_pattern]
        return res
    
    @staticmethod
    def computeSimilarity(data):
        """[summary]

        Args:
            data ([type]): [description]

        Returns:
            [type]: [description]
        """
        res = pd.DataFrame(columns= ["intraclass_features", "intraclass_images"])
        
        for p in range(1,17):
            for c in ["A","B"]:
                score_image = []
                score_feature = []
                for category in ["Neuf", "Dame", "Roi"]:
                    instances = data[Mapping.analysis2real(p)][Mapping.condition2uncertainty(c)]["instances"]
                    selection = instances.loc[instances["labels"] == category]
                    images = selection["images"].values
                    features = selection["features"].values
                    
                    scores_image = []
                    scores_feature = []
                    for image_duo in itertools.combinations(images,2):
                        scores_image.append(np.linalg.norm(np.asarray(image_duo[0])-np.asarray(image_duo[1])))
                    for feature_duo in itertools.combinations(features,2):
                        scores_feature.append(np.linalg.norm(np.array(feature_duo[0])-np.array(feature_duo[1])))
                    score_image.append(np.nanmean(scores_image))
                    score_feature.append(np.nanmean(scores_feature))
                res.loc[str(p)+c]=[np.nanmean(score_feature), np.nanmean(score_image)]
        return res

    @staticmethod
    def computeAccuracy(data):
        res = pd.DataFrame(columns= ["accuracy"])
        
        for p in range(1,17):
            for c in ["A","B"]:
                answers= data[Mapping.analysis2real(p)][Mapping.condition2uncertainty(c)]["test_accuracy"]
                truth = answers["truth"].values
                predictions = answers["predicted"].values
                res.loc[str(p)+c] = float(np.sum(truth == predictions))/len(answers)
        return res
                
        

        
class JSON:
    @staticmethod
    def load_file(path: str, mode: str) -> object:
        return json.load(open(path, mode), object_hook=lambda d: SimpleNamespace(**d))

    @staticmethod
    def load_file_multiple(path: str, mode: str, object_hook=lambda d: SimpleNamespace(**d)) -> object:
        res = []
        with open(path, mode) as f:
            data = f.readlines()

        single_string = ""
        for s in data:
            single_string += s
        single_string = single_string.replace("\n", "")
        res = json.loads(single_string, object_hook=object_hook)
        return res

    @staticmethod
    def load_data(data: str) -> object:
        return json.loads(data, object_hook=lambda d: SimpleNamespace(**d))


def select_embedding(z: np.ndarray, n: int) -> np.ndarray:
    res = []
    for inst in z:
        res.append(inst[n])
    return np.array(res)


def center_and_resize(img: PIL.Image, shape: tuple = (224, 224)) -> PIL.Image:
    size = img.size
    height = min(size)
    width = max(size)
    width_index = np.argmax(size)
    center = int(width/2)
    left = int(center - height/2)
    top = 0
    right = int(center + height/2)
    bottom = height
    img_cropped = img.crop((left, top, right, bottom))
    return img_cropped.resize(shape)


class Embeddings():
    """
    Compute features from embeddings
    """

    def __init__(self, embeddings: list = [tf.keras.applications.MobileNet, tf.keras.applications.MobileNetV2, tf.keras.applications.ResNet50]):
        """ Initialize parameters

        Args:
            embeddings (list, optional): Embeddings to consider. Defaults to [tf.keras.applications.MobileNet, tf.keras.applications.MobileNetV2,tf.keras.applications.ResNet50].
        """
        self.ensemble = [embedding(include_top=False, pooling='max')
                         for embedding in embeddings]
        self.freeze_weights()

    def freeze_weights(self):
        """ Freeze weights of the embeddings
        """
        for model in self.ensemble:
            for layer in model.layers:
                layer.trainable = False

    def reshape_X_array(self, X_array: np.array) -> np.ndarray:
        return np.array([np.squeeze(array, 0) for array in X_array])

    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """ Predict all features from image data

        Args:
            X (np.ndarray): Image data

        Returns:
            np.ndarray: Features from the three embeddings
        """
        if len(X.shape) == 1:
            X = self.reshape_X_array(X)
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=0)
        predictions_ensemble = []
        for model in self.ensemble:
            predictions_ensemble.append(model.predict(X)[0])
        return predictions_ensemble


class Preprocessing():
    """ Utility class for preprocessing data. Please download processed data here: 
    """
    @staticmethod
    def timestamp2seconds(timestamp: str) -> int:
        """ Convert date timestamp into seconds
        Args:
            timestamp (str): The timestamp

        Returns:
            int: Number of seconds
        """
        hms = timestamp.split("T")[1].split(".")[0].split(":")
        h = int(hms[0])
        m = int(hms[1])
        s = int(hms[2])
        return s + 60*m + 3600*h

    @staticmethod
    def imagedata2PIL(image_data: str) -> Image:
        """ Convert raw image data into PIL image

        Args:
            image_data (str): raw image data

        Returns:
            Image: PIL image
        """
        without_header = image_data.split('base64,')[1]
        return Image.open(io.BytesIO(base64.b64decode(without_header)), formats=("JPEG",))

    @staticmethod
    def center_and_resize(img: PIL.Image, shape: tuple = (224, 224)) -> PIL.Image:
        """ Center and resize. Used with cards data.

        Args:
            img (PIL.Image): Image to process
            shape (tuple, optional): Final shape of the processed image. Defaults to (224, 224).

        Returns:
            PIL.Image: Processed image
        """
        size = img.size
        height = min(size)
        width = max(size)
        center = int(width/2)
        left = int(center - height/2)
        top = 0
        right = int(center + height/2)
        bottom = height
        img_cropped = img.crop((left, top, right, bottom
                                ))
        return img_cropped.resize(shape)

    @staticmethod
    def process_raw_data(path: str = "../../../datasets/cards-dataset", final_path: str = "../../../datasets/cards-dataset-formatted", train_size: int = 160, test_size: int = 200, uncertainty_pool: int = 120, shape: tuple = (224, 224), seed: int = 2021):
        """[summary]

        Args:
            path (str, optional): Path to raw data. Defaults to "../../../datasets/cards-dataset".
            final_path (str, optional): Path to processed data. Defaults to "../../../datasets/cards-dataset-formatted".
            train_size (int, optional): Number of training images to process. Defaults to 160.
            test_size (int, optional): Number of testing images to process. Defaults to 200.
            uncertainty_pool (int, optional): Number of uncertain images to process. Defaults to 120.
            shape (tuple, optional): Shape of the final images. Defaults to (224,224).
            seed (int, optional): Random seed to shuffle data. Defaults to 2021.
        """

        if not os.path.exists(final_path):
            os.makedirs(final_path)
        for subpath in os.listdir(path):

            final_file_path = os.path.join(final_path, subpath)
            file_path = os.path.join(path, subpath)

            if not os.path.exists(final_file_path):
                os.makedirs(final_file_path)

            if subpath != ".DS_Store":
                files = os.listdir(file_path)
                random.seed = seed
                random.shuffle(files)
                selected_files = files[:train_size]

                for file in selected_files:
                    if file != ".DS_Store":
                        print(file)
                        img = Image.open( os.path.join(file_path, file))
                        img = Preprocessing.center_and_resize(img)
                        img.save(os.path.join(final_file_path, file))

    @staticmethod
    def load_mnist(path: str, kind: str = 'train'):
        """ Load MNIST data from path

        Args:
            path (str): Path
            kind (str, optional): Dataset. Defaults to 'train'.

        Returns:
            tuple: Image data and correponding labels
        """
        labels_path = os.path.join(path,
                                   '%s-labels-idx1-ubyte.gz'
                                   % kind)
        images_path = os.path.join(path,
                                   '%s-images-idx3-ubyte.gz'
                                   % kind)
        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                   offset=8)
        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                   offset=16).reshape(len(labels), 784)
        return images, labels

    @classmethod
    def format_mnist_data(cls, class_labels: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) -> pd.DataFrame:
        """ Extract and format MNIST data in a pd.DataFrame

        Args:
            class_labels (list, optional): List of the class name. Defaults to [0,1,2,3,4,5,6,7,8,9].

        Returns:
            pd.DataFrame: Formatted MNIST data
        """
        df = pd.DataFrame({
            "name": [],
            "class_label": [],
            "uncertainty_label": [],
            "group": [],
            "X_PIL": [],
            "X_array": [],
            "X_mobilenet": [],
            "y_onehot": [],
            "y": [],
            "z_ensemble": [],
        })
        mobile_ensemble = Embeddings()  # input_shape = (32,32,3))
        (X_train, y_train), (X_test, y_test) = mnist.load_data(path='mnist.npz')
        dirty_mnist_train = ddu_dirty_mnist.AmbiguousMNIST(
            "/Users/teosanchez/Recherche/Uncertainty-IUI2022/uncertainty-IUI-2022/benchmark/datasets/raw_data/ambiguous-mnist", train=False, download=False)
        X_ambiguous = dirty_mnist_train.data
        X_novel, y_novel = cls.load_mnist(
            '/Users/teosanchez/Recherche/Uncertainty-IUI2022/uncertainty-IUI-2022/benchmark/datasets/raw_data/fashion/', kind='t10k')
        # Data homogeneisation
        X_ambiguous = np.squeeze(X_ambiguous.numpy(), axis=1)
        X_ambiguous = np.interp(
            X_ambiguous, (X_ambiguous.min(), X_ambiguous.max()), (0, 255)).astype(np.uint8)
        X_ambiguous = X_ambiguous[np.arange(0, 60000, 10)]

        X_novel = X_novel.reshape(X_novel.shape[0], 28, 28)

        n_train = 160
        n_test = 200
        n_ambiguous = 120
        n_novel = 120
        y_novel = np.array([-1]*n_novel)
        y_ambiguous = np.array([-1]*n_ambiguous)
        X_all = np.concatenate([X_train[:n_train], X_test[:n_test],
                               X_ambiguous[:n_ambiguous], X_novel[:n_novel]], axis=0)
        y_all = np.concatenate(
            (y_train[:n_train], y_test[:n_test], y_ambiguous[:n_ambiguous], y_novel[:n_novel]), axis=0)

        labels_onehot = [np.zeros(len(class_labels), dtype=int)
                         for i in range(len(class_labels))]
        for i, onehot in enumerate(labels_onehot):
            onehot[i] = 1
        for i, X_array in enumerate(X_all):
            print("{0}/{1}".format(i+1, len(X_all)))

            if i < n_train:
                group = "train"
                uncertainty_label = "Classifiable"
            elif i < n_train+n_test:
                group = "test"
                uncertainty_label = "Classifiable"
            elif i < n_train+n_test+n_ambiguous:
                group = "uncertain"
                uncertainty_label = "Ambiguous"
            else:
                group = "uncertain"
                uncertainty_label = "Novel"
            name = "{0}_{1}_{2}".format(i+1, y_all[i], group)
            if uncertainty_label == "Classifiable":
                classe = int(y_all[i])
                y = class_labels.index(classe)
                y_onehot = labels_onehot[y]
            else:
                classe = None
                y = None
                y_onehot = None
            tmp = np.expand_dims(X_array, axis=0)
            X_mobilenet = tf.keras.applications.mobilenet.preprocess_input(tmp)
            X_array = scipy.ndimage.zoom(X_array, 1.14, order=2)
            X_array = np.repeat(X_array[:, :, np.newaxis], 3, axis=2)
            z_mobilenet_ensemble = mobile_ensemble.predict_ensemble(X_array)
            tmp = X_array.astype(float)*255/np.max(X_array)
            X_PIL = Image.fromarray(np.uint8(tmp)).convert("RGB")
            df_instance = pd.DataFrame({
                "name": name,
                "class_label": classe,
                "uncertainty_label": uncertainty_label,
                "group": group,
                "X_PIL": [X_PIL],
                "X_array": [X_array],
                "X_mobilenet": [X_mobilenet],
                "y_onehot": [y_onehot],
                "y": y,
                "z_ensemble": [z_mobilenet_ensemble]
            })
            df = df.append(df_instance, ignore_index=True)
        return df

    @staticmethod
    def format_card_data(path: str = "/Users/teosanchez/Recherche/Uncertainty-IUI2022/uncertainty-IUI-2022-benchmark_branch/benchmark/datasets/raw_data/processed_consistent_card_data", class_labels: list = ["nine", "queen", "king"]) -> pd.DataFrame:
        """ Read and format playing card data in a pd.DataFrame

        Args:
            path (str, optional): Path to the preprocessed data. Defaults to "./data/cards-dataset-formatted".
            class_labels (list, optional): List of the class name. Defaults to ["ace", "nine","queen", "king"].

        Returns:
            pd.DataFrame: Formatted Card data
        """
        df = pd.DataFrame({
            "name": [],
            "class_label": [],
            "uncertainty_label": [],
            "group": [],
            "X_PIL": [],
            "X_array": [],
            "X_mobilenet": [],
            "y_onehot": [],
            "y": [],
            "z_ensemble": [],
        })
        mobile_ensemble = Embeddings()
        labels_onehot = [np.zeros(len(class_labels), dtype=int)
                         for i in range(len(class_labels))]
        for i, onehot in enumerate(labels_onehot):
            onehot[i] = 1
        for i, group in enumerate(os.listdir(path)):
            if group != '.DS_Store' and group != "data.pkl":
                for name in os.listdir(os.path.join(path, group)):
                    if name != '.DS_Store':
                        X_PIL = tf.keras.preprocessing.image.load_img(
                            os.path.join(path, group, name))
                        X_array = tf.keras.preprocessing.image.img_to_array(
                            X_PIL)
                        tmp = np.expand_dims(X_array, axis=0)
                        X_mobilenet = tf.keras.applications.mobilenet.preprocess_input(
                            tmp)
                        if group == "uncertain":
                            classe = None
                            y = None
                            y_onehot = None
                        else:
                            classe = name.split("_")[1]
                            y = class_labels.index(classe)
                            y_onehot = labels_onehot[y]
                        # z_mobilenet = mobile.predict(X_mobilenet)
                        z_mobilenet_ensemble = mobile_ensemble.predict_ensemble(
                            X_array)
                        df_instance = pd.DataFrame({
                            "name": name,
                            "class_label": classe,
                            "uncertainty_label": None,
                            "group": group,
                            "X_PIL": [X_PIL],
                            "X_array": [X_array],
                            "X_mobilenet": [X_mobilenet],
                            "y_onehot": [y_onehot],
                            "y": y,
                            "z_ensemble": [z_mobilenet_ensemble]
                        })
                        df = df.append(df_instance, ignore_index=True)
        return df


class Data():
    """ Functions to process and get the data
    """
    

    @staticmethod
    def select_embedding(z: np.ndarray, n: int) -> np.ndarray:
        """ Select one of the three embeddings

        Args:
            z (np.ndarray): The aggregated feature vectors. Each instance contains a list of three arrays for the three embeddings we're using
            n (int): Index of the embedding used

        Returns:
            np.ndarray: Return an array with dimension (number_of_instance, number_of_features)
        """
        res = []
        for inst in z:
            res.append(inst[n])
        return np.array(res)

    @staticmethod
    def sel_keys(d: dict, fragment: str):
        """ Select a  sub-dictionnary based on a substring contained in the key

        Args:
            d (dict): dictionnary
            fragment (str): substring  contained  in the key to  select

        Returns:
            [type]: [description]
        """
        res = {}
        for key in d.keys():
            if fragment in key:
                res[key] = d[key]
        return res

    @staticmethod
    def normalize(uncertainty_values: np.ndarray) -> np.ndarray:
        """ Normalize uncertainty values

        Args:
            uncertainty_values (np.ndarray): Data frame with uncertainty measures

        Returns:
            np.ndarray: normalized values
        """
        scaler = MinMaxScaler()
        uncertainty_scaled = uncertainty_values.drop(
            ["GroundTruth", "Images", "UncertaintyLabel", "Name"], axis=1)
        uncertainty_scaled[list(uncertainty_scaled.keys())
                           ] = scaler.fit_transform(uncertainty_scaled)
        uncertainty_scaled["GroundTruth"] = uncertainty_values["GroundTruth"]
        uncertainty_scaled["UncertaintyLabel"] = uncertainty_values["UncertaintyLabel"]
        uncertainty_scaled["Images"] = uncertainty_values["Images"]
        uncertainty_scaled["Name"] = uncertainty_values["Name"]
        return uncertainty_scaled

    @staticmethod
    def split_data(df: pd.DataFrame, n_images_train: int = 160, n_images_test: int = 200, n_images_uncertain: int = 240, seed: int = 0) -> tuple:
        """ Split data into train, test and uncertain data

        Args:
            df (pd.DataFrame): Loaded pd.DataFrame
            classes (list, optional): List of the classes. Defaults to ["ace", "nine", "queen", "king"].
            n_images_train (int, optional): Number of training data. Defaults to 160.
            n_images_test (int, optional): Number of testing data. Defaults to 200.
            n_images_uncertain (int, optional): Number of uncertain data. Defaults to 240.
            seed (int, optional): Seed number to shuffle pd.DataFrame. Defaults to 0.

        Returns:
            tuple: Training, testing and uncertain data
        """
        list_frames_train = []
        list_frames_test = []
        list_frames_uncertainty = []
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        # for classe in classes:
        #list_frames_train.append(df.loc[(df["class_label"].isin([classe])) & (df["group"] == "train")][:int(n_images_train/len(classes))])
        #list_frames_test.append(df.loc[(df["class_label"].isin([classe])) & (df["group"] == "test")][:int(n_images_test/len(classes))])
        list_frames_train.append(
            df.loc[df["group"] == "train"][:int(n_images_train)])
        list_frames_test.append(
            df.loc[df["group"] == "test"][:int(n_images_test)])
        list_frames_uncertainty.append(
            df.loc[df["group"] == "uncertain"][:int(n_images_uncertain)])
        train = pd.concat(list_frames_train, axis=0)
        test = pd.concat(list_frames_test, axis=0)
        uncertain = pd.concat(list_frames_uncertainty, axis=0)
        return train, test, uncertain

    @classmethod
    def get_dataset(cls, filepath: str = "data/card_data.pkl", seed: int = 0) -> tuple:
        """ Load the data from a pickle file

        Args:
            filepath (str, optional): Path to pickle file. Defaults to "data/card_data.pkl".

        Returns:
            tuple: DataFrame of training, testing and uncertain data
        """
        raw_data_df = pd.read_pickle(filepath)
        print("Dataset loaded...")
        print("Number of images: ", len(raw_data_df))
        print("Features: ", ", ".join(map(str, raw_data_df.columns)))
        print("Labels: ", ", ".join(map(str, np.unique(
            [i for i in raw_data_df["class_label"].tolist() if i != None]))))
        # print("Feature size in the ensemble:", raw_data_df["z_ensemble"][0][0].shape, raw_data_df["z_ensemble"][0][1].shape,raw_data_df["z_ensemble"][0][2].shape,)
        train, test, uncertain = cls.split_data(
            raw_data_df, n_images_train=160, n_images_test=200, n_images_uncertain=240, seed = seed)
        print("Training set size: ", len(train))
        print("Testing set size: ", len(test))
        print("Uncertain pool size: ", len(uncertain))
        return train, test, uncertain

    @staticmethod
    def dataframe2arrays(train: pd.DataFrame, test: pd.DataFrame, uncertain: pd.DataFrame) -> tuple:
        """ Convert pd.Dataframe data into arrays that can be used by the models

        Args:
            train (pd.DataFrame): Training data
            test (pd.DataFrame): Testing data
            uncertain (pd.DataFrame): Uncertain data

        Returns:
            tuple: Data arrays
        """
        z_train = train["z_ensemble"].values
        z_test = test["z_ensemble"].values
        z_uncertain = uncertain["z_ensemble"].values
        y_train = np.array([v for v in train["y"].values])
        y_test = np.array([v for v in test["y"].values])
        if any(y_train != None) and any(y_test != None):
            y_train -= min(y_train)
            y_test -= min(y_test)
        else:
            y_train = None
            y_test = None
        return (z_train, y_train), (z_test, y_test), z_uncertain




class Annotator:
    """ Annotation widget for uncertainy labels
    """

    def __init__(self, database: pd.DataFrame, column: str = "uncertainty_label", image_column: str = "X_PIL", classes: list = ['Classifiable', 'Novel', 'Ambiguous', "I don't know"]):
        """ Initialize parameters

        Args:
            database (pd.DataFrame): Data
            column (str, optional): Column to annotate. Defaults to "uncertainty_label".
            image_column (str, optional): Column containing PIL images. Defaults to "X_PIL".
            classes (list, optional): Classes to label. Defaults to ['Classifiable', 'Novel', 'Ambiguous', "I don't know"].
        """
        self.data = database
        self.column = column
        self.image_column = image_column
        self.cpt = 0
        self.classes = classes
        # GUI
        self.description = widgets.HTML(
            value="",
            placeholder='',
            description='',
        )
        self.radio = widgets.ToggleButtons(
            options=self.classes,
            description='',
            disabled=False,
            button_style='',
            tooltips=['', 'The image is rather novel compared to existing images of Ace, Nine, Queen and King',
                      'The image is rather ambiguous compared to existing images of Ace, Nine, Queen and King']
        )
        self.next = widgets.Button(description="Next image")
        self.previous = widgets.Button(description="Previous image")
        self.navigation = widgets.HBox([self.previous, self.next], layout=widgets.Layout(
            width='100%', display='inline-flex', flex_flow='row wrap'))
        self.output = widgets.Output(layout={'border': '1px solid black'})

    def display_all(self):
        """ Display widgets 
        """
        self.output.clear_output()
        self.description.value = "{0}/{1}: {2}".format(
            self.cpt+1, len(self.data), self.data["name"].values[self.cpt])
        if self.data[self.column].values[self.cpt] == None:
            self.description.value += ", it has no label. What type of uncertainty is that image?"
        else:
            self.description.value += ", it was already classifier as {0}".format(
                self.data[self.column].values[self.cpt])
            self.radio.value = self.data[self.column].values[self.cpt]
        with self.output:
            display(self.description)
            display(self.navigation)
            display(self.data[self.image_column].values[self.cpt])
            display(self.radio)
        return self.output

    def update_next(self, b):
        """ Handler when button next clicked

        Args:
            b ([type]): Handler message
        """
        if self.cpt >= len(self.data)-1:
            self.cpt = 0
            self.data[self.column].values[self.cpt] = self.radio.value
        else:
            self.data[self.column].values[self.cpt] = self.radio.value
            self.cpt += 1
        self.display_all()

    def update_previous(self, b):
        """ Handler when button next clicked

        Args:
            b ([type]): Handler message
        """
        if self.cpt <= 1:
            self.cpt = len(self.data)-1
            self.data[self.column].values[self.cpt] = self.radio.value
        else:
            self.data[self.column].values[self.cpt] = self.radio.value
            self.cpt -= 1
        self.display_all()

    def start(self):
        """ Initialize widget
        """
        self.next.on_click(self.update_next)
        self.previous.on_click(self.update_previous)
        return self.display_all()


if __name__ == '__main__':
    # df_mnist = Preprocessing.format_mnist_data()
    df_cards = Preprocessing.format_card_data()

    pickle.dump(df_cards, open('/Users/teosanchez/Recherche/Uncertainty-IUI2022/uncertainty-IUI-2022-benchmark_branch/benchmark/datasets/processed_data/consistent_card_data.pkl', 'wb'))
    # pickle.dump(df_mnist, open('./data/cards_data.pkl', 'wb'))
    # stop()
    # Preprocessing.process_raw_data(path = "/Users/teosanchez/Recherche/Uncertainty-IUI2022/uncertainty-IUI-2022-benchmark_branch/benchmark/datasets/raw_data/processed_consistent_card_data", final_path = "/Users/teosanchez/Recherche/Uncertainty-IUI2022/uncertainty-IUI-2022-benchmark_branch/benchmark/datasets/raw_data/processed_consistent_card_data", train_size = 150, test_size = 150, uncertainty_pool = 50, shape = (224, 224), seed = 2021)
