#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib
font = {'family' : 'Times',
        'size'   : 22}

matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.figure import Figure   
from matplotlib.patches import Patch   
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pylab
from typing import List, Optional, Iterable, Tuple
from PIL import Image
import math
import tensorflow as tf
from sklearn.decomposition import PCA

import  seaborn as sns

import pdb
stop = pdb.set_trace

from .utilities import Data
from .models import MLP

import plot_likert


class Figures():
    """ Plot functions for the benchmark and paper
    """
    @staticmethod
    def roc(res, colors="", title= ""):
        """ Plot ROC curve

        Args:
            res (dict): Results from the function metrics.PerformanceMetrics.uncertainty_classification
            title (str, optional): Title of the plot. Defaults to "".
        """
        NUM_COLORS = len(res)
        if colors == "linear":
            cm_mobilenetv1 = pylab.get_cmap('hsv')
            cm_mobilenetv2 = pylab.get_cmap('hsv')
            cm_resnet = pylab.get_cmap('hsv')
            cpt_mobilenetv1 = 0
            cpt_mobilenetv2 = 0
            cpt_resnet = 0
        else:
            cm_mobilenetv1 = pylab.get_cmap('Blues')
            cm_mobilenetv2 = pylab.get_cmap('Oranges')
            cm_resnet = pylab.get_cmap('Greens')
            cpt_mobilenetv1 = 2
            cpt_mobilenetv2 = 2
            cpt_resnet = 2
        plt.figure(figsize=(25,12))
        for model_name in res.keys():
            if model_name[0] == "G":
                line = ":"
                linewidth =2
            else:
                line = "--"
                linewidth = 1.5
            if "MobileNetV1" in model_name:
                color = cm_mobilenetv1(1.*cpt_mobilenetv1/NUM_COLORS)
                cpt_mobilenetv1+=1
                plt.plot(np.array(res[model_name]["False Positive Rate"]), res[model_name]["True Positive Rate"], line, color = color, linewidth = linewidth, label = model_name)
            if "MobileNetV2" in model_name:
                color = cm_mobilenetv2(1.*cpt_mobilenetv2/NUM_COLORS)
                cpt_mobilenetv2+=1
                plt.plot(np.array(res[model_name]["False Positive Rate"]), res[model_name]["True Positive Rate"], line, color = color, linewidth = linewidth, label = model_name)    
            if "ResNet50" in model_name:
                color = cm_resnet(1.*cpt_resnet/NUM_COLORS)
                cpt_resnet+=1
                plt.plot(np.array(res[model_name]["False Positive Rate"]), res[model_name]["True Positive Rate"], line, color = color, linewidth = linewidth, label = model_name)
        plt.plot([0,1], [0,1],  "-", linewidth =3, color = "k", label = "Random" )
        plt.legend(loc = "best", fontsize = 16)
        plt.xlabel("False Positive Rate", fontsize = 22)
        plt.ylabel("True Positive Rate", fontsize =22)
        plt.title(title, fontsize = 36)
        plt.xticks(fontsize= 14)
        plt.yticks(fontsize= 14)
    
    @staticmethod
    def precision_recall(res, colors="", title = ""):
        """ Plot the precision-recall curve

        Args:
            res (dict): Results from the function metrics.PerformanceMetrics.uncertainty_classification
            title (str): Title of the curve
        """
        NUM_COLORS = len(res)
        if colors == "linear":
            cm_mobilenetv1 = pylab.get_cmap('hsv')
            cm_mobilenetv2 = pylab.get_cmap('hsv')
            cm_resnet = pylab.get_cmap('hsv')
            cpt_mobilenetv1 = 0
            cpt_mobilenetv2 = 0
            cpt_resnet = 0
        else:
            cm_mobilenetv1 = pylab.get_cmap('Blues')
            cm_mobilenetv2 = pylab.get_cmap('Oranges')
            cm_resnet = pylab.get_cmap('Greens')
            cpt_mobilenetv1 = 2
            cpt_mobilenetv2 = 2
            cpt_resnet = 2
        plt.figure(figsize=(25,12))
        for model_name in res.keys():
            if model_name[0] == "G":
                line = ":"
                linewidth =2
            else:
                line = "--"
                linewidth = 1.5   
            if "MobileNetV1" in model_name:
                color = cm_mobilenetv1(1.*cpt_mobilenetv1/NUM_COLORS)
                cpt_mobilenetv1+=1
                plt.plot(np.array(res[model_name]["recall"]), res[model_name]["precision"], line, color = color, linewidth = linewidth, label = model_name)
            if "MobileNetV2" in model_name:
                color = cm_mobilenetv2(1.*cpt_mobilenetv2/NUM_COLORS)
                cpt_mobilenetv2+=1
                plt.plot(np.array(res[model_name]["recall"]), res[model_name]["precision"], line, color = color, linewidth = linewidth, label = model_name)
            if "ResNet50" in model_name:
                color = cm_resnet(1.*cpt_resnet/NUM_COLORS)
                cpt_resnet+=1
                plt.plot(np.array(res[model_name]["recall"]), res[model_name]["precision"], line, color = color, linewidth = linewidth, label = model_name)
        plt.legend(loc = "best", fontsize = 14)
        plt.xlabel("Recall", fontsize = 22)
        plt.ylabel("Precision", fontsize = 22)
        plt.title(title, fontsize = 22)
        plt.xticks(fontsize= 14)
        plt.yticks(fontsize= 14)
    
    @staticmethod
    def f1_score(res, colors="", title = ""):
        """ Plot the F1 score curve

        Args:
            res (dict): Results from the function metrics.PerformanceMetrics.uncertainty_classification
            title (str): Title of the curve
        """
        NUM_COLORS = len(res)
        if colors == "linear":
            cm_mobilenetv1 = pylab.get_cmap('hsv')
            cm_mobilenetv2 = pylab.get_cmap('hsv')
            cm_resnet = pylab.get_cmap('hsv')
            cpt_mobilenetv1 = 0
            cpt_mobilenetv2 = 0
            cpt_resnet = 0
        else:
            cm_mobilenetv1 = pylab.get_cmap('Blues')
            cm_mobilenetv2 = pylab.get_cmap('Oranges')
            cm_resnet = pylab.get_cmap('Greens')
            cpt_mobilenetv1 = 2
            cpt_mobilenetv2 = 2
            cpt_resnet = 2
        plt.figure(figsize=(25,12))
        for model_name in res.keys():
            if model_name[0] == "G":
                line = ":"
                linewidth =2
            else:
                line = "--"
                linewidth = 1.5
            if "MobileNetV1" in model_name:
                color = cm_mobilenetv1(1.*cpt_mobilenetv1/NUM_COLORS)
                cpt_mobilenetv1+=1
                plt.plot(np.linspace(0,1,len(res[model_name]["Normalized Threshold"])), res[model_name]["F1"], line, color = color, linewidth = linewidth, label = model_name)
            if "MobileNetV2" in model_name:
                color = cm_mobilenetv2(1.*cpt_mobilenetv2/NUM_COLORS)
                cpt_mobilenetv2+=1
                plt.plot(np.linspace(0,1,len(res[model_name]["Normalized Threshold"])), res[model_name]["F1"], line, color = color, linewidth = linewidth, label = model_name)
            if "ResNet50" in model_name:
                color = cm_resnet(1.*cpt_resnet/NUM_COLORS)
                cpt_resnet+=1
                plt.plot(np.linspace(0,1,len(res[model_name]["Normalized Threshold"])), res[model_name]["F1"], line, color = color, linewidth = linewidth, label = model_name)
        plt.legend(loc = "best", fontsize = 14)
        plt.xlabel("Normalizde threshold indexes", fontsize = 22)
        plt.ylabel("F1 score", fontsize = 22)
        plt.title(title, fontsize = 22)
        plt.xticks(fontsize= 14)
        plt.yticks(fontsize= 14)

    @staticmethod
    def scree_plot(z_train: np.ndarray, z_test: np.ndarray, z_uncertain: np.ndarray, nb_components: int = 10, embeddings: dict = {0: "MobileNetV1", 1: "MobileNetV2", 2: "ResNet50", 3: "[MobileNetV1 + MobileNetV2]", 4: "All embeddings"},
                   figsize: tuple = (25,10)):
        """ Return a Screen plot of the embeddings

        Args:
            z_train (np.ndarray): Training data
            z_test (np.ndarray): Testing Data
            z_uncertain (np.ndarray): Uncertain Data
            nb_components (int, optional): Number of components to displat, maximum = 6. Defaults to 6.
            embeddings (dict, optional): Embeddings. Defaults to {0: "MobileNetV1", 1: "MobileNetV2", 2: "ResNet50", 3: "[MobileNetV1 + MobileNetV2]", 4: "All embeddings"}.
            figsize (tuple, optional): Size of the figure. Defaults to (25,10).
        """
        indexes = np.arange(nb_components)
        fig = plt.figure(figsize = figsize)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        z_all = np.concatenate((z_train, z_test, z_uncertain), axis = 0)
        variances = []
        for i in range(len(embeddings)):
            if i < 3:
                z = Data.select_embedding(z_all, i)
            elif i == 3: # All the embeddings
                z = np.concatenate( (Data.select_embedding(z_all, 0), Data.select_embedding(z_all, 1)) , axis = 1)
            else:
                z = np.concatenate( (Data.select_embedding(z_all, 0), Data.select_embedding(z_all, 1), Data.select_embedding(z_all, 2)) , axis = 1)    
            pca = PCA(n_components=nb_components, svd_solver='full')
            pca.fit(z)
            var = pca.explained_variance_ratio_
            variances.append(var)
            ax.bar(indexes/6 + i, var*100, width = 0.15, label = embeddings[i])
            ax.set_xticks(np.arange(len(embeddings)) + 0.325)
            ax.set_xticklabels(embeddings.values())
        plt.legend(loc="best")
        plt.xlabel("Principal Components of the PCA")
        plt.ylabel("% of the variance explained on the component")
        return variances
    
    @classmethod
    def image_grid(cls,  images: List[np.ndarray], columns: int = 5, titles: Optional[Iterable] = None, figsize: Tuple = (15,15)) -> Figure:
        """
        Return a figure holding the images arranged in a grid

        Optionally the number of columns and/or image titles can be provided.

        Example:

            >>> image_grid(images)
            >>> image_grid(images, titles=[....])

        """
        if type(images[0]) == np.ndarray: 
            images = [Image.fromarray((img * 255 / np.max(img)).astype('uint8')) for img in images] # Convertir l'array numpy à une image PIL
            #images = [Image.fromarray((img*255/(np.max(img)).astype('uint8'))) for img in images] # Convertir l'array numpy à une image PIL
        rows = math.ceil(1.0 * len(images) / columns)
        fig = Figure(figsize=(figsize[0], figsize[1] * rows / columns))
        if titles is None:
            titles = range(len(images))
        for k, (img, title) in enumerate(zip(images, titles)):
            ax = fig.add_subplot(rows, columns, k + 1)
            ax.imshow(img)
            ax.tick_params(axis="both", labelsize=0, length=0)
            ax.grid(b=False)
            # ax.set_xlabel(title, labelpad=-4)
        return fig
    
    @classmethod
    def display_most_uncertain_images(cls, df: pd.DataFrame, strategy: List[str], nb_images: int = 20, nb_columns: int = 4) -> Figure:
        """ Display most uncertain umages

        Args:
            df (pd.DataFrame): Dataframe containing images and uncertainties
            strategy ([type]): List of strategy names
            nb_images (int, optional): Number of images on the figure. Defaults to 12.
            nb_columns (int, optional): Number of columns. Defaults to 3.

        Returns:
            Figure: A pyplot figure showing most uncertain images on a grid
        """
        df_sorted = df.sort_values(strategy,ascending= False)
        images = df_sorted["Images"].values
        titles = ["{0}".format(row["Name"]) for idx, row in df_sorted.iterrows()]
        X_sorted = [tf.keras.preprocessing.image.img_to_array(img) for img in images]
        return cls.image_grid(X_sorted[:nb_images], nb_columns, titles = titles)

    @classmethod
    def display_most_certain_images(cls, df: pd.DataFrame, strategy: List[str], nb_images: int = 20, nb_columns: int = 4) -> Figure:
        """ Display most certain umages

        Args:
            df (pd.DataFrame): Dataframe containing images and uncertainties
            strategy ([type]): List of strategy names
            nb_images (int, optional): Number of images on the figure. Defaults to 12.
            nb_columns (int, optional): Number of columns. Defaults to 3.

        Returns:
            Figure: A pyplot figure showing most certain images on a grid
        """
        df_sorted = df.sort_values(strategy,ascending= True)
        images = df_sorted["Images"].values
        titles = ["{0}".format(row["Name"]) for idx, row in df_sorted.iterrows()]
        X_sorted = [tf.keras.preprocessing.image.img_to_array(img) for img in images]
        return cls.image_grid(X_sorted[:nb_images], nb_columns, titles = titles)

    @staticmethod
    def show_uncertainty(strat_name: str):
        """ Show most uncertain and most certain images

        Args:
            strat_name (str): [description]
        """
        print(strat_name)
        img1 = mpimg.imread('images/{0}_uncertain.png'.format(strat_name))
        img2 = mpimg.imread('images/{0}_certain.png'.format(strat_name))
        plt.figure(figsize= (20,30))
        plt.subplot(121)
        plt.imshow(img1)
        plt.xticks([]) 
        plt.yticks([])
        plt.title("The most uncertain images", font = "Times", fontsize = 22)
        plt.subplot(122)
        plt.imshow(img2)
        plt.xticks([])
        plt.yticks([])
        plt.title("The least uncertain images", font = "Times", fontsize = 22)
        plt.grid = False
        plt.show()

    @staticmethod
    def imscatter(x, y, image, ax=None, zoom=1):
        """ Function to display PCA of image data

        Args:
            x ([type]): [description]
            y ([type]): [description]
            image ([type]): [description]
            ax ([type], optional): [description]. Defaults to None.
            zoom (int, optional): [description]. Defaults to 1.

        Returns:
            [type]: [description]
        """
        if ax is None:
            ax = plt.gca()
            ax.set_facecolor('xkcd:white')
            ax.grid(False)

        im = OffsetImage(image, zoom=zoom, alpha= 0.8)
        x, y = np.atleast_1d(x, y)
        artists = []
        for x0, y0 in zip(x, y):
            ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
            artists.append(ax.add_artist(ab))
        ax.update_datalim(np.column_stack([x, y]))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.autoscale()
        return artists
    
    
    @staticmethod
    def check_convergence_mlp(z_train: np.ndarray, y_train: np.array, z_test: np.ndarray, y_test: np.array, num_classes : int = 10,  nb_epochs: int = 100, batch_size:  int = 16, figsize: tuple = (30, 7)):
        inverse_mapping = {"MobileNetV1":0, "MobileNetV2":1, "ResNet50": 2}
        plt.figure(figsize = figsize)
        for n, embedding in enumerate(["MobileNetV1", "MobileNetV2", "ResNet50"]):
            plt.subplot(1,3,n+1)
            model = MLP(embedding = embedding, num_classes = num_classes, hidden_layer_sizes=(64,64,32))
            train_history =  model.model.fit(Data.select_embedding(z_train, inverse_mapping[embedding]), y_train,
                        batch_size=batch_size, epochs= nb_epochs,
                        verbose=0, validation_data=(Data.select_embedding(z_test, inverse_mapping[embedding]), y_test))
                
            loss = train_history.history['loss']
            accuracy = train_history.history['accuracy']
            val_accuracy = train_history.history['val_accuracy']
            plt.plot(loss)
            plt.plot(accuracy)
            plt.plot(val_accuracy)
            plt.xticks(fontsize = 22)
            plt.yticks(fontsize = 22)
            plt.xlabel("Optimization epoch", fontsize = 22)
            plt.ylabel("Accuracy", fontsize = 22)
            plt.title(embedding, fontsize = 36)
            
            plt.legend(['Loss', "Train Accuracy", "Validation Accuracy"], fontsize = 22)
        plt.show()
        
    
    @staticmethod
    def accuracies(models: pd.DataFrame, figsize: tuple = (25,10)):
        """ Plot accuracies computed among the benchmark

        Args:
            models (pd.DataFrame): Benchmark data frame with accuracies
        """
        models_sorted = models.sort_values("accuracy", ascending = False)
        indexes = np.arange(len(models_sorted))

        fig = plt.figure(figsize = figsize)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])


        for i, (name, row) in enumerate(models_sorted.iterrows()):
            if "MobileNetV1" in name: 
                ax.bar(i, row["accuracy"], width = 0.5, color = "b")
            elif "MobileNetV2" in name: 
                ax.bar(i, row["accuracy"], width = 0.5, color = "orange")
            else:
                ax.bar(i,row["accuracy"], width = 0.5, color = "green")

        ax.set_xticks(indexes, )
        ax.set_xticklabels(models_sorted.index.values, rotation = 90, fontsize = 22)
                


        plt.legend(handles = [Patch(facecolor='b',label='MobileNetV1'),
                    Patch(facecolor='orange', label='MobileNetV2'),
                    Patch(facecolor='green', label='ResNet50')], loc='best', fontsize = 22)

        plt.ylabel("Accuray", fontsize = 22)
        plt.yticks(fontsize = 22)
        plt.show()
    
    @staticmethod
    def auroc(res: dict, figsize: tuple = (25,10)):
        """ Plot accuracies computed among the benchmark

        Args:
            res (dict): Benchmark data frame with accuracies
        """
        df = pd.DataFrame.from_dict(res, orient= 'index',  columns = ["auroc"])
        models_sorted = df.sort_values("auroc", ascending = False)
        indexes = np.arange(len(models_sorted))

        fig = plt.figure(figsize = figsize)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])


        for i, (name, row) in enumerate(models_sorted.iterrows()):
            if "MobileNetV1" in name: 
                ax.bar(i, row["auroc"], width = 0.5, color = "b")
            elif "MobileNetV2" in name: 
                ax.bar(i, row["auroc"], width = 0.5, color = "orange")
            else:
                ax.bar(i,row["auroc"], width = 0.5, color = "green")

        ax.set_xticks(indexes, )
        ax.set_xticklabels(models_sorted.index.values, rotation = 90, fontsize = 22)
                


        plt.legend(handles = [Patch(facecolor='b',label='MobileNetV1'),
                    Patch(facecolor='orange', label='MobileNetV2'),
                    Patch(facecolor='green', label='ResNet50')], loc='best', fontsize = 22)

        plt.ylabel("AUROC", fontsize = 22)
        plt.yticks(fontsize = 22)
        plt.show()
        
    @staticmethod
    def histogram(uncertainties: pd.DataFrame, model_names: list= ["GaussianKernel-MobileNetV1", "DeepEnsemble-MobileNetV1-epochs=60 aleatoric entropy"],
                            figsize: tuple = (25,10)):
        """[summary]

        Args:
            uncertainties (pd.DataFrame): [description]
            figsize (tuple, optional): [description]. Defaults to (25,10).
        """
        mapping = {"GaussianKernel-MobileNetV1": "Gaussian Kernel with MobileNetV1", "DeepEnsemble-MobileNetV1-epochs=10 aleatoric entropy": "MLP Ensemble + entropy with MobileNetV1",
                   "GaussianKernel-ResNet50": "Gaussian Kernel with ResNet50", "GMM-ResNet50": "GMM with ResNet50", "DeepEnsemble-ResNet50-epochs=10 epistemic std": "MLP Ensemble + std on ResNet50"}
        import matplotlib
        matplotlib.use('cairo')
        import matplotlib.pylab as pylab
        import matplotlib.font_manager as fm
        font = fm.FontProperties(family = 'Gill Sans', fname = '/Library/Fonts/GillSans.ttc')  
        plt.figure(figsize = figsize)
        for i,name in enumerate(model_names):
            # print(int(len(model_names)/2))
            plt.subplot(int(len(model_names)/2)+1,2,i+1)
            sns.histplot(data=uncertainties.loc[uncertainties['UncertaintyLabel'] != ""], x=name, hue="UncertaintyLabel", bins = 50, legend = i)
            plt.ylim([0,20])
            plt.xlabel(mapping[name])

            # plt.ylim([0,50])
            
    @staticmethod
    def plot_curriculum(res: dict, budgets = [10],  nb_replications: int = 1, metrics: list= ["accuracy", "model change", "auroc"], figsize = (15,25)):
        """ Plot curriculum

        Args:
            res (dict): Result from the function PerformanceMetrics.curriculum
            budgets (list, optional): Amount of instances at each retraining. Defaults to [10].
            nb_replications (int, optional): Number of replications. Defaults to 1.
            metrics (list, optional): Metrics to display. Defaults to ["accuracy", "model change", "auroc"].
            figsize (tuple, optional): Size of the figure. Defaults to (15,25).
        """
        sns.set_theme(style = "darkgrid")
        
        overall_frame  = pd.DataFrame(columns = ["budget", "strategy"] +  metrics)
        nb_instances = len(budgets)
        for strategy in res.keys():
            for replication in range(nb_replications):
                tmp_frame = pd.DataFrame(columns = ["budget", "strategy"] +  metrics)
                for metric in metrics:
                    tmp_frame[metric] = pd.Series(res[strategy][metric][0][0][replication])
                tmp_frame["budget"] = pd.Series(budgets)
                tmp_frame["strategy"]  = pd.Series([strategy]* nb_instances)
                overall_frame = pd.concat([overall_frame, tmp_frame])
        
        overall_frame.reset_index()
        
        figsize = (figsize[0]  * len(metrics), figsize[1])
        plt.figure(figsize=figsize)
        for i, metric in enumerate(metrics):
            plt.subplot(len(metrics), 1, i+1)
            sns.lineplot(x="budget", y=metric,
                hue="strategy",
                data=overall_frame)
            plt.title(metric)
        plt.show()
        
    @staticmethod
    def survey_analysis(data: dict, answers: list = ["Pas du tout d'accord", "Plutôt pas d'accord", "Neutre", "Plutôt d'accord", "Tout à fait d'accord"],
                        participants: list = [4,4], conditions: list = ["epistemic", "aleatoric"], figsize = (10,40), by_order = False):
        tmp = []
        for p in participants:
            for c in conditions:
                if len(data[p][c]["survey"]["question"]) != 0:
                    tmp.append(p)
                else:
                    "Data missing: skip participant {p} and condition {c} "
        participants = tmp
        if not by_order:
            for i, c in enumerate(conditions):
                questions = data[participants[0]][c]["survey"]["question"]
                overall_frame = pd.DataFrame(columns = questions)
                for k,p in enumerate(participants):
                    overall_frame.loc[p] = data[p][c]["survey"]["answer"].values
            
                overall_frame.reset_index()
                plot_likert.plot_likert(overall_frame, answers, figsize = figsize)
                plt.title(c.capitalize())
        else:
            for i in [0,1]:
                questions = data[participants[0]]["aleatoric"]["survey"]["question"]
                overall_frame = pd.DataFrame(columns = questions)
                for k,p in enumerate(participants):
                    
                    if p % 2 == i:
                        overall_frame.loc[p] = data[p]["aleatoric"]["survey"]["answer"].values
                    else:
                        overall_frame.loc[p] = data[p]["epistemic"]["survey"]["answer"].values
            
                overall_frame.reset_index()
                plot_likert.plot_likert(overall_frame, answers, figsize = figsize)
                plt.title("Iteration {}".format(i+1))
                
        return overall_frame
    @staticmethod
    def test_accuracy_analysis(data: dict, participants: list = [3,4], conditions: list = ["aleatoric", "epistemic"], log_scale = False, figsize: tuple = (10,5), by_order = False):
        sns.set_theme(style="darkgrid")
        overall_frame  = pd.DataFrame(columns=["participant", "condition", "score"])
        if not by_order:
            for i,p in enumerate(participants):
                for j,c in enumerate(conditions):
                    score = np.sum([1 if res == "right" else (0 if res == "dont_know" else -1 ) for res in data[p][c]["test_accuracy"]["result"]])
                    overall_frame.loc[i*len(conditions)+j] = [p,c.capitalize(),score]
            overall_frame["condition"] = overall_frame["condition"].map({"Aleatoric":"Data uncertainty", "Epistemic":"Model uncertainty"})
        else:
            for i,p in enumerate(participants):
                for j in [0,1]:
                    if p % 2 == j:
                        score = np.sum([1 if res == "right" else (0 if res == "dont_know" else -1 ) for res in data[p]["aleatoric"]["test_accuracy"]["result"]])
                    else:
                        score = np.sum([1 if res == "right" else (0 if res == "dont_know" else -1 ) for res in data[p]["epistemic"]["test_accuracy"]["result"]])
                    overall_frame.loc[i*len(conditions)+j] = [p,str(j+1),score]
        
        f, ax = plt.subplots(figsize=figsize)
        sns.boxplot(x="score", y="condition", data=overall_frame,
            whis=[0, 100], width=.6, palette="pastel")
        sns.stripplot(x="score", y="condition", data=overall_frame,
              size=4, color=".3", linewidth=0)
        if log_scale:
            plt.gca().set_xscale('log')
        ax.xaxis.grid(True)
        ax.set(ylabel="")
        ax.set(xlabel= "Score")
        plt.title("Prediction scores on the classification of new images")
        sns.despine(trim=True, left=True)
        
    @staticmethod
    def test_uncertainty_analysis(data: dict, participants: list = [3,4], conditions: list = ["aleatoric", "epistemic"], log_scale = False, figsize=(10,5), by_order = False):
        sns.set_theme(style="darkgrid")
        overall_frame  = pd.DataFrame(columns=["participant","condition", "score"])
        
        if not by_order:
            for i,p in enumerate(participants):
                for j,c in enumerate(conditions):
                    score = np.linalg.norm(np.clip(data[p][c]["test_uncertainty"]["predicted"].values, 0, 1)-data[p][c]["test_uncertainty"]["answer"].values)
                    overall_frame.loc[i*len(conditions)+j] = [p,c.capitalize(),score]
            overall_frame["condition"] = overall_frame["condition"].map({"Aleatoric":"Data uncertainty", "Epistemic":"Model uncertainty"})
        else:
            for i,p in enumerate(participants):
                for j in [0,1]:
                    if p % 2 == j:
                        score = np.linalg.norm(np.clip(data[p]["aleatoric"]["test_uncertainty"]["predicted"].values, 0, 1)-data[p]["aleatoric"]["test_uncertainty"]["answer"].values)
                    else:
                        score = np.linalg.norm(np.clip(data[p]["epistemic"]["test_uncertainty"]["predicted"].values, 0, 1)-data[p]["epistemic"]["test_uncertainty"]["answer"].values)
                    overall_frame.loc[i*len(conditions)+j] = [p,str(j+1),score]
            
        f, ax = plt.subplots(figsize=figsize)
        sns.boxplot(x="score", y="condition", data=overall_frame,
            whis=[0, 100], width=.6, palette="pastel")
        sns.stripplot(x="score", y="condition", data=overall_frame,
              size=4, color=".3", linewidth=0)
        if log_scale:
            plt.gca().set_xscale('log')
        ax.xaxis.grid(True)
        ax.set(ylabel="")
        ax.set(xlabel= "Error")
        plt.title("Prediction error on the uncertainty estimation on new images")
        sns.despine(trim=True, left=True)
        
    @staticmethod
    def save_embedding_accuracies():
        embeddings_accuracies = pd.DataFrame(columns = ["Dataset", "Embedding", "Classifier", "Accuracy"])
        
        
        # CF results benchmark on MNIST + CARDS
        embeddings_accuracies.loc[0] = ["MNIST", "MobileNetV1", "GMM", 0.095]
        embeddings_accuracies.loc[1] = ["MNIST", "MobileNetV2", "GMM", 0.135]
        embeddings_accuracies.loc[2] = ["MNIST", "ResNet50", "GMM", 0.145]
        embeddings_accuracies.loc[4] = ["MNIST", "MobileNetV1", "MLP", 0.305]
        embeddings_accuracies.loc[6] = ["MNIST", "MobileNetV2", "MLP", 0.255]
        embeddings_accuracies.loc[8] = ["MNIST", "ResNet50", "MLP", 0.685]

        embeddings_accuracies.loc[9] = ["Playing cards", "MobileNetV1", "GMM", 0.333]
        embeddings_accuracies.loc[10] = ["Playing cards", "MobileNetV2", "GMM", 0.593]
        embeddings_accuracies.loc[11] = ["Playing cards", "ResNet50", "GMM", 0.354]
        embeddings_accuracies.loc[13] = ["Playing cards", "MobileNetV1", "MLP", 0.920]
        embeddings_accuracies.loc[15] = ["Playing cards", "MobileNetV2", "MLP", 0.933]
        embeddings_accuracies.loc[17] = ["Playing cards", "ResNet50", "MLP", 0.886]


        import seaborn as sns
        sns.set_theme(style="whitegrid")
        fig = plt.figure(figsize=(10,15))
        g = sns.catplot(x="Classifier", y="Accuracy",

                        hue="Embedding", col="Dataset",

                        data=embeddings_accuracies, kind="bar",

                        height=4, aspect=.7, legend= True, legend_out = False, palette=sns.color_palette("colorblind"))
        g.set(ylim=(0, 1))
        # plt.tight_layout()
        fig.savefig("../../figures_paper/embedding_accuracy.pdf")
        return fig
        
    @staticmethod
    def save_uncertain_instances_paper():
        strat_name_1 = "GaussianKernel-MobileNetV1"
        strat_name_2 = "DeepEnsemble-MobileNetV1-epochs=10 aleatoric entropy"
        if strat_name_1 not in ["GroundTruth", "UncertaintyLabel", "Images"]:
            fig1 = Figures.display_most_uncertain_images(uncertainty_scaled, strat_name_1, nb_images = 10, nb_columns = 5)
            fig1.savefig("../../figures_paper/benchmark_model_uncertainty.pdf", bbox_inches = "tight")
            
        if strat_name_2 not in ["GroundTruth", "UncertaintyLabel", "Images"]:
            fig2 = Figures.display_most_uncertain_images(uncertainty_scaled, strat_name_2, nb_images = 10, nb_columns = 5)
            fig2.savefig("../../figures_paper/benchmark_data_uncertainty.pdf", bbox_inches = "tight")
            fname = open("images/{0}_{1}".format(strat_name, "uncertain.png"), "wb")
            fig.savefig(fname)
                
            
    @staticmethod
    def save_auroc_paper(type_uncertainty: str = "Aleatoric uncertainty"):
        import pickle
        auroc_mnist = pickle.load(open("../benchmark/auroc_mnist_b.pkl", "rb"))
        auroc_cards = pickle.load(open("../benchmark/auroc_cards_b.pkl", "rb"))
        AUROC = pd.DataFrame(columns=["Dataset", "Type of uncertainty", "Embedding", "Model", "Acquisition  function", "AUROC"])
        
        cpt = 0
        for k in auroc_mnist["samples"].keys():
            row = [None]*6
            if "MobileNetV1" in k:
                row[2] = "MobileNetV1"
            elif "MobileNetV2" in k:
                row[2] = "MobileNetV2"
            else:
                row[2] = "ResNet50"
            if "GMM" in k:
                row[1] = "Epistemic uncertainty"
                row[3] = "GMM"
                row[4] = "Log-Likelihood"
            elif "Kernel" in k:
                row[1] = "Epistemic uncertainty"
                row[3] = "Gaussian Kernel"
                row[4] = "Density estimation"
            elif "MLP" in k:
                row[1] = "Aleatoric uncertainty"
                row[3] = "Simple MLP"
                row[4] = "Shannon entropy"                
                
            else:
                if "entropy" in k:
                    row[1] = "Aleatoric uncertainty"
                    row[3] = "MLP Ensemble"
                    row[4] = "Shannon entropy"   
                else:
                    row[1] = "Epistemic uncertainty"
                    row[3] = "MLP Ensemble + std"
                    row[4] = "Standard deviation"  
            for sample in auroc_mnist["samples"][k]:
                row[0] = "MNIST"
                row[5] = sample
                AUROC.loc[cpt] = list(row)
                cpt+=1
            for sample in auroc_cards["samples"][k]:
                row[0] = "Playing cards"
                row[5] = sample
                AUROC.loc[cpt] = row
                cpt+=1

        import seaborn as sns

        # AUROC[['Model', 'Dataset', 'Embedding', 'Type of uncertainty']] = AUROC[['Model', 'Dataset', 'Embedding', 'Type of uncertainty']].astype("category")
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(15,15))
        g = sns.catplot(x="Model", y="AUROC",

                        hue="Embedding", col="Dataset",

                        data=AUROC.loc[AUROC["Type of uncertainty"] == type_uncertainty], kind="bar",

                        height=4, aspect=.7, legend = False, legend_out = True, palette=sns.color_palette("colorblind"))
        g.set(ylim=(0, 1))
        g.set_xticklabels(rotation = 15)
        g.axes[0][0].axhline(y = 0.5, color = 'k', linestyle = '--', linewidth = 1)
        g.axes[0][1].axhline(y = 0.5, color = 'k', linestyle = '--', linewidth = 1)

        plt.savefig("../../figures_paper/embedding_detection_{}.pdf".format(type_uncertainty), bbox_inches = "tight")
        return g

    
        
        