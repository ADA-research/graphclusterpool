import time
import random
import torch
import pickle
from datetime import datetime
import os
import sys

import numpy as np
import math
from sklearn import metrics

from matplotlib import pyplot as plt
import seaborn as sns

"Model Interface, few base definitions that each model needs"
class ModelInterface:
    "Internal data object. Is in general a two dimensional list of self.data[patient][tensor]"
    "Each patient has three tensors and a string:"
    "[0]: Node tensor, containing each node and its features"
    "[1]: Edge tensor"
    "[2]: Node label tensor, containing the class of each node"

    def __init__(self, data, labels, test_set_idx):
        "Receives data from controller"
        self.test = [e for i,e in enumerate(data) if i in test_set_idx]
        self.data = [e for i,e in enumerate(data) if i not in test_set_idx]

        self.labels = labels
        self.n_labels = len(labels)
        self.bnry = (self.n_labels == 2)
        self.MetricName = "F1-Score" if self.bnry else "Accuracy"
        
        
        self.clf = None
        self.clfName = "ModelInterface"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print("Selected Device: ", self.device)

        self.maxSeed = 4294967295 #Maximum value in 32 bits
        self.randomSeed = random.randint(0, self.maxSeed)
        random.Random(self.randomSeed).shuffle(self.data)
        
        self.threshold = 0.5 #Standard threshold for binary classifications

        self.cross_test_size = -1
        self.train = []
        self.valid = []

        self.generate_train_validation(validation=True)

    def generate_train_validation(self, split=0.89, validation=False):
        "Creates a train/validation/test split from the internal data object"

        train_index = int(len(self.data) * split)
        self.train = self.data[:train_index]
        if validation:
            #size = (1 - split) / 2
            #valid_index = int(len(self.data) * (split + size))
            self.valid = self.data[train_index:]
            
        else:
            self.train = self.data

        #If the presented data needs some modification, the following function can be overwritten
        self.format_data_values(validation=validation)

    def generate_k_fold(self, folds):
        "Creates a train/validation/test split based on the number of folds. Returns the number of folds possible."       
        self.cross_test_size = math.ceil(len(self.data)/ folds)
        return math.ceil(len(self.data) / self.cross_test_size)

    "Finishes data set formatting after generating train/test sets. Overwrite if no Tensors are used."
    def format_data_values(self, validation=False):
        "Send the tensors to the correct device"
        for i in range(len(self.train)):
            self.train[i][0] = self.train[i][0].to(self.device) #Send nodes
            self.train[i][1] = self.train[i][1].to(self.device) #Send edges
            self.train[i][2] = self.train[i][2].to(self.device) #Send node labels
        
        for i in range(len(self.valid)):
            self.valid[i][0] = self.valid[i][0].to(self.device) #Send nodes
            self.valid[i][1] = self.valid[i][1].to(self.device) #Send edges
            self.valid[i][2] = self.valid[i][2].to(self.device) #Send node labels

        for i in range(len(self.test)):
            self.test[i][0] = self.test[i][0].to(self.device) #Send nodes
            self.test[i][1] = self.test[i][1].to(self.device) #Send edges
            self.test[i][2] = self.test[i][2].to(self.device) #Send node labels

        "Extracts the x and/or y from the train/validation/test sets"
        self.y_train = []
        for patient in self.train:
            self.y_train.extend(patient[2].cpu().numpy().tolist())

        self.y_valid = []
        for patient in self.valid:
            self.y_valid.extend(patient[2].cpu().numpy().tolist())

        self.y_test = []
        for patient in self.test:
            self.y_test.extend(patient[2].cpu().numpy().tolist())

    def train_model(self, replace_model=True, verbose=True):
        "Function to fit the model to the train set"
        pass

    def validate_model(self):
        
        self.y_valid_pred = []
        self.y_valid_dist = []
        
        for data in self.valid: #For every graph in the data set
            out = self.clf(data) #Get the labels from all the nodes in one graph 

            if type(out) == tuple: out = out[0]

            labels = ((out > self.threshold).int()).cpu().detach().numpy()

            self.y_valid_dist.extend(out.cpu().detach().numpy().tolist())
            self.y_valid_pred.extend(labels.tolist())
        return metrics.f1_score(self.y_valid, self.y_valid_pred)

    def test_model(self):
        "Function that calculates labelling results on the test set"
        self.y_test_pred = []
        self.y_test_dist = []
        vr = []
        for data in self.test: #For every graph in the data set
            out = self.clf(data) #Get the labels from all the nodes in one graph (Each node gets 12 outputs: one for each class)
            labels = ((out > self.threshold).int()).cpu().detach().numpy()
            vr.extend(self.calculate_prediction_variance(data[0], data[1], out, data[2])) #Move this one to test_model
            self.y_test_dist.extend(out.cpu().detach().numpy().tolist())
            self.y_test_pred.extend(labels.tolist())
        return vr
    
    """The calculation of the metrics based on the labels and prediction"""
    def metric(self, y_actual, y_prediction):
        if self.bnry:
            return metrics.precision_recall_fscore_support(y_actual, y_prediction, average=None, labels=self.labels)
        else:
            return metrics.accuracy_score(y_actual, y_prediction)
    
    def calculate_test_metrics(self):
        #self.correct_test_labels()
        precision, recall, f, _ = metrics.precision_recall_fscore_support(self.y_test, self.y_test_pred, average=None, labels=self.labels)
        p,r, t = metrics.precision_recall_curve(self.y_test, self.y_test_dist)
        f1s = (2*(p * r)) / (p + r)
        #if self.bnry: #Binary, return positive F1 score
        return f[1], precision, recall, np.max(f1s)
        #return np.average(f), precision, recall

    def get_valid_preds(self):
        self.validate_model()
        return self.y_valid, self.y_valid_dist

    "(int) folds: Number of folds. If k-cross, then folds <= len(data)"
    "(Boolean) kCross: Use k-fold-cross-validation"
    "(Boolean) display: Print statistics"
    "(Boolean) record_roc: Save and return test probabilities and values for ROC curve"
    def run_folds(self, folds, kCross=True, display=True, validation=True, contExperiment=None):
        "Function that runs train and test for models"
        train_acc_f = []
        train_loss_f = []
        validation_acc_f = []
        validation_loss_f = []
        models = []
        start_fold = 0
        timestamp = datetime.now().strftime('%Y-%m-%d %H.%M.%S')
        dirname = f"results/{self.clfName} {timestamp}"
        filepath = f"{dirname}/result_dictionary.pkl"

        if contExperiment is str:
            print(f"Continuing experiment from {contExperiment}."
                  "\nWARNING: Any change to the architecture or Hyperparameters will render this experiment useless.")
            dirname = contExperiment
            filepath = contExperiment
            if dirname.endswith(".pkl"):
                dirname = "/".join(dirname.split("/")[:-1])
            else:
                filepath += "/".join(dirname.split("/")) + "result_dictionary.pkl"
            
            if not os.path.isfile(filepath):
                print(f"Incorrect file for continuation: {filepath}")
                sys.exit(-1)
            
            data = None
            with open(filepath, 'rb') as pkl:
                data = pickle.load(pkl)

            # clfName, poolLayer, widthString = data["description"][0], data["description"][1], data["description"][2]
            folds = data["description"][3]
            train_loss_f, train_acc_f = data["train_loss_folds"], data["train_acc_folds"]
            validation_loss_f, validation_acc_f = data["validation_loss_folds"], data["validation_acc_folds"]
            start_fold = len(train_loss_f)
            timestamp = None
        else:
            os.mkdir(dirname)

        if not kCross:
            self.generate_train_validation(validation=True)
        elif self.cross_test_size <= 0: #Have to create a cross validation
            if folds > len(self.data):
                folds = len(self.data)
            folds = self.generate_k_fold(folds)

        if display:
            print("\nRunning " + str(folds-start_fold) + " folds with " + str(self.clfName) + ":")
        
        for i in range(start_fold, folds):
            if display:
                start_time = time.time()
                print("\tFold " + str(i+1) + "/" + str(folds) + "...", end='')

            if not kCross: #Generate new sets
                self.generate_train_validation(validation=validation)
            else: #Shift the sets one up
                tindex = self.cross_test_size * (i)
                tend = min(tindex + self.cross_test_size, len(self.data))
                self.valid = self.data[tindex:tend]
                self.train = self.data[:tindex] + self.data[tend:]
                if len(self.train) == 0:
                    self.train = self.data[tindex:tend]
                    self.valid = []
                self.format_data_values(validation=True)

            t_acc, t_loss, v_acc, v_loss, model = self.train_model(verbose=False)

            train_acc_f.append(t_acc)
            train_loss_f.append(t_loss)
            validation_acc_f.append(v_acc)
            validation_loss_f.append(v_loss)
            models.append(model)

            #Save data
            resdict = {
                "description": [self.clfName, str(self.clf.poolLayer), f"Layer width {self.clf.hid_channel}", folds],
                "train_loss_folds": train_loss_f,
                "train_acc_folds": train_acc_f,
                "validation_loss_folds": validation_loss_f,
                "validation_acc_folds": validation_acc_f,
                "models": models
            }
            with open(filepath, 'wb') as handle:
                pickle.dump(resdict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if display:
                elapsed_time = time.time() - start_time
                elapsed_minutes = int( elapsed_time / 60.0 )
                elapsed_seconds = elapsed_time % 60   
                print(f"\t\t Fold {i} completed. ({elapsed_minutes} m {elapsed_seconds} s)")
                
                mtacc = np.max(t_acc)
                mvacc = np.max(v_acc)
                mtloss = np.min(t_loss)
                mvloss = np.min(v_loss)

                print(f"\t\t{mtacc:.4f} Best Train Accuracy, {mvacc:.4f} Best Validation Accuracy.")
                print(f"\t\t{mtloss:.4f} Lowest Train Loss, {mvloss:.4f} Lowest Validation Loss")

        def plot_line_with_std(ax, mean, std, color):
            lower_bound = [M_new - Sigma for M_new, Sigma in zip(mean, std)]
            upper_bound = [M_new + Sigma for M_new, Sigma in zip(mean, std)]
            sns.lineplot(mean, ax=ax, color=color)
            ax.fill_between([x for x in range(len(lower_bound) )], lower_bound, upper_bound, color=color, alpha=.3)
        
        

        if display:
            sns.set()
            sns.set_style("darkgrid")
            fig, axes = plt.subplots(2, 2)
            fig.tight_layout(pad=1.0)
            fig.suptitle(f"Training metrics {self.clfName} ({str(self.clf.poolLayer)})")
            fig.subplots_adjust(top=0.9)

            tloss = np.mean(train_loss_f, axis=0)
            vloss = np.mean(validation_loss_f, axis=0)
            tacc = np.mean(train_acc_f, axis=0)
            vacc = np.mean(validation_acc_f, axis=0)

            min_tloss = np.min(train_loss_f, axis=1)
            min_vloss = np.min(validation_loss_f, axis=1)
            avg_min_t_loss = np.mean(min_tloss)
            avg_min_v_loss = np.mean(min_vloss)
            std_min_t_loss = np.std(min_tloss)
            std_min_v_loss = np.std(min_vloss)

            max_tacc = np.max(train_acc_f, axis=1)
            max_vacc = np.max(validation_acc_f, axis=1)
            avg_max_t_acc = np.mean(max_tacc)
            avg_max_v_acc = np.mean(max_vacc)
            std_max_t_acc = np.std(max_tacc)
            std_max_v_acc = np.std(max_vacc)

            axes[0][0].set_title(f"Training loss ({avg_min_t_loss:.4f} ± {std_min_t_loss:.4f})")
            axes[0][1].set_title(f"Validation loss ({avg_min_v_loss:.4f} ± {std_min_v_loss:.4f})")
            axes[1][0].set_title(f"Training Accuracy ({avg_max_t_acc:.4f} ± {std_max_t_acc:.4f})")
            axes[1][1].set_title(f"Validation Accuracy ({avg_max_v_acc:.4f} ± {std_max_v_acc:.4f})")


            plot_line_with_std(axes[0][0], tloss, np.std(train_loss_f, axis=0), color="red")
            plot_line_with_std(axes[0][1], vloss, np.std(validation_loss_f, axis=0), color="red")
            plot_line_with_std(axes[1][0], tacc, np.std(train_acc_f, axis=0), color="blue")
            plot_line_with_std(axes[1][1], vacc, np.std(validation_acc_f, axis=0), color="blue")
    
            plt.show()
            