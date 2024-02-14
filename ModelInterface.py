import time
import random
import torch
import pickle
from datetime import datetime
import sys
from pathlib import Path

import numpy as np
import math
from sklearn import metrics

"Model Interface, few base definitions that each model needs"
class ModelInterface:
    "Internal data object. Is in general a two dimensional list of self.data[patient][tensor]"
    "Each patient has three tensors and a string:"
    "[0]: Node tensor, containing each node and its features"
    "[1]: Edge tensor"
    "[2]: Node label tensor, containing the class of each node"

    def __init__(self, data, labels, test_set_idx, seed = None):
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

        self.randomSeed = seed
        if seed is None:
            self.randomSeed = random.randint(0, 4294967295)

        self.threshold = 0.5 #Standard threshold for binary classifications

        self.train = []
        self.valid = []

    def generate_train_validation_test(self, split_train=0.8, split_validation=0.1, shuffle_for=1):
        "Creates a random train/validation/test split from the internal data object"
        for _ in range(shuffle_for):
            random.Random(self.randomSeed).shuffle(self.data)
        train_index = int(len(self.data) * split_train)
        validation_index = train_index + int(len(self.data) * split_validation)
        self.train = self.data[:train_index]
        self.valid = self.data[train_index:validation_index]
        self.test = self.data[validation_index:]
        
        #If the presented data needs some modification, the following function can be overwritten
        self.format_data_values()

    "Finishes data set formatting after generating train/test sets. Overwrite if no Tensors are used."
    def format_data_values(self):
        "Send the tensors to the correct device"
        for i in range(len(self.train)):
            self.train[i][0] = self.train[i][0].to(self.device) #Send nodes
            self.train[i][1] = self.train[i][1].to(self.device) #Send edges
            self.train[i][2] = self.train[i][2].to(self.device) #Send labels
        
        for i in range(len(self.valid)):
            self.valid[i][0] = self.valid[i][0].to(self.device) #Send nodes
            self.valid[i][1] = self.valid[i][1].to(self.device) #Send edges
            self.valid[i][2] = self.valid[i][2].to(self.device) #Send labels

        for i in range(len(self.test)):
            self.test[i][0] = self.test[i][0].to(self.device) #Send nodes
            self.test[i][1] = self.test[i][1].to(self.device) #Send edges
            self.test[i][2] = self.test[i][2].to(self.device) #Send labels

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
        "Function to validate a model on the validation set"
        self.y_valid_pred = []
        self.y_valid_dist = []
        
        for data in self.valid: #For every graph in the data set
            out = self.clf(data) #Get the labels from all the nodes in one graph 

            if type(out) == tuple:
                out = out[0]

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
        #if self.bnry:
        #    return metrics.precision_recall_fscore_support(y_actual, y_prediction, average=None, labels=self.labels)
        #else:
        return metrics.accuracy_score(y_actual, y_prediction)

    def get_valid_preds(self):
        self.validate_model()
        return self.y_valid, self.y_valid_dist
    
    def save_results(self, filepath, folds, kCross, train_loss_f, train_acc_f, validation_loss_f, validation_acc_f, model):
        if not Path(filepath).exists(): # Create the empty-ish dict
            if not Path(filepath).parent.exists():
                Path(filepath).parent.mkdir(parents=True)
            
            with Path(filepath).open('wb') as fileobj:
                resdict = {"description": [self.clfName, str(self.clf.poolLayer), f"Layer width {self.clf.hid_channel}", folds, kCross, self.randomSeed],
                        "train_loss_folds": [],
                        "train_acc_folds": [],
                        "validation_loss_folds": [],
                        "validation_acc_folds": [],
                        "models": []}
                pickle.dump(resdict, fileobj)

        resdict = {}
        with open(filepath, 'rb+') as handle:
            resdict = pickle.load(handle)
            resdict["train_loss_folds"].append(train_loss_f)
            resdict["train_acc_folds"].append(train_acc_f)
            resdict["validation_loss_folds"].append(validation_loss_f)
            resdict["validation_acc_folds"].append(validation_acc_f)
            resdict["models"].append(model)
        with open(filepath, 'wb') as handle:
            pickle.dump(resdict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def run_folds(self, folds, display=True, contExperimentFile=None, seed=None, iteration_id=None):
        "Function that runs train and test for models"
        kCross = False # Old variable
        start_fold = 0
        timestamp = datetime.now().strftime('%Y-%m-%d %H.%M.%S')
        dirname = f"results/{self.clfName} {timestamp}"
        filepath = f"{dirname}/result_dictionary.pkl"
        if seed is not None:
            self.randomSeed = seed

        if contExperimentFile is str and iteration_id is None:
            print(f"Continuing experiment from {contExperimentFile}."
                  "\nWARNING: Any change to the architecture or Hyperparameters will render this experiment useless.", flush=True)
            dirname = contExperimentFile
            filepath = contExperimentFile
            if dirname.endswith(".pkl"):
                dirname = "/".join(dirname.split("/")[:-1])
            else:
                filepath += "/".join(dirname.split("/")) + "result_dictionary.pkl"
            
            if not Path(filepath).is_file():
                print(f"Incorrect file for continuation: {filepath}")
                sys.exit(-1)
            
            data = None
            with open(filepath, 'rb') as pkl:
                data = pickle.load(pkl)

            # clfName, poolLayer, widthString = data["description"][0], data["description"][1], data["description"][2]
            folds = data["description"][3]
            start_fold = len(data["train_loss_folds"])
            timestamp = None
        elif iteration_id is not None: #
            filepath = contExperimentFile
        else:
            Path(dirname).mkdir(parents=True)

        if display:
            print("\nRunning " + str(folds-start_fold) + " folds with " + str(self.clfName) + ":", flush=True)

        run_folds = folds
        if iteration_id is not None: # We will only run one fold
            start_fold = iteration_id
            run_folds = start_fold + 1
            # Scroll through the seed to the current state
            self.generate_train_validation_test(shuffle_for=start_fold)

        for i in range(start_fold, run_folds):
            if display:
                start_time = time.time()
                print("\tFold " + str(i+1) + "/" + str(folds) + "...")

            self.generate_train_validation_test()

            t_acc, t_loss, v_acc, v_loss, model = self.train_model(verbose=False)
            self.save_results(filepath, folds, kCross, t_loss, t_acc, v_loss, v_acc, model)

            if display:
                elapsed_time = time.time() - start_time
                elapsed_minutes = int( elapsed_time / 60.0 )
                elapsed_seconds = elapsed_time % 60   
                print(f"\t\t Fold {i} completed. ({elapsed_minutes} m {elapsed_seconds} s)", flush=True)
                
                mtacc, mvacc = np.max(t_acc), np.max(v_acc)
                mtloss, mvloss = np.min(t_loss), np.min(v_loss)
                print(f"\t\t{mtacc:.4f} Best Train Accuracy, {mvacc:.4f} Best Validation Accuracy.", flush=True)
                print(f"\t\t{mtloss:.4f} Lowest Train Loss, {mvloss:.4f} Lowest Validation Loss", flush=True)

