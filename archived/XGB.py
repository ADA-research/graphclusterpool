import torch
import numpy as np
import random
import sklearn
from xgboost import XGBClassifier

from ThesisModel import ThesisModelInterface

class XGBoost(ThesisModelInterface):
    def __init__(self, data, labels, test_set_idx):
        super().__init__(data, labels, test_set_idx)
        self.clfName = "XGBoost Classifier"
        self.n_labels = len(labels)

    def format_data_values(self, validation):

        def extract_nodes(pdata):

            dset = None
            labels = None
            for patient in pdata:
                
                nodes, edges, classes, pname = patient
                if dset is None:
                    dset = nodes.detach().numpy()
                    labels = classes.detach().numpy()
                else:
                    dset = np.append(dset, nodes.detach().numpy(), axis=0)
                    labels = np.append(labels, classes.detach().numpy(), axis=0)
                
            return np.array(dset), np.array(labels)
        
        self.x_train, self.y_train = extract_nodes(self.train)
        if validation: self.x_valid, self.y_valid = extract_nodes(self.valid)
        self.x_test, self.y_test = extract_nodes(self.test)

    def train_model(self, replace_model=True):
        "Function to fit the model to the data"
        if self.clf is None or replace_model is True:
            obj = 'binary:logistic'
            e_metric = 'aucpr'

            self.clf = XGBClassifier(objective=obj, eta=0.3, n_estimators=100, max_depth=1000, eval_metric=e_metric, seed=random.randint(0, self.maxSeed), use_label_encoder=False)#, scale_pos_weight=4) #, tree_method='gpu_hist', gpu_id=0)

        
        self.clf.fit(self.x_train, self.y_train, eval_set=[(self.x_valid, self.y_valid)], verbose=False)
        

    def validate_model(self):
        
        self.y_valid_pred = self.clf.predict(self.x_valid)
        self.y_valid_dist = self.clf.predict_proba(self.x_valid)
        
        if self.bnry:
            res = []
            for i, e in enumerate(self.y_valid_dist):
                res.append(e[1])
            self.y_valid_dist = res

        prec, rec, threshold =  sklearn.metrics.precision_recall_curve(self.y_valid, self.y_valid_dist)
        if not ((prec+rec) == 0).any():
            f1s = (2*(prec * rec)) / (prec + rec)
            self.threshold = threshold[np.argmax(f1s)]
            print("Updated threshold to: ", self.threshold)
        

    def test_model(self):
        "Function that calculates test set classifications"
        self.y_test_pred = self.clf.predict(self.x_test)
        self.y_test_dist = self.clf.predict_proba(self.x_test)
        prob_var = []
        if self.bnry:
            res = []
            for i, e in enumerate(self.y_test_dist):
                res.append(e[1])
            self.y_test_dist = res
    
        off_set = 0

        for patient in self.test:
            psize = len(patient[2])
            prob_var.extend(self.calculate_prediction_variance(patient[0], patient[1], torch.tensor(self.y_test_dist[off_set:off_set+psize]),patient[2]))
            off_set += psize
        return prob_var

    def calculate_feature_importances(self):
        return self.clf.feature_importances_
