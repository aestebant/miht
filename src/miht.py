import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import mode
from sklearn.metrics import accuracy_score
import copy
from river.tree.hoeffding_tree import HoeffdingTree
from river.tree import HoeffdingTreeClassifier


class MultiInstanceHoeffdingTreeClassifier():
    """Multi-Instance online classifier.

    Parameters
    ----------
    grace_period
        Number of instances a leaf should observe between split attempts.
    delta
        Significance level to calculate the Hoeffding bound. The significance level is given by 1 - delta. Values closer to zero imply longer split decision delays.
    mil_assumption
        Multi-instance learning assumption to model the relationship between instances in a bag.
        - 'max' - The bag label is the maximum of the instances labels.
        - 'mean' - The bag label is the mean average of the instances labels.
        - 'mode' - The bag label is the most repeated instance label.
    inst_len
        Length with which to construct the instances that will compose the bag oa time series. If it is a integer, it will be interpreted as time steps. If it a float in [0, 1] it will be interpreted as a percentage of the total time series.
    inst_stride
        Displacement between the start of a instance and the next one (the opposite to the overlap between instances). If it is a integer, it will be interpreted as time steps. If it a float in [0, 1] it will be interpreted as a percentage of the instance length.
    k
        Number of consecutive instances in a bag that should contain the concept of the series.
    max_it
        Hard limit in the number of iterations to stop the optimization process of the k best instances of the bag.
    max_patience
        Early stopping mechanism to stop the optimization process if the accuracy does not improve for this number of iterations.
    """
    def __init__(self, grace_period=500, delta=8.02e-4, mil_assumption='mode', inst_len=0.2, inst_stride=0.8, k=5, max_it=30, max_patience=5) -> None:
        self.mil_assumption = mil_assumption
        self.inst_len = inst_len
        self.inst_stride = inst_stride
        self.k = k
        self.max_patience = max_patience
        self.max_it = max_it
        self.ref_online_learner = HoeffdingTreeClassifier(grace_period=grace_period, delta=delta)


    def fit(self, X: pd.MultiIndex, y: np.ndarray):
        """Train the model on a dataset of time series in pandas multi-index format and corresponding targets y.

        Parameters
        ----------
        X
            Time series dataset. It should be pandas.MultiIndex dataframe of shape [n_timeseries, n_dimensions, equal or unequal series length].
        y
            The target values (class labels) corresponding to the time series, as integers or strings. It should be a numpy.array of shape [n_timeseries].

        Returns
        -------
        [acc_hist, best_k]
            The evolution of the accuracy during the training process and the index of the instances best k instances of each bag.
        """
        if 0 <= self.inst_len <= 1:
            self.inst_len = round(np.mean(X.groupby(level=0).size()) * self.inst_len)
        if 0 <= self.inst_stride <= 1:
            self.inst_stride = round(self.inst_len * self.inst_stride)

        self.online_learner = copy.deepcopy(self.ref_online_learner)

        X_bags = list()
        # First training with all the instances
        for i, seq in X.groupby(level=0):
            np_seq = seq.to_numpy()
            actual_len = min(self.inst_len, len(np_seq))
            roll_win = sliding_window_view(np_seq, window_shape=actual_len, axis=0)[::self.inst_stride]
            X_bags.append(roll_win)
            for instance in roll_win:
                for moment in instance.transpose():
                    moment_dict = dict(zip(X.columns, moment))
                    self.online_learner.learn_one(moment_dict, y[i])
        # Preparing for comparing the evolution of the model during convergence
        backup = copy.deepcopy(self.online_learner)
        y_pred = self._predict(X.columns, X_bags)
        best_acc = accuracy_score(y_true=y, y_pred=y_pred)
        result = {
            'selection': None,
            'acc_hist': [best_acc]
        }

        # Convergence process for finding signature of each time series
        curr_it = 0
        curr_pat = self.max_patience
        while curr_it < self.max_it and curr_pat > 0:
            # Selecting k consecutive best instanes per bag
            selection = list()
            for i, bag in enumerate(X_bags):
                instances_prob = np.zeros(len(bag))
                actual_k = min(self.k, len(bag))
                for j, instance in enumerate(bag):
                    for moment in instance.transpose():
                        instances_prob[j] += self.online_learner.predict_proba_one(dict(zip(X.columns, moment)))[y[i]]
                roll_probs = sliding_window_view(instances_prob, window_shape=actual_k)
                max_win = np.argmax(np.sum(roll_probs, axis=1))
                start = max_win * self.inst_stride
                end = start + actual_len * actual_k - self.inst_stride * (actual_k - 1)
                selection.append([max_win, actual_k, start, end])

            # Retraining or insisting now only with selected instances
            for i, bag in enumerate(X_bags):
                max_win = selection[i][0]
                actual_k = selection[i][1]
                relevant_instances = bag[max_win:max_win+actual_k]
                for instance in relevant_instances:
                    for moment in instance.transpose():
                        moment_dict = dict(zip(X.columns, moment))
                        self.online_learner.learn_one(moment_dict, y[i])
            # Comparing current acc with best obtained
            y_pred = self._predict(X.columns, X_bags)
            curr_acc = accuracy_score(y_true=y, y_pred=y_pred)
            result['acc_hist'].append(curr_acc)
            if curr_acc > best_acc:
                best_acc = curr_acc
                curr_pat = self.max_patience
                result['selection'] = copy.deepcopy(selection)
                backup = copy.deepcopy(self.online_learner)
            else:
                curr_pat -= 1
            curr_it += 1

        self.online_learner = backup
        return result['acc_hist'], result['selection']


    def _predict(self, X_columns: list, X_bags: list) -> np.ndarray:
        """Internal operations in the prediction process.
        """
        y_pred = np.zeros(len(X_bags))
        for i, bag in enumerate(X_bags):
            bag_outputs = list()
            # Generate labels at instance level
            for instance in bag:
                instance_outputs = list()
                for moment in instance.transpose():
                    moment_dict = dict(zip(X_columns, moment))
                    instance_outputs.append(self.online_learner.predict_one(moment_dict))
                bag_outputs.append(mode(instance_outputs)[0]) # Summarization of time information with mode of all instances
            # MIL assumption to pass to bag label
            if self.mil_assumption == 'max':
                y_pred[i] = max(bag_outputs)
            elif self.mil_assumption == 'mode':
                y_pred[i] = mode(bag_outputs)[0]
            elif self.mil_assumption == 'mean':
                y_pred[i] = round(np.mean(bag_outputs))
        return y_pred


    def predict(self, X: pd.MultiIndex) -> np.ndarray:
        """Predicts labels for time series in X.

        Parameters
        ----------
        X
            Time series dataset. It should be pandas.MultiIndex dataframe of shape [n_timeseries, n_dimensions, equal or unequal series length].

        Returns
        -------
        y
            Predicted class labels in numpy.ndarray format. Indices correspond to time series indices in X.
        """
        # Pass from time series dataset to multi-instance bags of sequences
        X_bags = list()
        for _, seq in X.groupby(level=0):
            np_seq = seq.to_numpy()
            actual_len = min(self.inst_len, len(np_seq))
            roll_win = sliding_window_view(np_seq, window_shape=actual_len, axis=0)[::self.inst_stride]
            X_bags.append(roll_win)
        return self._predict(X.columns, X_bags)


    def predict_bestk(self, X: pd.MultiIndex):
        """Get most relevant instances from a dataset without prior knowledge about class labels.

        Parameters
        ----------
        X
            Time series dataset. It should be pandas.MultiIndex dataframe of shape [n_timeseries, n_dimensions, equal or unequal series length].

        Returns
        -------
        [y, selection]
            Predicted class labels in numpy.ndarray format. Indices correspond to time series indices in X.
            Indices of the k best instances per bag.
        """
        # Getting y_pred
        X_bags = list()
        for _, seq in X.groupby(level=0):
            np_seq = seq.to_numpy()
            actual_len = min(self.inst_len, len(np_seq))
            roll_win = sliding_window_view(np_seq, window_shape=actual_len, axis=0)[::self.inst_stride]
            X_bags.append(roll_win)
        y_pred = self._predict(X.columns, X_bags)
        # Selecting k consecutive best instanes per bag
        selection = list()
        for i, bag in enumerate(X_bags):
            instances_prob = np.zeros(len(bag))
            actual_k = min(self.k, len(bag))
            for j, instance in enumerate(bag):
                for moment in instance.transpose():
                    instances_prob[j] += self.online_learner.predict_proba_one(dict(zip(X.columns, moment)))[y_pred[i]]
            roll_probs = sliding_window_view(instances_prob, window_shape=actual_k)
            max_win = np.argmax(np.sum(roll_probs, axis=1))
            start = max_win * self.inst_stride
            end = start + actual_len * actual_k - self.inst_stride * (actual_k - 1)
            selection.append([max_win, actual_k, start, end])
        return y_pred, selection
