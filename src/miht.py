import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import mode
from sklearn.metrics import accuracy_score
import copy
from river.tree.hoeffding_tree import HoeffdingTree
from river.tree import HoeffdingTreeClassifier
from sktime.transformations.panel.padder import PaddingTransformer
import random


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
    def __init__(self, grace_period=150, delta=1e-7, mil_assumption='mode', inst_len=0.9, inst_stride=0.01, k=3, max_it=30, max_patience=5, consecutive_signature=False) -> None:
        self.mil_assumption = mil_assumption
        self.inst_len = inst_len
        self.inst_stride = inst_stride
        self.k = k
        self.max_patience = max_patience
        self.max_it = max_it
        self.dependent_graceper = grace_period
        self.consecutive_signature = consecutive_signature
        self.padder = PaddingTransformer()
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

        if self.dependent_graceper > 0 and isinstance(self.ref_online_learner, HoeffdingTree):
            self.ref_online_learner.grace_period = self.inst_len * self.dependent_graceper
        self.online_learner = copy.deepcopy(self.ref_online_learner)

        X_train = X
        y_train = y

        # First training with all the instances
        X_bags_flatten, bags_info = self._sequence2bag(X_train)
        for i, bag in enumerate(bags_info):
            start_bag = bag['start']
            end_bag = start_bag + bag['n_insts']
            instances = X_bags_flatten.loc[start_bag:end_bag-1] # loc includes the last element of the range in the selection
            for _, instance in instances.iterrows():
                self.online_learner.learn_one(dict(instance), y_train[i])

        # Preparing for comparing the evolution of the model during convergence
        backup = copy.deepcopy(self.online_learner)
        y_pred = self._predict(X_bags_flatten, bags_info)
        best_acc = accuracy_score(y_true=y, y_pred=y_pred)
        result = {
            'selection': None,
            'acc_hist': [best_acc]
        }

        # Convergence process for finding signature of each time series
        curr_it = 0
        curr_pat = self.max_patience
        prev_acc = best_acc
        while curr_it < self.max_it and curr_pat > 0:
            if self.consecutive_signature:
                # Selecting k consecutive best instanes per bag
                selection = self._predict_signature_consecutive(X_bags_flatten, bags_info, y_train)
            else:
                # Selecting k best instances per bag
                selection = self._predict_signature_separated(X_bags_flatten, bags_info, y_train)

            # Retraining or insisting now only with selected instances
            if self.consecutive_signature:
                for i, bag_selection in enumerate(selection):
                    start_win = bag_selection[2]
                    end_win = bag_selection[3]
                    best_instances = X_bags_flatten.loc[start_win:end_win-1]
                    for _, instance in best_instances.iterrows():
                        self.online_learner.learn_one(instance, y_train[i])
            else:
                for i, bag_selection in enumerate(selection):
                    for instance in bag_selection[1]:
                        if instance == -1:
                            continue
                        self.online_learner.learn_one(X_bags_flatten.loc[instance], y_train[i])

            # Comparing current acc with best obtained
            y_pred = self._predict(X_bags_flatten, bags_info)
            curr_acc = accuracy_score(y_true=y_train, y_pred=y_pred)
            result['train_acc_hist'].append(curr_acc)
            # Backup the best model
            if curr_acc > best_acc:
                best_acc = curr_acc
                curr_pat = self.max_patience
                result['selection'] = copy.deepcopy(selection)
                backup = copy.deepcopy(self.online_learner)
            # Early stopping mechanism
            if curr_acc >= prev_acc:
                curr_pat = self.max_patience
            else:
                curr_pat -= 1
            if curr_acc == 1:
                break
            prev_acc = curr_acc
            curr_it += 1

        self.online_learner = backup
        return result


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
        X_bags = self._sequence2bag(X)
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
        X_bags = self._sequence2bag(X)
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
            end = start + len(instance.transpose()) * actual_k - self.inst_stride * (actual_k - 1)
            selection.append([max_win, actual_k, start, end])
        return y_pred, selection


    def predict_signature(self, X: pd.MultiIndex):
        X_bags, bags_info = self._sequence2bag(X)
        y_pred = self._predict(X_bags, bags_info)
        if self.consecutive_signature:
            signature = self._predict_signature_consecutive(X_bags, bags_info, y_pred)
        else:
            signature = self._predict_signature_separated(X_bags, bags_info, y_pred)
        return y_pred, signature


    def _sequence2bag(self, X: pd.MultiIndex):
        """
        Return a dataframe with an instance per row. Each row has all the time steps that compose the instance for each category ordered by columns like [c1_t1, c1_t2..., cC_t1..., cC_tT]. Additionally, the bags_info structure keeps the reference of the separation of instances of the different bags by keeping the index of the first instance of a bag and the number of instances in it (rows from the bag start).
        """
        X_bags_aux = list()
        bags_info = list()
        ref = 0
        for _, seq in X.groupby(level=0):
            np_seq = seq.to_numpy()
            actual_len = min(self.inst_len, len(np_seq))
            actual_stride = min(self.inst_len, len(np_seq))
            roll_win = sliding_window_view(np_seq, window_shape=actual_len, axis=0)[::actual_stride]
            for instance in roll_win:
                X_bags_aux.append(pd.DataFrame(data=instance.transpose(), columns=X.columns))
            bags_info.append({'start': ref,
                              'n_insts': len(roll_win),
                              'inst_len': actual_len})
            ref += len(roll_win)
        X_bags_mi = pd.concat(X_bags_aux, keys=range(len(X_bags_aux)), axis=0)
        X_bags_mi = self.padder.fit_transform(X_bags_mi)
        X_columns_flatten = list()
        for t in range(len(X_bags_mi.loc[0])):
            for c in X.columns:
                X_columns_flatten.append(f'{c}_t{t}')
        X_bags_aux = list()
        for _, inst in X_bags_mi.groupby(level=0):
            flat_inst = dict(zip(X_columns_flatten, inst.values.flatten()))
            X_bags_aux.append(flat_inst)
        X_bags_flatten = pd.DataFrame(X_bags_aux)
        return X_bags_flatten, bags_info


    def _predict(self, X_bags_flatten: pd.DataFrame, bags_info: dict) -> np.ndarray:
        """Internal operations in the prediction process.
        """
        y_pred = np.zeros(len(bags_info))
        for i, bag in enumerate(bags_info):
            start = bag['start']
            end = bag['start'] + bag['n_insts']
            instances = X_bags_flatten.loc[start:end-1]
            bag_outputs = list()
            for _, x in instances.iterrows():
                bag_outputs.append(self.online_learner.predict_one(dict(x)))
            # MIL assumption to pass to bag label
            if self.mil_assumption == 'max':
                y_pred[i] = max(bag_outputs)
            elif self.mil_assumption == 'mode':
                y_pred[i] = mode(bag_outputs, keepdims=False)[0]
            elif self.mil_assumption == 'mean':
                y_pred[i] = round(np.mean(bag_outputs))
        return y_pred


    def _predict_signature_consecutive(self, X_bags: pd.DataFrame, bags_info: list, y: list) -> list:
            # Selecting k consecutive best instanes per bag
            selection = list()
            for i, bag in enumerate(bags_info):
                instances_prob = np.zeros(bag['n_insts'])
                # Extract the portion for the bag
                actual_k = min(self.k, bag['n_insts'])
                start_bag = bag['start']
                end_bag = bag['start'] + bag['n_insts']
                instances = X_bags.loc[start_bag:end_bag-1]
                # Accumalate the probabilities of all the time steps in an instance for the bag class
                for j, (_, instance) in enumerate(instances.iterrows()):
                    instances_prob[j] += self.online_learner.predict_proba_one(instance)[y[i]]
                # Accumulate probabilities per succesive windows of k instances
                roll_probs = sliding_window_view(instances_prob, window_shape=actual_k)
                windows = np.sum(roll_probs, axis=1)
                # Get the window with highest score (or one of them randomly)
                max_wins = np.argwhere(windows == np.max(windows)).reshape(-1)
                max_win = random.choice(max_wins)
                # Reverse case also, just in case
                roll_probs = sliding_window_view(instances_prob[::-1], window_shape=actual_k)
                windows_r = np.sum(roll_probs, axis=1)
                max_wins = np.argwhere(windows_r == np.max(windows_r)).reshape(-1)
                max_win_r = random.choice(max_wins)

                # Duration of the selected window in time steps
                if windows[max_win] > windows_r[max_win_r]:
                    take_reverse = False
                elif windows[max_win] < windows_r[max_win_r]:
                    take_reverse = True
                else:
                    take_reverse = random.choice([True, False])
                if take_reverse:
                    start_win = bag['start'] + bag['n_insts'] - max_win_r - actual_k
                else:
                    start_win = bag['start'] + max_win
                end_win = start_win + actual_k
                # Duration of the instance with the highest probability
                max_insts = np.argwhere(instances_prob == np.max(instances_prob))
                start_inst = bag['start'] + random.choice(max_insts)
                end_inst = start_inst + bag['inst_len']
                # Save everything
                selection.append([max_win, actual_k, start_win, end_win, start_inst, end_inst])
            return selection


    def _predict_signature_separated(self, X_bags: pd.DataFrame, bags_info: list, y: list) -> list:
        # Selecting k best instances per bag – don't need to be consecutive
        selection = list()
        for i, bag in enumerate(bags_info):
            instances_prob = np.zeros(bag['n_insts'])
            # Extract the portion for the bag
            actual_k = min(self.k, bag['n_insts'])
            start_bag = bag['start']
            end_bag = bag['start'] + bag['n_insts']
            instances = X_bags.loc[start_bag:end_bag-1]
            # Accumalate the probabilities of all the time steps in an instance for the bag class
            for j, (_, instance) in enumerate(instances.iterrows()):
                instances_prob[j] += self.online_learner.predict_proba_one(instance)[y[i]]
            # Get the k instances with highest score
            max_insts = np.argsort(instances_prob)[-actual_k:]
            max_insts_idx = max_insts+bag['start']
            # Save everything
            max_insts = np.concatenate((max_insts, [-1] * (self.k - len(max_insts))))
            max_insts_idx = np.concatenate((max_insts_idx, [-1] * (self.k - len(max_insts_idx))))
            selection.append([max_insts, max_insts_idx]) # The best instance is the last one of the max_insts array
        return selection