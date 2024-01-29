# MIHT: A Hoeffding Tree Algorithm for Time Series Classification using Multiple Instance Learning

Associated repository with complementary materials to the manuscript *MIHT: A Hoeffding Tree Algorithm for Time Series Classification using Multiple Instance Learning* submitted to the 41st International Conference on Machine Learning (ICML). The following materials are included:

* Source code of the MIHT proposal.
* Datasets used in the experimentation.
* Complete tables of results.
* Complete instructions to execute the model and reproduce the experimentation.

## Source code

The purpose of this repository is to make public and accessible the source code of MIHT. This includes the dependencies of the library and the necessary instructions to use it.

The source code of MIHT is available in the file [src/miht.py](src/miht.py). And a complete tutorial for its execution is presented in the [Quick start notebook](src/tutorial.ipynb).

```python
from miht import MultiInstanceHoeffdingTreeClassifier

miht = MultiInstanceHoeffdingTreeClassifier(
    grace_period=500,
    delta=8.02e-4,
    mil_assumption='mode',
    inst_len=0.6,
    inst_stride=0.4,
    k=2,
    max_it=30,
    max_patience=5,
)
miht.fit(X_train, y_train)
```

## Datasets

MIHT's performance has been validated on a large selection of time-series classification datasets publicly available. All of them belong to the popular [UCR/UEA archive](http://www.timeseriesclassification.com/index.php), using in all the cases the train/test partitions provided by them. The datasets used are:

| Dataset | Vars | Train class dist | Train series | Train avg length | Train std length | Test class dist | Test series | Test avg length | Test std length |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [ArrowHead](http://www.timeseriesclassification.com/description.php?Dataset=ArrowHead) | 1 | 0.33/0.33/0.33 | 36 | 251.0 | 0.0 | 0.39/0.3/0.3 | 175 | 251.0 | 0.0 |
| [UnitTest](http://www.timeseriesclassification.com/dataset.php) | 1 | 0.5/0.5 | 20 | 24.0 | 0.0 | 0.55/0.45 | 22 | 24.0 | 0.0 |
| [ArticularyWordRecognition](http://www.timeseriesclassification.com/description.php?Dataset=ArticularyWordRecognition) | 9 | 0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04 | 275 | 144.0 | 0.0 | 0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04/0.04 | 300 | 144.0 | 0.0 |
| [AtrialFibrillation](http://www.timeseriesclassification.com/description.php?Dataset=AtrialFibrillation) | 2 | 0.33/0.33/0.33 | 15 | 640.0 | 0.0 | 0.33/0.33/0.33 | 15 | 640.0 | 0.0 |
| [BasicMotions](http://www.timeseriesclassification.com/description.php?Dataset=BasicMotions) | 6 | 0.25/0.25/0.25/0.25 | 40 | 100.0 | 0.0 | 0.25/0.25/0.25/0.25 | 40 | 100.0 | 0.0 |
| [Cricket](http://www.timeseriesclassification.com/description.php?Dataset=Cricket) | 6 | 0.08/0.08/0.08/0.08/0.08/0.08/0.08/0.08/0.08/0.08/0.08/0.08 | 108 | 1197.0 | 0.0 | 0.08/0.08/0.08/0.08/0.08/0.08/0.08/0.08/0.08/0.08/0.08/0.08 | 72 | 1197.0 | 0.0 |
| [DuckDuckGeese](http://www.timeseriesclassification.com/description.php?Dataset=DuckDuckGeese) | 1345 | 0.2/0.2/0.2/0.2/0.2 | 50 | 270.0 | 0.0 | 0.2/0.2/0.2/0.2/0.2 | 50 | 270.0 | 0.0 |
| [EigenWorms](http://www.timeseriesclassification.com/description.php?Dataset=EigenWorms) | 6 | 0.43/0.17/0.13/0.17/0.09 | 128 | 17984.0 | 0.0 | 0.42/0.17/0.14/0.18/0.1 | 131 | 17984.0 | 0.0 |
| [FingerMovements](http://www.timeseriesclassification.com/description.php?Dataset=FingerMovements) | 28 | 0.5/0.5 | 316 | 50.0 | 0.0 | 0.49/0.51 | 100 | 50.0 | 0.0 |
| [Heartbeat](http://www.timeseriesclassification.com/description.php?Dataset=Heartbeat) | 61 | 0.72/0.28 | 204 | 405.0 | 0.0 | 0.72/0.28 | 205 | 405.0 | 0.0 |
| [MotorImagery](http://www.timeseriesclassification.com/description.php?Dataset=MotorImagery) | 64 | 0.5/0.5 | 278 | 3000.0 | 0.0 | 0.5/0.5 | 100 | 3000.0 | 0.0 |
| [SelfRegulationSCP1](http://www.timeseriesclassification.com/description.php?Dataset=SelfRegulationSCP1) | 6 | 0.5/0.5 | 268 | 896.0 | 0.0 | 0.5/0.5 | 293 | 896.0 | 0.0 |
| [SelfRegulationSCP2](http://www.timeseriesclassification.com/description.php?Dataset=SelfRegulationSCP2) | 7 | 0.5/0.5 | 200 | 1152.0 | 0.0 | 0.5/0.5 | 180 | 1152.0 | 0.0 |
| [StandWalkJump](http://www.timeseriesclassification.com/description.php?Dataset=StandWalkJump) | 4 | 0.33/0.33/0.33 | 12 | 2500.0 | 0.0 | 0.33/0.33/0.33 | 15 | 2500.0 | 0.0 |
| [AsphaltRegularity](http://www.timeseriesclassification.com/description.php?Dataset=AsphaltRegularity) | 1 | 0.49/0.51 | 751 | 387.1 | 252.33 | 0.49/0.51 | 751 | 380.9 | 205.6 |
| [AllGestureWiimoteX](http://www.timeseriesclassification.com/description.php?Dataset=AllGestureWiimoteX) | 1 | 0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1 | 300 | 124.9 | 65.88 | 0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1 | 700 | 124.7 | 68.9 |
| [AllGestureWiimoteY](http://www.timeseriesclassification.com/description.php?Dataset=AllGestureWiimoteY) | 1 | 0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1 | 300 | 128.6 | 69.61 | 0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1 | 700 | 123.1 | 67.2 |
| [AllGestureWiimoteZ](http://www.timeseriesclassification.com/description.php?Dataset=AllGestureWiimoteZ) | 1 | 0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1 | 300 | 125.5 | 66.31 | 0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1 | 700 | 124.4 | 68.7 |
| [GesturePebbleZ2](http://www.timeseriesclassification.com/description.php?Dataset=GesturePebbleZ2) | 1 | 0.17/0.16/0.16/0.16/0.16/0.18 | 146 | 223.5 | 88.7 | 0.15/0.14/0.19/0.18/0.18/0.16 | 158 | 215.4 | 60.0 |
| [PickupGestureWiimoteZ](http://www.timeseriesclassification.com/description.php?Dataset=PickupGestureWiimoteZ) | 1 | 0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1 | 50 | 145.9 | 78.09 | 0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1 | 50 | 145.5 | 69.0 |
| [AsphaltObstaclesCoordinates](http://www.timeseriesclassification.com/description.php?Dataset=AsphaltObstaclesCoordinates) | 3 | 0.21/0.24/0.27/0.28 | 390 | 297.8 | 114.75 | 0.2/0.24/0.27/0.28 | 391 | 299.5 | 114.2 |
| [AsphaltRegularityCoordinates](http://www.timeseriesclassification.com/description.php?Dataset=AsphaltRegularityCoordinates) | 3 | 0.49/0.51 | 751 | 387.1 | 252.33 | 0.49/0.51 | 751 | 380.9 | 205.6 |
| [InsectWingbeat](http://www.timeseriesclassification.com/description.php?Dataset=InsectWingbeat) | 200 | 0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1 | 25000 | 6.7 | 1.6 | 0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1 | 25000 | 6.7 | 1.6 |
| [JapaneseVowels](http://www.timeseriesclassification.com/description.php?Dataset=JapaneseVowels) | 12 | 0.11/0.11/0.11/0.11/0.11/0.11/0.11/0.11/0.11 | 270 | 15.8 | 3.59 | 0.08/0.09/0.24/0.12/0.08/0.06/0.11/0.14/0.08 | 370 | 15.4 | 3.6 |
| [SpokenArabicDigits](http://www.timeseriesclassification.com/description.php?Dataset=SpokenArabicDigits) | 13 | 0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1 | 6599 | 39.9 | 8.72 | 0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1 | 2199 | 39.6 | 8.0 |

## Results

The complete results of the experimentation carried out in this work and presented and discussed in the associated paper are available in CSV format for download in the [results folder](results/) attending to the metrics:

| Metric | File |
|---|---|
|Accuracy in train | [acc_train.csv](results/acc_train.csv) |
|Accuracy in test | [acc_test.csv](results/acc_test.csv) |
|Execution time (seconds) in train | [exec_time_s_train.csv](results/exec_time_s_train.csv) |
|Execution time (seconds) in test | [exec_time_s_test.csv](results/exec_time_s_test.csv) |
|Size of the generated model (MB) | [memory_mb.csv](results/memory_mb.csv) |

Moreover, the results are summarized in the following graphs for both accuracy and time of execution (considering both train and test times in seconds). These graphs show the distribution per dataset of the tested models and at which point is our proposed MLHT.

![Accuracy on test](results/boxplot_acc_test.jpg)
![Total execution time](results/boxplot_exec_time_s_total.jpg)

The raw measures per model and dataset have been used to find statistically significant differences between the studied methods. Specifically we use the Friedman test of the ranks of the metrics and the post-hoc Bonferroni-Dumm test to find the pair of groups which are significantly different.

We have use R and its [scmamp](https://github.com/b0rxa/scmamp) library in the following way:

```R
library(scmamp)

# Load raw data
rd <- read.csv(csv_path)
nAlgorithms <- ncol(rd)-1
nDatasets <- nrow(rd)
rdm <- rd[, 2: (nAlgorithms+1)]
# Friedman test. Multiple comparison
alpha <- 0.01
friedman <- friedmanTest(data=rdm,alpha=alpha)
if(friedman$p.value < alpha) {
    # Post-Hoc test
    test <- postHocTest(data=rdm, test='friedman', correct='bonferroni', alpha=alpha, use.rank=FALSE, sum.fun=mean)
}
```

And the critical distance plots are:

![cd for accuracy in test](results/cd_acc_test.jpg)

![cd for total running time in seconds](results/cd_exec_time_s_total.jpg)

## Reproductible experimentation

All the experimentation has been run in Python, using for the comparative analysis the implementations available in [Sktime](https://www.sktime.net/en/stable/) of the main time series classification methods, with the default parameters proposed by the authors. The methods used, their parameters and the reference implementation used are detailed below.

| Method | Family | Parameters | Implementation reference
|---|---|---|---|
| MIHT | Multi-instance learning + incremental decision tree | `mil_assumption=mode`,`inst_len=0.4688`, `inst_stride=0.3039`, `k=4`, `grace_period=582`, `delta=2.508e-6`,`iters=30`, `patience=5`, `reset_model=False` | This repository |
| DrCif | Feature-based | `n_estimators=200`, `n_intervals=None`, `att_subsample_size=10`, `min_interval=4`, `max_interval=None`, `base_estimator='CIT'`, `time_limit_in_minutes=0.0`, `contract_max_n_estimators=500`, `save_transformed_data=False`, `n_jobs=1`, `random_state=None` | [DrCif in Sktime](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.classification.interval_based.DrCIF.html#sktime.classification.interval_based.DrCIF)
| ST | Shapelet-based | `n_shapelet_samples=10000`, `max_shapelets=None`, `max_shapelet_length=None`, `estimator=ContinuousIntervalTree()`, `transform_limit_in_minutes=0`, `time_limit_in_minutes=0`, `contract_max_n_shapelet_samples=inf`, `save_transformed_data=False`, `n_jobs=1`, `batch_size=100`, `random_state=None` | [ShapeletTransformClassifier in Sktime](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.classification.shapelet_based.ShapeletTransformClassifier.html) |
| MUSE | Dictionary-based | `anova=True`, `variance=False`, `bigrams=True`, `window_inc=2`, `alphabet_size=4`, `use_first_order_differences=True`, `feature_selection='chi2'`, `p_threshold=0.05`, `support_probabilities=False`, `n_jobs=1`, `random_state=None` | [MUSE in Sktime](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.classification.dictionary_based.MUSE.html) |
| SVM-Linear | Kernel-based | `kernel=AggrDist(PairwiseKernel(metric='linear'))`, `kernel_params=None`, `kernel_mtype=None`, `C=1`, `shrinking=True`, `probability=False`, `tol=0.001`, `cache_size=200`, `class_weight=None`, `verbose=False`, `max_iter=30`, `decision_function_shape='ovr'`, `break_ties=False`, `random_state=None` | [TimeSeriesSVC in Sktime](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.classification.kernel_based.TimeSeriesSVC.html) |
| SVM-RBF | Kernel-based | `kernel=AggrDist(PairwiseKernel(metric='rbf'))`, `kernel_params=None`, `kernel_mtype=None`, `C=1`, `shrinking=True`, `probability=False`, `tol=0.001`, `cache_size=200`, `class_weight=None`, `verbose=False`, `max_iter=30`, `decision_function_shape='ovr'`, `break_ties=False`, `random_state=None` | [TimeSeriesSVC in Sktime](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.classification.kernel_based.TimeSeriesSVC.html) |
| kNN-ED | Distance-based | `n_neighbors=1`, `weights='uniform'`, `algorithm='brute'`, `distance=DistFromAligner(AlignerDTW(dist_method='euclidean'))`, `distance_params=None`, `distance_mtype=None`, `pass_train_distances=False`, `leaf_size=30`, `n_jobs=None` | [KNeighborsTimeSeriesClassifier in Sktime](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.classification.distance_based.KNeighborsTimeSeriesClassifier.html) |
| kNN-DTW | Distance-based | `n_neighbors=1`, `weights='uniform'`, `algorithm='brute'`, `distance=DistFromAligner(AlignerDTWfromDist(DtwDist(weighted=False, derivative=False)))`, `distance_params=None`, `distance_mtype=None`, `pass_train_distances=False`, `leaf_size=30`, `n_jobs=None` | [KNeighborsTimeSeriesClassifier in Sktime](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.classification.distance_based.KNeighborsTimeSeriesClassifier.html) |
| TapNet | Deep learning | `n_epochs=500`, `batch_size=16`, `dropout=0.5`, `filter_sizes=(256, 256, 128)`, `kernel_size=(8, 5, 3)`, `dilation=1`, `layers=(500, 300)`, `use_rp=True`, `rp_params=(-1, 3)`, `activation='sigmoid'`, `use_bias=True`, `use_att=True`, `use_lstm=True`, `use_cnn=True`, `random_state=None`, `padding='same'`, `loss='binary_crossentropy'`, `optimizer=None`, `metrics=None`, `callbacks=None`, `verbose=False` | [TapNetClassifier in Sktime](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.classification.deep_learning.TapNetClassifier.html) |
