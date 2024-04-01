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

The average results across in test for all the studied metrics are:

| Model             | Accuracy | Balanced Acc | Micro F1-score | Macro F1-score | Hamming loss | Execution time (s) | Model size (MB) |
|-------------------|----------|--------------|----------------|----------------|--------------|------------------:|------------------:|
| DrCIF             | 0.514    | 0.513        | 0.514          | 0.509          | 0.486                 | **12.635**             | **0.023**           |
| ShapeletTransform | 0.506    | 0.503        | 0.506          | 0.501          | 0.494                     | 8890.524           | 5.940           |
| MUSE              | 0.578    | 0.569        | 0.578          | 0.563          | 0.422                      | 1123.451           | 107.779         |
| SVM-Liner         | 0.408    | 0.399        | 0.408          | 0.351          | 0.592                  | 9286.714           | 43.709          |
| SVM-RBF           | 0.487    | 0.478        | 0.487          | 0.419          | 0.513                     | 20224.181          | 43.708          |
| Rocket            | 0.602    | 0.595        | 0.602          | **0.591**          | 0.398                    | 912.619            | 2.524           |
| HIVECOTEV2        | 0.473    | 0.464        | 0.473          | 0.453          | 0.527               | 8446.169           | 31.316          |
| KNN-ED            | 0.567    | 0.557        | 0.567          | 0.538          | 0.433                 | 13909.038          | 258.307         |
| KNN-DTW           | 0.371    | 0.362        | 0.371          | 0.364          | 0.629                   | 12511.160          | 54.649          |
| TapNet            | 0.556    | 0.548        | 0.556          | 0.521          | 0.444                  | 6594.003           | 2.580           |
| InceptionTime     | 0.585    | 0.583        | 0.585          | 0.573          | 0.415               | 962.320            | 1.623           |
| MIHT              | **0.636**    | **0.621**        | **0.636**          | 0.589          | **0.364**                   | 1508.548           | 92.304          |

In particular, the main metric studied, the accuracy obtained in test, is the following:

| Dataset                      | DrCIF | ST    | MUSE  | SVM-Linear | SVM-RBF | ROCKET | HIVECOTE2 | kNN-ED | kNN-DTW | TapNet | InceptionTime    | MIHT  |
|------------------------------|------:|------:|------:|-----------:|--------:|-------:|----------:|-------:|--------:|-------:|------:|------:|
| ArrowHead                    | 0.554 | 0.697 | **0.846** | 0.337      | 0.503   | 0.811  | -       | 0.800  | 0.703   | 0.480  | 0.703 | 0.589 |
| UnitTest                     | 0.909 | 0.818 | 0.773 | 0.227      | 0.682   | 0.909  | -       | 0.864  | **0.955**   | **0.955**  | 0.909 | 0.818 |
| ArticularyWordRecognition    | 0.693 | 0.850 | **0.993** | 0.053      | 0.813   | **0.993**  | **0.993**     | 0.970  | 0.987   | 0.900  | 0.990 | 0.933 |
| AtrialFibrillation           | **0.400** | 0.333 | 0.200 | 0.333      | 0.267   | 0.067  | 0.133     | 0.267  | 0.200   | 0.333  | 0.267 | **0.400** |
| BasicMotions                 | 0.950 | **1.000** | **1.000** | **1.000**      | 0.750   | **1.000**  | **1.000**     | 0.600  | 0.975   | 0.975  | **1.000** | **1.000** |
| Cricket                      | 0.847 | 0.903 | **1.000** | 0.167      | 0.917   | **1.000**  | **1.000**     | 0.944  | **1.000**   | 0.944  | 0.986 | 0.972 |
| DuckDuckGeese                | 0.300 | 0.360 | 0.460 | 0.540      | 0.260   | 0.500  | 0.300     | 0.500  | 0.580   | 0.480  | **0.640** | 0.620 |
| EigenWorms                   | 0.817 | 0.702 | **0.939** | 0.275      | 0.420   | 0.893  | 0.901     | 0.496  | 0.618   | -    | 0.809 | 0.779 |
| FingerMovements              | 0.390 | 0.620 | 0.560 | 0.560      | 0.490   | 0.540  | 0.490     | 0.550  | 0.530   | 0.530  | 0.530 | **0.630** |
| Heartbeat                    | 0.688 | 0.624 | 0.741 | 0.712      | 0.517   | **0.746**  | 0.707     | 0.620  | 0.717   | 0.717  | 0.702 | 0.722 |
| MotorImagery                 | 0.510 | **0.570** | 0.500 | 0.500      | 0.500   | 0.520  | 0.550     | 0.490  | 0.500   | 0.490  | 0.510 | 0.550 |
| SelfRegulationSCP1           | **0.826** | 0.720 | 0.792 | 0.457      | 0.502   | 0.846  | 0.840     | 0.778  | 0.775   | **0.826**  | -   | 0.805 |
| SelfRegulationSCP2           | 0.533 | 0.522 | **0.583** | 0.522      | 0.494   | 0.528  | 0.522     | 0.483  | 0.539   | 0.500  | 0.517 | 0.578 |
| StandWalkJump                | **0.600** | 0.333 | 0.400 | 0.467      | 0.200   | 0.467  | 0.267     | 0.200  | 0.200   | 0.133  | 0.333 | 0.533 |
| AsphaltRegularity            | 0.848 | 0.795 | 0.732 | 0.482      | 0.799   | 0.748  | 0.858     | 0.566  | -     | 0.877  | **0.940** | 0.885 |
| AllGestureWiimoteX           | 0.156 | 0.110 | 0.126 | 0.130      | 0.271   | 0.137  | 0.114     | **0.629**  | -     | 0.199  | 0.176 | 0.290 |
| AllGestureWiimoteY           | 0.204 | 0.113 | 0.150 | 0.170      | 0.243   | 0.236  | 0.164     | **0.651**  | -     | 0.264  | 0.229 | 0.277 |
| AllGestureWiimoteZ           | 0.166 | 0.169 | 0.240 | 0.167      | 0.237   | 0.259  | 0.259     | **0.611**  | -     | 0.283  | 0.361 | 0.266 |
| GesturePebbleZ2              | 0.234 | 0.323 | 0.367 | 0.399      | 0.310   | 0.348  | 0.361     | **0.658**  | -     | 0.266  | 0.386 | 0.525 |
| PickupGestureWiimoteZ        | 0.280 | 0.160 | 0.400 | 0.360      | 0.420   | 0.320  | 0.440     | 0.600  | **0.700**     | 0.400  | 0.460 | 0.420 |
| AsphaltObstaclesCoordinates  | 0.427 | 0.432 | 0.509 | 0.271      | 0.427   | **0.601**  | 0.560     | 0.340  | -     | 0.563  | 0.565 | 0.430 |
| AsphaltRegularityCoordinates | 0.867 | 0.819 | 0.885 | 0.507      | 0.698   | 0.876  | 0.895     | 0.555  | -     | 0.925  | **0.952** | 0.912 |
| InsectWingbeat               | -   | -   | 0.222 | 0.126      | 0.113   | 0.187  | -       | 0.165  | -     | 0.267  | -   | **0.337** |
| JapaneseVowels               | 0.651 | 0.432 | 0.435 | 0.957      | **0.968**   | 0.873  | 0.465     | 0.343  | 0.949     | 0.922  | **0.968** | 0.951 |
| SpokenArabicDigits           | -   | 0.242 | 0.608 | 0.475      | 0.387   | 0.636  | -       | 0.486  | -     | 0.662  | **0.693** | 0.680 |
| | | | | | | | | | | | | |
| *Average* | 0.514 | 0.506 | 0.578 | 0.408 | 0.487 | 0.602 | 0.473 | 0.567 | 0.437 | 0.556 | 0.585 | **0.636** |
| *Friedman's rank* | 7.54 | 7.68 | 5.46 | 7.78 | 8.34 | 4.72 | 6.70 | 6.78 | 8.54 | 6.10 | 4.76 | **3.60** |

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

And the critical distance plot is:

![cd for accuracy in test](results/cd_acc_test.jpg)

The complete results of the experimentation carried out in this work and presented and discussed in the associated paper are available in CSV format for download in the [results folder](results/) attending to the metrics:

| Metric | File |
|---|---|
|Accuracy in train | [acc_train.csv](results/acc_train.csv) |
|Accuracy in test | [acc_test.csv](results/acc_test.csv) |
|Execution time (seconds) in train | [exec_time_s_train.csv](results/exec_time_s_train.csv) |
|Execution time (seconds) in test | [exec_time_s_test.csv](results/exec_time_s_test.csv) |
|Size of the generated model (MB) | [memory_mb.csv](results/memory_mb.csv) |

## Reproductible experimentation

All the experimentation has been run in Python, using for the comparative analysis the implementations available in [Sktime](https://www.sktime.net/en/stable/) of the main time series classification methods, with the default parameters proposed by the authors. The methods used, their parameters and the reference implementation used are detailed below.

| Method | Family | Parameters | Implementation reference |
|---|---|---|---|
| MIHT | Multi-instance learning + incremental decision tree | `mil_assumption=mode`,`inst_len=0.4688`, `inst_stride=0.3039`, `k=4`, `grace_period=582`, `delta=2.508e-6`,`iters=30`, `patience=5`, `reset_model=False` | This repository |
| DrCif | Feature-based | `n_estimators=200`, `n_intervals=None`, `att_subsample_size=10`, `min_interval=4`, `max_interval=None`, `base_estimator='CIT'`, `time_limit_in_minutes=0.0`, `contract_max_n_estimators=500`, `save_transformed_data=False`, `n_jobs=1`, `random_state=None` | [DrCif in Sktime](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.classification.interval_based.DrCIF.html#sktime.classification.interval_based.DrCIF) |
| ST | Shapelet-based | `n_shapelet_samples=10000`, `max_shapelets=None`, `max_shapelet_length=None`, `estimator=ContinuousIntervalTree()`, `transform_limit_in_minutes=0`, `time_limit_in_minutes=0`, `contract_max_n_shapelet_samples=inf`, `save_transformed_data=False`, `n_jobs=1`, `batch_size=100`, `random_state=None` | [ShapeletTransformClassifier in Sktime](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.classification.shapelet_based.ShapeletTransformClassifier.html) |
| MUSE | Dictionary-based | `anova=True`, `variance=False`, `bigrams=True`, `window_inc=2`, `alphabet_size=4`, `use_first_order_differences=True`, `feature_selection='chi2'`, `p_threshold=0.05`, `support_probabilities=False`, `n_jobs=1`, `random_state=None` | [MUSE in Sktime](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.classification.dictionary_based.MUSE.html) |
| ROCKET | Convolutional-based | `num_kernels=10000`, `rocket_transform='rocket'`, `max_dilations_per_kernel=32`, `n_features_per_kernel=4`, `use_multivariate='auto'`, `n_jobs=1`, `random_state=None` | [RocketClassifier in Sktime](https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.classification.kernel_based.RocketClassifier.html) |
| HIVECOTE2 | Hybrid | `stc_params={'estimator': ContinuousIntervalTree()}`, `drcif_params={'n_estimators': 1}`, `arsenal_params={'n_estimators': 10, 'num_kernels': 100, 'rocket_transform': 'rocket'}`, `tde_params={'max_ensemble_size': 10}`, `time_limit_in_minutes=0`, `save_component_probas=False`, `verbose=0`, `n_jobs=1`, `random_state=None` | [HIVECOTEV2 in Sktime](https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.classification.hybrid.HIVECOTEV2.html) |
| SVM-Linear | Kernel-based | `kernel=AggrDist(PairwiseKernel(metric='linear'))`, `kernel_params=None`, `kernel_mtype=None`, `C=1`, `shrinking=True`, `probability=False`, `tol=0.001`, `cache_size=200`, `class_weight=None`, `verbose=False`, `max_iter=30`, `decision_function_shape='ovr'`, `break_ties=False`, `random_state=None` | [TimeSeriesSVC in Sktime](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.classification.kernel_based.TimeSeriesSVC.html) |
| SVM-RBF | Kernel-based | `kernel=AggrDist(PairwiseKernel(metric='rbf'))`, `kernel_params=None`, `kernel_mtype=None`, `C=1`, `shrinking=True`, `probability=False`, `tol=0.001`, `cache_size=200`, `class_weight=None`, `verbose=False`, `max_iter=30`, `decision_function_shape='ovr'`, `break_ties=False`, `random_state=None` | [TimeSeriesSVC in Sktime](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.classification.kernel_based.TimeSeriesSVC.html) |
| kNN-ED | Distance-based | `n_neighbors=1`, `weights='uniform'`, `algorithm='brute'`, `distance=DistFromAligner(AlignerDTW(dist_method='euclidean'))`, `distance_params=None`, `distance_mtype=None`, `pass_train_distances=False`, `leaf_size=30`, `n_jobs=None` | [KNeighborsTimeSeriesClassifier in Sktime](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.classification.distance_based.KNeighborsTimeSeriesClassifier.html) |
| kNN-DTW | Distance-based | `n_neighbors=1`, `weights='uniform'`, `algorithm='brute'`, `distance=DistFromAligner(AlignerDTWfromDist(DtwDist(weighted=False, derivative=False)))`, `distance_params=None`, `distance_mtype=None`, `pass_train_distances=False`, `leaf_size=30`, `n_jobs=None` | [KNeighborsTimeSeriesClassifier in Sktime](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.classification.distance_based.KNeighborsTimeSeriesClassifier.html) |
| TapNet | Deep learning | `n_epochs=500`, `batch_size=16`, `dropout=0.5`, `filter_sizes=(256, 256, 128)`, `kernel_size=(8, 5, 3)`, `dilation=1`, `layers=(500, 300)`, `use_rp=True`, `rp_params=(-1, 3)`, `activation='sigmoid'`, `use_bias=True`, `use_att=True`, `use_lstm=True`, `use_cnn=True`, `random_state=None`, `padding='same'`, `loss='binary_crossentropy'`, `optimizer=None`, `metrics=None`, `callbacks=None`, `verbose=False` | [TapNetClassifier in Sktime](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.classification.deep_learning.TapNetClassifier.html) |
| InceptionTime | Deep learning | `n_epochs=500`, `batch_size=64`, `kernel_size=40`, `n_filters=32`, `use_residual=True`, `use_bottleneck=True`, `bottleneck_size=32`, `depth=6`, `callbacks=None`, `random_state=None`, `verbose=False`, `loss='categorical_crossentropy'`, `metrics=None` | [InceptionTimeClassifier in Sktime](https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.classification.deep_learning.InceptionTimeClassifier.html) |
