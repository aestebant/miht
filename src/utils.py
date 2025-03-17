def ordinal_targets(dataset, y) -> list:
    """ Transform classes of the UCR/UEA repository from symbols to numerical categories.
    Parameters
    ----------
    dataset
        String with the name of the dataset as it appears in http://www.timeseriesclassification.com/dataset.php
    y
        The array with the original targets
    Return
    ------
    An array of the same size of y with the targets transformed to int
    """
    if dataset in {'AsphaltObstacles', 'AsphaltObstaclesCoordinates'}:
        converter = {'speed_bump': 0, 'raised_crosswalk': 1, 'raised_markers': 2, 'vertical_patch': 3}
    elif dataset in {'AsphaltPavementType', 'AsphaltPavementTypeCoordinates'}:
        converter = {'cobblestone': 0, 'dirt': 1, 'flexible': 2}
    elif dataset in {'AsphaltRegularity', 'AsphaltRegularityCoordinates'}:
        converter = {'regular': 0, 'deteriorated': 1}
    elif dataset == 'AtrialFibrillation':
        converter = {'n': 0, 's': 1, 't': 2}
    elif dataset == 'BasicMotions':
        converter = {'standing': 0, 'walking': 1, 'running': 2, 'badminton': 3}
    elif dataset == 'DuckDuckGeese':
        converter = {'white-faced_whistling_duck': 0, 'canadian_goose': 1, 'pink-footed_goose': 2, 'greylag_goose': 3, 'black-bellied_whistling_duck': 4}
    elif dataset == 'Epilepsy':
        converter = {'epilepsy': 3, 'walking': 0, 'running': 1, 'sawing': 2}
    elif dataset == 'EthanolConcentration':
        converter = {'e40': 2, 'e45': 3, 'e35': 0, 'e38': 1}
    elif dataset == 'FingerMovements':
        converter = {'left': 0, 'right': 1}
    elif dataset == 'HandMovementDirection':
        converter = {'forward': 2, 'left': 0, 'right': 1, 'backward': 3}
    elif dataset == 'Heartbeat':
        converter = {'abnormal': 1, 'normal': 0}
    elif dataset == 'InsectWingbeat':
        converter = {'stigma_male': 0, 'stigma_female': 1, 'quinx_male': 2, 'quinx_female': 3, 'tarsalis_male': 4, 'tarsalis_female': 5, 'aedes_male': 6, 'aedes_female': 7, 'fruit_flies': 8, 'house_flies': 9}
    elif dataset == 'MotorImagery':
        converter = {'finger': 0, 'tongue': 1}
    elif dataset == 'PhonemeSpectra':
        converter = {'hh': 0, 'dh': 1, 'f': 2, 's': 3, 'sh': 4, 'th': 5,  'v': 6,  'z': 7, 'zh': 8, 'ch': 9, 'jh': 10, 'b': 11, 'd': 12, 'g': 13, 'k': 14,  'p': 15,  't': 16, 'm': 17, 'n': 18, 'ng': 19, 'aa': 20, 'ae': 21, 'ah': 22, 'uw': 23, 'ao': 24, 'aw': 25, 'ay': 26, 'uh': 27, 'eh': 28, 'er': 29, 'ey': 30, 'oy': 31, 'ih': 32, 'iy': 33, 'ow': 34, 'w': 35, 'y': 36, 'l': 37, 'r': 38}
    elif dataset == 'RacketSports':
        converter = {'squash_backhandboast': 2, 'badminton_clear': 0, 'squash_forehandboast': 3, 'badminton_smash': 1}
    elif dataset in {'SelfRegulationSCP1', 'SelfRegulationSCP2'}:
        converter = {'negativity': 0, 'positivity': 1}
    elif dataset == 'StandWalkJump':
        converter = {'jumping': 2, 'walking': 1, 'standing': 0}
    elif dataset in {'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ', 'Beef', 'BeetleFly', 'BirdChicken', 'BME', 'Car', 'CBF', 'Chinatown', 'ChlorineConcentration', 'CinCECGTorso', 'Computers', 'CricketX', 'CricketY', 'CricketZ', 'Crop', 'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup', 'DodgerLoopDay', 'DodgerLoopGame', 'DodgerLoopWeekend', 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'EOGHorizontalSignal', 'EOGVerticalSignal', 'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords', 'Fish', 'FreezerRegularTrain', 'FreezerSmallTrain', 'Fungi', 'GestureMidAirD1', 'GestureMidAirD2', 'GestureMidAirD3', 'GesturePebbleZ1',  'GesturePebbleZ2', 'GunPointAgeSpan', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'Ham', 'Haptics', 'Herring', 'HouseTwenty', 'InlineSkate', 'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'LargeKitchenAppliances', 'Mallat', 'Meat', 'MedicalImages', 'MelbournePedestrian', 'MiddlePhalanxOutlineAgeGroup', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain', 'MoteStrain', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2', 'OliveOil', 'Phoneme', 'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP', 'PLAID', 'Plane',  'PowerCons', 'ProximalPhalanxOutlineAgeGroup', 'RefrigerationDevices', 'Rock', 'ScreenType', 'SemgHandGenderCh2', 'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ', 'ShapesAll', 'SmallKitchenAppliances', 'SmoothSubspace', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl', 'TwoLeadECG', 'TwoPatterns', 'CharacterTrajectories', 'JapaneseVowels', 'SpokenArabicDigits', 'EigenWorms', 'ERing', 'FaceDetection', 'Libras', 'LSST', 'PenDigits', 'GunPoint', 'ItalyPowerDemand', 'OSULeaf', 'UMD', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'UnitTest', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga'}:
        class LabelToInt(dict):
            def __missing__(self, key):
                return int(key)-1
        converter = LabelToInt()
    elif dataset in {'ArticularyWordRecognition', 'Cricket', 'Handwriting', 'NATOPS', 'PEMS-SF', 'UWaveGestureLibrary'}:
        class LabelToInt2(dict):
            def __missing__(self, key):
                return int(float(key))-1
        converter = LabelToInt2()
    elif dataset in {'ACSF1', 'ArrowHead', 'Coffee', 'DistalPhalanxOutlineCorrect', 'Earthquakes', 'HandOutlines', 'Lightning7', 'MiddlePhalanxOutlineCorrect', 'PhalangesOutlinesCorrect', 'ProximalPhalanxOutlineCorrect', 'ShapeletSim', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace'}:
        class LabelToInt3(dict):
            def __missing__(self, key):
                return int(key)
        converter = LabelToInt3()
    elif dataset in {'DistalPhalanxTW', 'MiddlePhalanxTW', 'ProximalPhalanxTW'}:
        class LabelToInt4(dict):
            def __missing__(self, key):
                return int(key)-3
        converter = LabelToInt4()
    elif dataset in {'ECG200', 'FordA', 'FordB', 'Lightning2', 'Wafer'}:
        converter = {'-1': 0, '1': 1}
    else:
        class IntToInt(dict):
            def __missing__(self, key):
                return key
        converter = IntToInt()

    return [converter[i] for i in y]
