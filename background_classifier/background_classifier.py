#!/usr/bin/env python-real

import os
import sys
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from joblib import dump, load
import numpy as np
import time
import pickle
import math
from scipy import ndimage

MATRIX_SIZE = 7


def loadStorageNode(pathToStorageNode):
    # EMPTYPATH is default value for predictionStoragePath parameter, it is set up in xml file.
    if pathToStorageNode != 'EMPTY_PATH':
        return pickle.load(open(pathToStorageNode, 'rb'))
    return None


def saveStorageNode(storageNode, pathToStorageNode):
    with open(pathToStorageNode, 'wb') as f:
        pickle.dump(storageNode, f, pickle.HIGHEST_PROTOCOL)


def prepareData(preparationType, storageNode, features):
    if preparationType == 'TRAINING_PREPARE_FEATURES':
        print('Preparing feature vector for training data...')
        return prepareFeatures(storageNode['trainingData'], features), prepareMasks(storageNode['trainingMask'])
    elif preparationType == 'PREDICTION_PREPARE_FEATURES':
        print('Preparing feature vector for prediction data...')
        return prepareFeatures(storageNode['predictionData'], features)
    elif preparationType == 'VALIDATION_PREPARE_FEATURES':
        print('Preparing feature vector for validation data...')
        return prepareFeatures(storageNode['validationData'], features)
    else:
        raise ValueError('Type of feature preparation is not valid!')


def toSigned(volumeSlice):
    minimum = np.min(volumeSlice)
    if minimum < 0:
        volumeSlice = volumeSlice + abs(minimum)
    return volumeSlice


def normalize(volumeSlice):
    # This method normalizes value of each pixel, converts slice to unsigned int
    # to get only 256 possible pixel values and then normalizes values to range between 0-1
    volumeSlice = volumeSlice.astype(np.float64)
    volumeSlice /= np.max(volumeSlice)
    volumeSlice *= 255

    volumeSlice = volumeSlice.astype(np.uint8)

    volumeSlice = volumeSlice.astype(np.float16)
    volumeSlice /= 255
    return volumeSlice


def equalizeHistogram(volumeSlice):
    reshapedSlice = volumeSlice.ravel()
    histogram, bins = np.histogram(reshapedSlice, bins=10, density=True)
    cumSum = histogram.cumsum()
    # Normalize cumulative sum, value at last index is maximum
    cumSum /= cumSum[-1]
    equalizedSlice = np.interp(reshapedSlice, bins[:-1], cumSum)
    equalizedSlice = equalizedSlice / np.max(equalizedSlice)
    return equalizedSlice.reshape(volumeSlice.shape)


def countGradient(volumeSlice):
    # There is also gradient method in numpy module but doesn't behave exactly same
    assert (MATRIX_SIZE % 2 == 1)
    gradientsX = np.zeros((volumeSlice.shape[0], volumeSlice.shape[1]), dtype=np.float32)
    gradientsY = np.zeros((volumeSlice.shape[0], volumeSlice.shape[1]), dtype=np.float32)
    gradientsFeature = np.zeros((volumeSlice.size, MATRIX_SIZE, MATRIX_SIZE), dtype=np.float32)
    middle = math.trunc(MATRIX_SIZE / 2)

    for i in range(volumeSlice.shape[0]):
        for j in range(volumeSlice.shape[1] - 1):
            gradientsX[i, j + 1] = math.fabs((volumeSlice[i, j + 1]) - (volumeSlice[i, j]))

    for i in range(volumeSlice.shape[0] - 1):
        for j in range(volumeSlice.shape[1]):
            gradientsY[i + 1, j] = math.fabs((volumeSlice[i + 1, j]) - (volumeSlice[i, j]))

    gradientsX = np.pad(gradientsX, middle, mode='constant')
    gradientsY = np.pad(gradientsY, middle, mode='constant')
    gradientsSum = (np.add(gradientsX, gradientsY)) / 2
    pixel = 0
    for i in range(volumeSlice.shape[0]):
        for j in range(volumeSlice.shape[1]):
            gradientsFeature[pixel] = gradientsSum[i:i + MATRIX_SIZE, j:j + MATRIX_SIZE]
            pixel += 1
    return gradientsFeature.reshape(-1, MATRIX_SIZE * MATRIX_SIZE)


def gaussianFiltering(volumeSlice):
    gaussian = ndimage.gaussian_filter(volumeSlice, sigma=2)
    gaussian /= np.max(gaussian)
    return gaussian.ravel()


def medianFiltering(volumeSlice):
    median = ndimage.median_filter(volumeSlice, size=MATRIX_SIZE)
    median /= np.max(median)
    return median.ravel()


def sobelOperator(volumeSlice):
    sobel = ndimage.sobel(volumeSlice)
    sobel /= np.max(sobel)
    return sobel.ravel()


def mean(volumeSlice):
    meanValues = ndimage.uniform_filter(volumeSlice, (MATRIX_SIZE, MATRIX_SIZE))
    meanValues /= np.max(meanValues)
    return meanValues.ravel()


def variance(volumeSlice):
    # Implementation of and alternative variance formula: var = sum(X^2)/N - μ^2
    meanValues = ndimage.uniform_filter(volumeSlice, (MATRIX_SIZE, MATRIX_SIZE))
    sqr_mean = ndimage.uniform_filter(volumeSlice ** 2, (MATRIX_SIZE, MATRIX_SIZE))
    var = sqr_mean - meanValues ** 2
    var /= np.max(var)
    return var.ravel()


def laplacian(volumeSlice):
    laplacianValues = ndimage.laplace(volumeSlice)
    laplacianValues /= np.max(laplacianValues)
    return laplacianValues.ravel()


def extractChosenFeatures(data, chosen_features):
    # Simple registration and connection between chosen feature and actual processing method
    featureDictionary = {}
    if chosen_features['voxelValue']:
        featureDictionary['voxelValue'] = np.array(data).ravel()
    if chosen_features['mean']:
        featureDictionary['mean'] = mean(data)
    if chosen_features['variance']:
        featureDictionary['variance'] = variance(data)
    if chosen_features['gaussianFilter']:
        featureDictionary['gaussianFilter'] = gaussianFiltering(data)
    if chosen_features['medianFilter']:
        featureDictionary['medianFilter'] = medianFiltering(data)
    if chosen_features['sobelOperator']:
        featureDictionary['sobelOperator'] = sobelOperator(data)
    if chosen_features['gradientMatrix']:
        featureDictionary['gradientMatrix'] = countGradient(data)
    if chosen_features['laplacian']:
        featureDictionary['laplacian'] = laplacian(data)
    return featureDictionary


def prepareFeatures(inputSlices, chosenFeatures):
    numOfSlices = len(inputSlices)
    features = []
    slicesPrepared = 0
    print('-' * 30)
    for s in inputSlices:
        # These two methods are called by default
        data = toSigned(s)
        data = normalize(data)
        data = equalizeHistogram(data)
        featureDictionary = extractChosenFeatures(data, chosenFeatures)
        for j in range(data.size):
            featureRow = []
            for extractedFeature in featureDictionary.values():
                if isinstance(extractedFeature[0], np.ndarray):
                    featureRow.extend(extractedFeature[j])
                else:
                    featureRow.append(extractedFeature[j])
            features.append(featureRow)
        slicesPrepared += 1
        print('Prepared slices: {0} of {1} '.format(slicesPrepared, numOfSlices))
    print('-' * 30)
    return np.asarray(features)


def prepareMasks(maskSlices):
    numOfMasks = len(maskSlices)
    masks = []
    print('Preparing masks vector...')
    print('-' * 30)
    i = 0
    for mask in maskSlices:
        mask = mask.astype(np.uint8)
        mask = mask.ravel()
        masks.extend(mask)
        i += 1
        print('Prepared mask slices: {0} of {1} '.format(i, numOfMasks))
    print('-' * 30)
    return np.asarray(masks)


def chooseTrainingType(TRAINING_TYPE, parameters):
    if TRAINING_TYPE == 'FROM_SAVED_CLASSIFIER':
        return load(os.path.abspath(parameters['classifierPath']))

    estimators = parameters['estimators']
    svc = BaggingClassifier(
        svm.SVC(max_iter=parameters['maxIterations'], tol=parameters['tolerance'],
                cache_size=parameters['cacheSize'], kernel='rbf', shrinking=0, verbose=0),
        max_samples=1.0 / estimators, n_estimators=estimators, n_jobs=parameters['parallelJobs'])

    if TRAINING_TYPE == 'DEFAULT_TRAINING':
        return svc

    elif TRAINING_TYPE == 'GRID_SEARCH':
        gridSearchParams = {
            'base_estimator__C': parameters['c'],
            'base_estimator__gamma': parameters['gamma']
        }
        return GridSearchCV(svc, gridSearchParams, n_jobs=parameters['parallelJobs'],
                            cv=parameters['crossValidations'], verbose=1)
    else:
        raise ValueError('No training has been selected!')


def classify(featureStorage, predictionStorage):
    isValidationSetPresent = None
    if 'validationFeatureVector' in predictionStorage:
        isValidationSetPresent = True     
    TRAINING_TYPE = predictionStorage['TRAINING_TYPE']
    print('-' * 30)
    print('TRAINING TYPE: {0}'.format(TRAINING_TYPE))
    print('-' * 30) 
    xTrain = featureStorage['trainingFeatureVector']
    yTrain = featureStorage['trainingMaskVector']  
    # Standardize data
    scaler = preprocessing.StandardScaler().fit(xTrain)
    xTrain = scaler.transform(xTrain)
    xTrain = preprocessing.normalize(xTrain)
    xPrediction = scaler.transform(predictionStorage['predictionFeatureVector'])
    xPrediction = preprocessing.normalize(xPrediction)
    xValidation = []   
    if isValidationSetPresent:
        xValidation = scaler.transform(predictionStorage['validationFeatureVector'])
        xValidation = preprocessing.normalize(xValidation)  
    clf = chooseTrainingType(TRAINING_TYPE, predictionStorage['parameters'])
    if TRAINING_TYPE != 'FROM_SAVED_CLASSIFIER':
        print('Training started...')
        startTime = time.time()
        clf.fit(xTrain, yTrain)
        print('Training ended: {:.2f} s'.format(time.time() - start_time))
        print('-' * 30)
        if TRAINING_TYPE == 'GRID_SEARCH':
            print("GRID SEARCH RESULTS\n")
            predictionStorage['bestParameters'] = clf.best_params_
            print('Best parameters: {}\n'.format(clf.best_params_))
            means = clf.cv_results_['mean_test_score']
            for mean, params in zip(means, clf.cv_results_['params']):
                print('Mean score: {:0.3f} Parameters: {}'.format(mean, params))
            print('-' * 30)
        if 'classifierPath' in predictionStorage['parameters']:
            print('Saving classifier...')
            startTime = time.time()
            # TODO: Check path, should i used os.path, not sure if this line would work on a mac)
            dump(clf, predictionStorage['parameters']['classifierPath'])
            print('Saving  ended: {:.2f} s'.format(time.time() - start_time))
            print('-' * 30)

    if isValidationSetPresent:
        # TODO: REFACTOR - extract to method, try to validate at least two slices...
        print('Validation started...')
        startTime = time.time()
        yPredictedValidation = clf.predict(xValidation)
        yPredictedValidation = yPredictedValidation.reshape(-1,
                                                            predictionStorage['validationMask'].shape[1],
                                                            predictionStorage['validationMask'].shape[2])
        idx = 0
        diceParameter = 0
        for i in predictionStorage['validationSlices']:
            openedImg = ndimage.binary_opening(yPredictedValidation[idx])
            closedImg = ndimage.binary_closing(openedImg)
            diceParameter += dice(predictionStorage['validationMask'][idx], closedImg)
            idx += 1
        diceParameter = diceParameter / len(predictionStorage['validationSlices'])
        print('Validation ended %s' % (time.time() - startTime))
        print('-' * 30)
        print('DICE: ' + str(diceParameter))
        predictionStorage['DICE'] = diceParameter
        print('-' * 30)
    print('Predicting started...')
    startTime = time.time()
    predictedMasks = clf.predict(xPrediction)
    print('Predicting ended: {:.2f} s'.format(time.time() - start_time))
    return predictedMasks


def dice(ref, pred):
    intersection = np.logical_and(ref, pred)
    return 2. * intersection.sum() / (ref.sum() + pred.sum())


def prepareTrainingSet(featureStorage, pathToFeatureStorage):
    features, mask = prepareData('TRAINING_PREPARE_FEATURES', featureStorage, featureStorage['features'])
    # In case of creating new feature storage
    if featureStorage['trainingFeatureVector'] is None:
        featureStorage['trainingFeatureVector'] = features
        featureStorage['trainingMaskVector'] = mask
        featureStorage['trainingMaskVector'] = mask
    else:
        featureStorage['trainingFeatureVector'] = np.append(featureStorage['trainingFeatureVector'], features, axis=0)
        featureStorage['trainingMaskVector'] = np.append(featureStorage['trainingMaskVector'], mask)
    saveStorageNode(featureStorage, pathToFeatureStorage)


def runClassifying(featureStorage, pathToFeatureStorage, predictionStorage, pathToPredictionStorage):
    predictionStorage['predictionFeatureVector'] = prepareData('PREDICTION_PREPARE_FEATURES',
                                                               predictionStorage,
                                                               featureStorage['features'])
    if 'validationData' in predictionStorage:
        predictionStorage['validationFeatureVector'] = prepareData('VALIDATION_PREPARE_FEATURES',
                                                                   predictionStorage,
                                                                   featureStorage['features'])
    prediction = classify(featureStorage, predictionStorage)
    prediction = prediction.reshape(-1,
                                    predictionStorage['predictionMask'].shape[1],
                                    predictionStorage['predictionMask'].shape[2])
    idx = 0
    for i in predictionStorage['predictionSlices']:
        opened_img = ndimage.binary_opening(prediction[idx])
        closed_img = ndimage.binary_closing(opened_img)
        predictionStorage['predictionMask'][i] = closed_img
        idx += 1
    saveStorageNode(predictionStorage, pathToPredictionStorage)


if __name__ == '__main__':
    operationType = sys.argv[1]
    pathToFeatureStorage = sys.argv[2]
    pathToPredictionStorage = sys.argv[3]
    featureStorage = loadStorageNode(pathToFeatureStorage)
    predictionStorage = loadStorageNode(pathToPredictionStorage)
    # ================================================
    if operationType == 'PREPARING_TRAINING_SET':
        prepareTrainingSet(featureStorage, pathToFeatureStorage)
    # ===============================================
    elif operationType == 'CLASSIFYING':
        runClassifying(featureStorage, pathToFeatureStorage, predictionStorage, pathToPredictionStorage)
    # ===============================================
    else:
        raise ValueError('Invalid operation type!')
