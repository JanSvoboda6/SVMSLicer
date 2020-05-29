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

#TODO: Make better logging, cleanup
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
        # TODO: Proper naming of methods to camelcase
        print(str(storageNode['trainingData'].shape))
        return prepare_features(storageNode['trainingData'], features), prepare_mask(storageNode['trainingMask'])
    elif preparationType == 'PREDICTION_PREPARE_FEATURES':
        return prepare_features(storageNode['predictionData'], features)
    elif preparationType == 'VALIDATION_PREPARE_FEATURES':
        return prepare_features(storageNode['validationData'], features)
    else:
        raise ValueError('Type of feature preparation is not valid!')


def to_signed(img):
    minimum = np.min(img)
    if minimum < 0:
        img = img + abs(minimum)
    return img


def normalize(img):
    img = img.astype(np.float32)
    img /= np.max(img)
    return img


def count_gradient(img):
    # There is also gradient method in numpy module
    assert (MATRIX_SIZE % 2 == 1)
    gradients_x = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    gradients_y = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    gradients_feature = np.zeros((img.size, MATRIX_SIZE, MATRIX_SIZE), dtype=np.float32)
    middle = math.trunc(MATRIX_SIZE / 2)

    for i in range(img.shape[0]):
        for j in range(img.shape[1] - 1):
            gradients_x[i, j + 1] = math.fabs((img[i, j + 1]) - (img[i, j]))

    for i in range(img.shape[0] - 1):
        for j in range(img.shape[1]):
            gradients_y[i + 1, j] = math.fabs((img[i + 1, j]) - (img[i, j]))

    gradients_x = np.pad(gradients_x, middle, mode='constant')
    gradients_y = np.pad(gradients_y, middle, mode='constant')
    gradients_sum = (np.add(gradients_x, gradients_y)) / 2
    pixel = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gradients_feature[pixel] = gradients_sum[i:i + MATRIX_SIZE, j:j + MATRIX_SIZE]
            pixel += 1
    return gradients_feature.reshape(-1, MATRIX_SIZE * MATRIX_SIZE)


def gaussian_filtering(img):
    gaussian = ndimage.gaussian_filter(img, sigma=2)
    gaussian /= np.max(gaussian)
    return gaussian.ravel()


def median_filtering(img):
    median = ndimage.median_filter(img, size=7)
    median /= np.max(median)
    return median.ravel()


def sobel_operator(img):
    sobel = ndimage.sobel(img)
    sobel /= np.max(sobel)
    return sobel.ravel()

def mean(img):
    mean = ndimage.uniform_filter(img, (MATRIX_SIZE, MATRIX_SIZE))
    mean /= np.max(mean)
    return mean.ravel()

def variance(img):
    # Implementation of and alternative variance formula: var = sum(X^2)/N - μ^2
    mean = ndimage.uniform_filter(img, (MATRIX_SIZE, MATRIX_SIZE))
    sqr_mean = ndimage.uniform_filter(img ** 2, (MATRIX_SIZE, MATRIX_SIZE))
    var = sqr_mean - mean ** 2
    var /= np.max(var)
    return var.ravel()


def laplacian(img):
    laplacian = ndimage.laplace(img,)
    laplacian /= np.max(laplacian)
    return laplacian.ravel()


def extractChosenFeatures(data, chosen_features):
    # Simple registration and connection between chosen feature and actual processing method
    feature_dictionary = {}
    if chosen_features['voxelValue']:
        feature_dictionary['voxelValue'] = np.array(data).ravel()
    if chosen_features['mean']:
        feature_dictionary['mean'] = mean(data)
    if chosen_features['variance']:
        feature_dictionary['variance'] = variance(data)
    if chosen_features['gaussianFilter']:
        feature_dictionary['gaussianFilter'] = gaussian_filtering(data)
    if chosen_features['medianFilter']:
        feature_dictionary['medianFilter'] = median_filtering(data)
    if chosen_features['sobelOperator']:
        feature_dictionary['sobelOperator'] = sobel_operator(data)
    if chosen_features['gradientMatrix']:
        feature_dictionary['gradientMatrix'] = count_gradient(data)
    if chosen_features['laplacian']:
        feature_dictionary['laplacian'] = laplacian(data)
    return feature_dictionary


def prepare_features(input_data, chosen_features):
    num_of_images = len(input_data)
    features = []
    images_prepared = 0
    print('Preparing feature vector...')
    print('-' * 30)
    for slice in input_data:
        # This two methods are called by default
        data = to_signed(slice)
        data = normalize(data)
        feature_dictionary = extractChosenFeatures(data, chosen_features)
        for j in range(data.size):
            feature_row = []
            for extracted_feature in feature_dictionary.values():
                if isinstance(extracted_feature[0], np.ndarray):
                    feature_row.extend(extracted_feature[j])
                else:
                    feature_row.append(extracted_feature[j])
            features.append(feature_row)
        images_prepared += 1
        print('Prepared images: {0} of {1} '.format(images_prepared, num_of_images))
    print('-' * 30)
    return np.asarray(features)


def prepare_mask(training_mask):
    num_of_masks = len(training_mask)
    masks = []
    print('Preparing masks vector...')
    print('-' * 30)
    i = 0
    for mask in training_mask:
        mask = mask.astype(np.uint8)
        mask = mask.ravel()
        masks.extend(mask)
        i += 1
        print('Prepared masks: {0} of {1} '.format(i, num_of_masks))
    print('-' * 30)
    return np.asarray(masks)


def choose_training_type(TRAINING_TYPE, parameters):
    if TRAINING_TYPE == 'FROM_SAVED_CLASSIFIER':
        return load(os.path.abspath(parameters['classifierPath']))

    n_estimators = parameters['estimators']
    svc = BaggingClassifier(
        svm.SVC(max_iter=parameters['maxIterations'], tol=parameters['tolerance'],
                cache_size=parameters['cacheSize'], kernel='rbf', shrinking=0, verbose=0),
        max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=parameters['parallelJobs'])

    if TRAINING_TYPE == 'DEFAULT_TRAINING':
        return svc

    elif TRAINING_TYPE == 'GRID_SEARCH':
        grid_search_params = {
            'base_estimator__C': parameters['c'],
            'base_estimator__gamma': parameters['gamma']
        }
        return GridSearchCV(svc, grid_search_params, n_jobs=parameters['parallelJobs'],
                            cv=parameters['crossValidations'], verbose=1)
    else:
        raise ValueError('No training has been selected!')


def classify(featureStorage, predictionStorage):
    isValidationSetPresent = None
    if 'validationFeatureVector' in predictionStorage:
        isValidationSetPresent = True
    TRAINING_TYPE = predictionStorage['TRAINING_TYPE']
    print('Shape of prediction Vector ' + str(predictionStorage['predictionFeatureVector'].shape))
    print('-' * 20)
    print('TRAINING TYPE: {0}'.format(TRAINING_TYPE))
    print('-' * 20)
    x_train = featureStorage['trainingFeatureVector']
    y_train = featureStorage['trainingMaskVector']
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_train = preprocessing.normalize(x_train)
    # TODO: Rename testing to predicting
    x_predicting = scaler.transform(predictionStorage['predictionFeatureVector'])
    x_predicting = preprocessing.normalize(x_predicting)
    x_validation = []
    if isValidationSetPresent:
        x_validation = scaler.transform(predictionStorage['validationFeatureVector'])
        x_validation = preprocessing.normalize(x_validation)
    print('x train: ' + str(x_train.shape))
    print('y train: ' + str(y_train.shape))
    print('-' * 20)
    clf = choose_training_type(TRAINING_TYPE, predictionStorage['parameters'])

    if TRAINING_TYPE != 'FROM_SAVED_CLASSIFIER':
        print('Training started')
        print('-' * 20)
        start_time = time.time()
        clf.fit(x_train, y_train)
        print('Training ended: %s' % (time.time() - start_time))
        print('-' * 20)
        if TRAINING_TYPE == 'GRID_SEARCH':
            print('Best parameters')
            print(clf.best_params_)
            predictionStorage['bestParameters'] = clf.best_params_
            print('-' * 20)
        if 'classifierPath' in predictionStorage['parameters']:
            print('Saving model...')
            start_time = time.time()
            # TODO: Check path, should i used os.path, not sure if this line would work on a mac)
            dump(clf, predictionStorage['parameters']['classifierPath'])
            print('Saving  ended: %s' % (time.time() - start_time))
            print('-' * 20)

    if isValidationSetPresent:
        # TODO: REFACTOR - extract to method, try to validate at least two slices...
        print('Validation started')
        print('-' * 20)
        y_predicted_validation = clf.predict(x_validation)
        y_predicted_validation = y_predicted_validation.reshape(-1,
                                                                predictionStorage['validationMask'].shape[1],
                                                                predictionStorage['validationMask'].shape[2])
        idx = 0
        dice_param = 0
        print('predicted validation: ' + str(y_predicted_validation.shape))
        print('prediction storage:' + str(predictionStorage['validationMask'][idx].shape))
        for i in predictionStorage['validationSlices']:
            opened_img = ndimage.binary_opening(y_predicted_validation[idx])
            closed_img = ndimage.binary_closing(opened_img)
            dice_param += dice(predictionStorage['validationMask'][idx], closed_img)
            idx += 1
        dice_param = dice_param / len(predictionStorage['validationSlices'])
        print('DICE: ' + str(dice_param))
        predictionStorage['DICE'] = dice_param
        print('-' * 20)
    print('Predicting started')
    print('-' * 20)
    start_time = time.time()
    predicted_masks = clf.predict(x_predicting)
    # predicted_masks = parallel_predict(clf, x_testing, -1)
    print('Predicting ended: %s' % (time.time() - start_time))
    return predicted_masks


def dice(ref, pred):
    intersection = np.logical_and(ref, pred)
    return 2. * intersection.sum() / (ref.sum() + pred.sum())


if __name__ == '__main__':
    # TODO: Make pretty - all lines
    operationType = sys.argv[1]
    pathToFeatureStorage = sys.argv[2]
    pathToPredictionStorage = sys.argv[3]
    featureStorage = loadStorageNode(pathToFeatureStorage)
    predictionStorage = loadStorageNode(pathToPredictionStorage)
    # ================================================
    if operationType == 'PREPARING_TRAINING_SET':
        features, mask = prepareData('TRAINING_PREPARE_FEATURES', featureStorage, featureStorage['features'])
        # In case of creating new feature storage
        print(str(features.shape))
        if featureStorage['trainingFeatureVector'] is None:
            featureStorage['trainingFeatureVector'] = features
            featureStorage['trainingMaskVector'] = mask
        else:
            featureStorage['trainingFeatureVector'] = np.append(featureStorage['trainingFeatureVector'], features,
                                                                axis=0)
            featureStorage['trainingMaskVector'] = np.append(featureStorage['trainingMaskVector'], mask)
        saveStorageNode(featureStorage, pathToFeatureStorage)
    # ===============================================
    elif operationType == 'CLASSIFYING':
        predictionStorage['predictionFeatureVector'] = prepareData('PREDICTION_PREPARE_FEATURES', predictionStorage,
                                                                   featureStorage['features'])
        if 'validationData' in predictionStorage:
            predictionStorage['validationFeatureVector'] = prepareData('VALIDATION_PREPARE_FEATURES', predictionStorage,
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
    # ===============================================
    else:
        raise ValueError('Invalid operation type!')
