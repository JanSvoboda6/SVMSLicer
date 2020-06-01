import os
import logging
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import time

# Dependency modules check
logging.info('-' * 30)
logging.info('SVM Module: There is check needed whether all dependency modules are installed...')
dependencyModules = ['scipy', 'scikit-learn', 'numpy', 'joblib', 'pickle']
for module in dependencyModules:
    try:
        module_obj = __import__(module)
        logging.info('Module {0} was succesfully imported.'.format(module))
    except ImportError:
        logging.info("Module {0} was not found.\n Attempting to install {0}...".format(module))
        slicer.util.pip_install(module)
logging.info('SVM Module: All dependency modules should be installed! To be sure, check logging messages above.')
logging.info('-' * 30)

# Import dependency modules used in this file
import numpy as np
import pickle


class SVMClassifier(ScriptedLoadableModule):

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = 'SVMClassifier'
        self.parent.categories = ['Segmentation']
        self.parent.dependencies = []
        self.parent.contributors = ['Jan Svoboda (Brno University of Technology)']
        self.parent.helpText = "SVM Classifier"
        self.parent.helpText += self.getDefaultModuleDocumentationLink()
        self.parent.acknowledgementText = 'BUT Brno'


class SVMClassifierWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        # Init methods provided by Slicer
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.ui = None
        self.featureStorage = None
        self.featureStoragePath = None
        self.predictionStorage = None
        self.predictionStoragePath = None
        self.total_time = None
        self.messageBox = None
        self.checkedIdx = 0

    def setup(self):
        # Setup method provided by Slicer
        ScriptedLoadableModuleWidget.setup(self)

        # Load UI made in Qt Designer
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/SVMClassifier.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set the current MRML scene to the widget
        self.ui.trainingVolume.setMRMLScene(slicer.mrmlScene)
        self.ui.trainingMask.setMRMLScene(slicer.mrmlScene)
        self.ui.predictionVolume.setMRMLScene(slicer.mrmlScene)
        self.ui.predictionMask.setMRMLScene(slicer.mrmlScene)
        self.ui.validationVolume.setMRMLScene(slicer.mrmlScene)
        self.ui.validationMask.setMRMLScene(slicer.mrmlScene)

        # Training Data
        self.ui.trainingVolume.connect('currentNodeChanged(vtkMRMLNode*)', self.updateStateOfPrepareFeaturesButton)
        self.ui.trainingVolume.connect('currentNodeChanged(vtkMRMLNode*)', self.orientVolumeNodeToSagittal)
        self.ui.trainingMask.connect('currentNodeChanged(vtkMRMLNode*)', self.updateStateOfPrepareFeaturesButton)
        self.ui.trainingMask.connect('currentNodeChanged(vtkMRMLNode*)', self.orientVolumeNodeToSagittal)
        self.ui.createFeatureStorageButton.connect('clicked(bool)', self.onFeatureStorageButtons)
        self.ui.addToFeatureStorageButton.connect('clicked(bool)', self.onFeatureStorageButtons)
        self.ui.loadFeatureStorageButton.connect('clicked(bool)', self.onFeatureStorageButtons)
        self.ui.featureStorageFileBrowserButton.connect('clicked(bool)', self.onfeatureStorageFileBrowserButton)
        self.ui.prepareFeaturesButton.connect('clicked(bool)', self.onPrepareFeaturesButton)

        # Features
        self.ui.voxelValueCheckBox.connect('clicked(bool)', self.onFeaturesCheckBoxes)
        self.ui.meanCheckBox.connect('clicked(bool)', self.onFeaturesCheckBoxes)
        self.ui.varianceCheckBox.connect('clicked(bool)', self.onFeaturesCheckBoxes)
        self.ui.gaussianFilterCheckBox.connect('clicked(bool)', self.onFeaturesCheckBoxes)
        self.ui.medianFilterCheckBox.connect('clicked(bool)', self.onFeaturesCheckBoxes)
        self.ui.sobelOperatorCheckBox.connect('clicked(bool)', self.onFeaturesCheckBoxes)
        self.ui.gradientMatrixCheckBox.connect('clicked(bool)', self.onFeaturesCheckBoxes)
        self.ui.laplacianCheckBox.connect('clicked(bool)', self.onFeaturesCheckBoxes)

        # Training
        self.ui.validationVolume.connect('currentNodeChanged(vtkMRMLNode*)', self.orientVolumeNodeToSagittal)
        self.ui.validationMask.connect('currentNodeChanged(vtkMRMLNode*)', self.orientVolumeNodeToSagittal)
        self.ui.defaultTrainingButton.connect('clicked(bool)', self.onTrainingButtons)
        self.ui.gridSearchTrainingButton.connect('clicked(bool)', self.onTrainingButtons)
        self.ui.noSavingButton.connect('clicked(bool)', self.onSavingLoadingClassifierButtons)
        self.ui.saveClassifierButton.connect('clicked(bool)', self.onSavingLoadingClassifierButtons)
        self.ui.loadClassifierButton.connect('clicked(bool)', self.onSavingLoadingClassifierButtons)
        self.ui.classifierFileBrowseButton.connect('clicked(bool)', self.onClassifierFileBrowserButton)

        # Default values of parameters
        self.ui.cParameterText.setText('10')
        self.ui.gammaParameterText.setText('1')
        self.ui.parallelJobsText.setText('-1')
        self.ui.estimatorsText.setText('16')
        self.ui.cacheSizeText.setText('500')
        self.ui.toleranceText.setText('0.5')
        self.ui.maxIterationsText.setText('15000')

        # Predicting
        self.ui.predictionVolume.connect('currentNodeChanged(vtkMRMLNode*)', self.updateStateOfRunClassifierButton)
        self.ui.predictionVolume.connect('currentNodeChanged(vtkMRMLNode*)', self.orientVolumeNodeToSagittal)
        self.ui.predictionMask.connect('currentNodeChanged(vtkMRMLNode*)', self.updateStateOfRunClassifierButton)
        self.ui.runClassifierButton.connect('clicked(bool)', self.onRunClassifierButton)

    def cleanup(self):
        # Method provided by Slicer - called when the application closes and the module widget is destroyed
        self.removeObservers()

    # region TRAINING DATA SECTION =====================================================================================
    def onFeaturesCheckBoxes(self):
        totalChecked = 0
        for index in range(self.ui.featuresCheckBoxesLayout.count()):
            item = self.ui.featuresCheckBoxesLayout.itemAt(index)
            widget = item.widget()
            if widget.checked:
                self.checkedIdx = index
                totalChecked += 1
        if totalChecked is 0:
            item = self.ui.featuresCheckBoxesLayout.itemAt(self.checkedIdx)
            widget = item.widget()
            widget.checked = True
            self.showMessageBox(severity='WARNING', message='At least one feature must be selected!')

    def onFeatureStorageButtons(self):
        self.ui.featureStoragePathText.setText('')
        self.updateStateOfFeatureCheckBoxes()
        self.updateStateOfPrepareFeaturesButton()
        self.updateStateOfRunClassifierButton()
        if self.ui.loadFeatureStorageButton.checked:
            self.disableInputOfTrainingData()
        else:
            self.enableInputOfTrainingData()

    def enableInputOfTrainingData(self):
        self.ui.trainingVolume.enabled = True
        self.ui.trainingSliceNumbersText.enabled = True
        self.ui.trainingMask.enabled = True

    def disableInputOfTrainingData(self):
        self.ui.trainingVolume.enabled = False
        self.ui.trainingSliceNumbersText.enabled = False
        self.ui.trainingMask.enabled = False

    def updateStateOfFeatureCheckBoxes(self):
        if self.ui.featureStoragePathText.toPlainText() is not '':
            if self.ui.addToFeatureStorageButton.checked or self.ui.loadFeatureStorageButton.checked:
                self.checkActiveFeaturesCheckBoxes()
                self.enableCheckBoxes(False)
                return
        self.enableCheckBoxes(True)

    def enableCheckBoxes(self, isEnabled):
        for index in range(self.ui.featuresCheckBoxesLayout.count()):
            item = self.ui.featuresCheckBoxesLayout.itemAt(index)
            widget = item.widget()
            widget.enabled = isEnabled

    def checkActiveFeaturesCheckBoxes(self):
        path = self.ui.featureStoragePathText.toPlainText()
        if path is '':
            return
        features = self.loadStorage(path)['features']
        self.ui.voxelValueCheckBox.checked = features['voxelValue']
        self.ui.meanCheckBox.checked = features['mean']
        self.ui.varianceCheckBox.checked = features['variance']
        self.ui.gaussianFilterCheckBox.checked = features['gaussianFilter']
        self.ui.medianFilterCheckBox.checked = features['medianFilter']
        self.ui.sobelOperatorCheckBox.checked = features['sobelOperator']
        self.ui.gradientMatrixCheckBox.checked = features['gradientMatrix']
        self.ui.laplacianCheckBox.checked = features['laplacian']

    def onfeatureStorageFileBrowserButton(self):
        valid, filePath = self.getFeatureStorageFile()
        if not valid:
            self.ui.featureStoragePathText.setText('')
            self.showMessageBox(severity='WARNING', message='No file has been selected.')
        else:
            self.ui.featureStoragePathText.setText(filePath)
        self.updateStateOfFeatureCheckBoxes()
        self.updateStateOfPrepareFeaturesButton()
        self.updateStateOfRunClassifierButton()

    def getFeatureStorageFile(self):
        valid = None
        filePath = None
        if self.ui.createFeatureStorageButton.checked:
            filePath = qt.QFileDialog().getSaveFileName(None, 'Create New File', 'feature_storage', '*.pickle')
            if not filePath:
                return False, None
            return True, filePath

        if self.ui.addToFeatureStorageButton.checked:
            valid, filePath = self.getFeatureStorageFileName()

        elif self.ui.loadFeatureStorageButton.checked:
            valid, filePath = self.getFeatureStorageFileName()

        if not valid:
            return False, None

        return True, filePath

    def getFeatureStorageFileName(self):
        filePath = qt.QFileDialog().getOpenFileName(None, 'Open File', '', '*.pickle')
        if not filePath:
            return False, None
        return True, filePath

    def updateStateOfPrepareFeaturesButton(self):
        if self.ui.featureStoragePathText.toPlainText() is not "":
            if self.ui.loadFeatureStorageButton.checked:
                self.ui.prepareFeaturesButton.enabled = False
                return
            elif self.ui.trainingVolume.currentNode() and self.ui.trainingMask.currentNode():
                self.ui.prepareFeaturesButton.enabled = True
                return
        self.ui.prepareFeaturesButton.enabled = False

    def onPrepareFeaturesButton(self):
        try:
            self.total_time = time.time()
            self.ui.prepareFeaturesButton.enabled = False
            if not self.validateBeforePreparation():
                self.showMessageBox(severity='ERROR',
                                    message='There was a problem with a preparation of features.'
                                            'Please ensure that all parameters are in correct format.')
                return
            if self.ui.createFeatureStorageButton.checked:
                self.featureStorage = self.getEmptyFeatureStorage()
                self.featureStorage['features'] = self.getCheckedFeatures()
            else:
                self.featureStorage = self.loadStorage(self.ui.featureStoragePathText.toPlainText())

            slices = self.getParsedSlices(self.ui.trainingVolume.currentNode(),
                                          self.ui.trainingSliceNumbersText.toPlainText())
            self.fillFeatureStorageWithTrainingData(slices)
            self.saveFeatureStorage(self.ui.featureStoragePathText.toPlainText())
            self.featureStoragePath = self.ui.featureStoragePathText.toPlainText()
            cliNodeParameters = {
                'operationType': 'PREPARING_TRAINING_SET',
                'pathToFeatureStorage': self.featureStoragePath
            }
            cliNode = slicer.cli.run(slicer.modules.background_classifier, parameters=cliNodeParameters)
            self.ui.featurePreparationProgressBar.setCommandLineModuleNode(cliNode)
            cliNode.AddObserver('ModifiedEvent', self.checkFeaturePreparationStatus)
        except Exception as e:
            self.updateStateOfPrepareFeaturesButton()
            raise ValueError('Failed to prepare features!')

    def validateBeforePreparation(self):
        if not self.ui.trainingVolume.currentNode() or not self.ui.trainingMask.currentNode():
            self.showMessageBox(severity='ERROR', message='Training Volume or Training Mask is not selected!')
            return False
        trainingVolume = slicer.util.arrayFromVolume(self.ui.trainingVolume.currentNode())
        trainingMask = slicer.util.arrayFromVolume(self.ui.trainingMask.currentNode())
        if not self.isShapeOfVolumesEqual(trainingVolume, trainingMask):
            self.showMessageBox(severity='ERROR', message='Shape of Training Volume is different from shape of'
                                                          ' Training Mask!')
            return False
        return True

    def getEmptyFeatureStorage(self):
        return {
            'trainingData': None,
            'trainingMask': None,
            'trainingFeatureVector': None,
            'trainingMaskVector': None,
            'features': {}
        }

    def fillFeatureStorageWithTrainingData(self, slices):
        self.featureStorage['trainingData'] = self.getDataFromVolumeNode(self.ui.trainingVolume.currentNode(), slices)
        self.featureStorage['trainingMask'] = self.getDataFromVolumeNode(self.ui.trainingMask.currentNode(), slices)

    def getCheckedFeatures(self):
        return {
            'voxelValue': self.ui.voxelValueCheckBox.checked,
            'mean': self.ui.meanCheckBox.checked,
            'variance': self.ui.varianceCheckBox.checked,
            'gaussianFilter': self.ui.gaussianFilterCheckBox.checked,
            'medianFilter': self.ui.medianFilterCheckBox.checked,
            'sobelOperator': self.ui.sobelOperatorCheckBox.checked,
            'gradientMatrix': self.ui.gradientMatrixCheckBox.checked,
            'laplacian': self.ui.laplacianCheckBox.checked
        }

    def checkFeaturePreparationStatus(self, caller, event):
        if caller.IsA('vtkMRMLCommandLineModuleNode'):
            if caller.GetStatusString() == 'Completed':
                try:
                    self.featureStorage = self.loadStorage(self.featureStoragePath)
                    # At this point, we can delete training data and mask, we don't need them anymore
                    self.featureStorage['trainingData'] = None
                    self.featureStorage['trainingMask'] = None
                    self.saveFeatureStorage(self.featureStoragePath)
                    self.showMessageBox(severity='INFO', message=self.getAfterPreparationMessage())
                    self.updateStateOfPrepareFeaturesButton()
                except Exception as e:
                    caller.GetErrorText()
                    self.updateStateOfPrepareFeaturesButton()
                    raise Exception()

            elif caller.GetStatusString() != 'Running':
                # No freeze in case of error
                logging.info(caller.GetErrorText())
                self.updateStateOfPrepareFeaturesButton()

    def getAfterPreparationMessage(self):
        shapeOfFeatures = self.featureStorage['trainingFeatureVector'].shape
        message = '''\
          Features have been sucesfully prepared!
          Total time: ~ {} sec
          Size of whole feature vector: {} 
          Size of a feature row: {}\
          '''.format(int(time.time() - self.total_time), shapeOfFeatures[0], shapeOfFeatures[1])
        return message

    # endregion TRAINING DATA SECTION===================================================================================

    # region TRAINING SECTION ==========================================================================================
    def onSavingLoadingClassifierButtons(self):
        self.ui.classifierPathText.setText("")
        if self.ui.noSavingButton.checked:
            self.updateParametersState(True)
            self.ui.classifierPathText.enabled = False
            self.ui.classifierFileBrowseButton.enabled = False
            self.onTrainingButtons()

        elif self.ui.saveClassifierButton.checked:
            self.updateParametersState(True)
            self.ui.classifierPathText.enabled = True
            self.ui.classifierFileBrowseButton.enabled = True
            self.onTrainingButtons()

        elif self.ui.loadClassifierButton.checked:
            self.updateParametersState(False)
            self.ui.classifierPathText.enabled = True
            self.ui.classifierFileBrowseButton.enabled = True

        self.updateStateOfRunClassifierButton()

    def updateParametersState(self, isEnabled):
        for index in range(self.ui.parametersLayout.count()):
            item = self.ui.parametersLayout.itemAt(index)
            widget = item.widget()
            widget.enabled = isEnabled
        self.ui.defaultTrainingButton.enabled = isEnabled
        self.ui.gridSearchTrainingButton.enabled = isEnabled

    def onTrainingButtons(self):
        if self.ui.defaultTrainingButton.checked:
            self.ui.cParameterText.enabled = True
            self.ui.gammaParameterText.enabled = True
            self.ui.cParametersText.enabled = False
            self.ui.gammaParametersText.enabled = False
            self.ui.crossValidationsText.enabled = False

        elif self.ui.gridSearchTrainingButton.checked:
            self.ui.cParameterText.enabled = False
            self.ui.gammaParameterText.enabled = False
            self.ui.cParametersText.enabled = True
            self.ui.gammaParametersText.enabled = True
            self.ui.crossValidationsText.enabled = True

    def onClassifierFileBrowserButton(self):
        valid, path = self.getClassifierFile()
        if not valid:
            self.ui.classifierPathText.setText('')
            self.showMessageBox(severity='WARNING', message='No file has been selected.')
        else:
            self.ui.classifierPathText.setText(path)
        self.updateStateOfRunClassifierButton()

    def getClassifierFile(self):
        filePath = None
        if self.ui.saveClassifierButton.checked:
            filePath = qt.QFileDialog().getSaveFileName(None, 'Create New File', 'classifier', '*.joblib')

        elif self.ui.loadClassifierButton.checked:
            filePath = qt.QFileDialog().getOpenFileName(None, 'Load File', 'classifier', '*.joblib')

        if not filePath:
            return False, None
        return True, filePath

    # endregion TRAINING SECTION========================================================================================

    # region PREDICTING SECTION ========================================================================================
    def onRunClassifierButton(self):
        try:
            self.total_time = time.time()
            self.ui.runClassifierButton.enabled = False
            self.fillPredictionStorageWithData()
            if not self.validateBeforeRunning():
                self.updateStateOfRunClassifierButton()
                return
            self.featureStoragePath = self.ui.featureStoragePathText.toPlainText()
            # predictionStorage is used just for saving data temporarily to be able to send data to cliNode,
            # after predicting will be predictionStorage deleted. Current working directory will be used as
            # path to predictionStorage.
            self.predictionStoragePath = os.path.join(os.path.abspath(os.getcwd()), 'temporaryPredictionStorage.pickle')
            self.savePredictionStorage(self.predictionStoragePath)
            cliNodeParameters = {
                'operationType': 'CLASSIFYING',
                'pathToFeatureStorage': self.featureStoragePath,
                'pathToPredictionStorage': self.predictionStoragePath
            }
            cliNode = slicer.cli.run(slicer.modules.background_classifier, parameters=cliNodeParameters)
            cliNode.AddObserver('ModifiedEvent', self.checkClassifierStatus)
            self.ui.classifyingProgressBar.setCommandLineModuleNode(cliNode)
        except Exception as e:
            self.updateStateOfRunClassifierButton()
            raise ValueError('Failed to compute results!')

    def validateBeforeRunning(self):
        if not self.isPredictionStorageFilled():
            return False

        if not self.validateParametersValues(self.predictionStorage['parameters']):
            return False

        if 'validationVolume' in self.predictionStorage:
            validationVolume = slicer.util.arrayFromVolume(self.ui.validationVolume.currentNode())
            validationMask = slicer.util.arrayFromVolume(self.ui.validationMask.currentNode())
            if not self.isShapeOfVolumesEqual(validationVolume, validationMask):
                self.showMessageBox(severity='ERROR', message='Shape of Validation Volume is different from shape of'
                                                              ' Validation Mask!')
                return False

        return True

    def checkClassifierStatus(self, caller, event):
        if caller.IsA('vtkMRMLCommandLineModuleNode'):
            logging.info('Status is %s' % caller.GetStatusString())
            if caller.GetStatusString() == 'Completed':
                try:
                    self.predictionStorage = self.loadStorage(self.predictionStoragePath)
                    slicer.util.updateVolumeFromArray(self.ui.predictionMask.currentNode(),
                                                      self.predictionStorage['predictionMask'])
                    slicer.util.setSliceViewerLayers(self.ui.predictionMask.currentNode())
                    slicer.util.resetSliceViews()
                    self.showMessageBox(severity='INFO', message=self.getAfterPredictionMessage())
                    self.updateStateOfRunClassifierButton()
                    os.remove(self.predictionStoragePath)
                    self.predictionStorage = None
                except Exception as e:
                    self.updateStateOfRunClassifierButton()
                    raise Exception()

            elif caller.GetStatusString() != 'Running':
                # No freeze in case of error
                self.updateStateOfRunClassifierButton()

    def getAfterPredictionMessage(self):
        message = 'Prediction Volume was successfully classified!\nTotal time: ~ {} sec'.format(
            int(time.time() - self.total_time))
        if 'DICE' in self.predictionStorage:
            message += '\nDICE: {:.3f}'.format(self.predictionStorage['DICE'])
        if self.predictionStorage['TRAINING_TYPE'] == 'GRID_SEARCH':
            message += '\nBEST PARAMETERS:  C: {}   Gamma: {}'.format(
                self.predictionStorage['bestParameters']['base_estimator__C'],
                self.predictionStorage['bestParameters']['base_estimator__gamma'])
        return message

    def updateStateOfRunClassifierButton(self):
        if self.isPredictionSetFilled():
            if self.ui.saveClassifierButton.checked or self.ui.loadClassifierButton.checked:
                if self.ui.classifierPathText.toPlainText() is not "":
                    self.ui.runClassifierButton.enabled = True
                    return
                else:
                    self.ui.runClassifierButton.enabled = False
                    return
            else:
                self.ui.runClassifierButton.enabled = True
                return
        else:
            self.ui.runClassifierButton.enabled = False

    def fillPredictionStorageWithData(self):
        try:
            predictionSlices = self.getParsedSlices(self.ui.predictionVolume.currentNode(),
                                                    self.ui.predictionSliceNumbersText.toPlainText())
            self.predictionStorage = {'predictionData': self.getDataFromVolumeNode(self.ui.predictionVolume.currentNode(),
                                                                                   predictionSlices),
                                      'predictionMask': self.getEmptyPredictionMask(),
                                      'predictionSlices': predictionSlices}

            if self.isValidationSetFilled():
                validationSlices = self.getParsedSlices(self.ui.validationVolume.currentNode(),
                                                        self.ui.validationSliceNumbersText.toPlainText())
                self.predictionStorage['validationData'] = self.getDataFromVolumeNode(
                    self.ui.validationVolume.currentNode(),
                    validationSlices)
                self.predictionStorage['validationMask'] = self.getDataFromVolumeNode(
                    self.ui.validationMask.currentNode(),
                    validationSlices)
                self.predictionStorage['validationSlices'] = validationSlices

            if self.ui.loadClassifierButton.checked:
                self.predictionStorage['TRAINING_TYPE'] = 'FROM_SAVED_CLASSIFIER'
                self.predictionStorage['parameters'] = {'classifierPath': self.ui.classifierPathText.toPlainText()}
                return

            TRAINING_TYPE = None
            parameters = {
                'c': None,
                'gamma': None,
                'parallelJobs': None,
                'estimators': None,
                'cacheSize': None,
                'tolerance': None,
                'maxIterations': None
            }

            if self.ui.saveClassifierButton.checked:
                parameters['classifierPath'] = self.ui.classifierPathText.toPlainText()

            if self.ui.defaultTrainingButton.checked:
                TRAINING_TYPE = 'DEFAULT_TRAINING'
                parameters['c'] = float(self.ui.cParameterText.toPlainText())
                parameters['gamma'] = float(self.ui.gammaParameterText.toPlainText())

            elif self.ui.gridSearchTrainingButton.checked:
                c = self.getParsedArrayOfFloats(self.ui.cParametersText.toPlainText())
                gamma = self.getParsedArrayOfFloats(self.ui.gammaParametersText.toPlainText())
                TRAINING_TYPE = 'GRID_SEARCH'
                parameters['c'] = c
                parameters['gamma'] = gamma
                parameters['crossValidations'] = int(self.ui.crossValidationsText.toPlainText())

            # Rest of parameters
            parameters['estimators'] = int(self.ui.estimatorsText.toPlainText())
            parameters['cacheSize'] = float(self.ui.cacheSizeText.toPlainText())
            parameters['tolerance'] = float(self.ui.toleranceText.toPlainText())
            parameters['maxIterations'] = int(self.ui.maxIterationsText.toPlainText())
            parameters['parallelJobs'] = int(self.ui.parallelJobsText.toPlainText())

            self.predictionStorage['TRAINING_TYPE'] = TRAINING_TYPE
            self.predictionStorage['parameters'] = parameters

        except Exception as e:
            self.showMessageBox(severity='ERROR',
                                message='There was a problem with parsing parameters. '
                                        'Please ensure that all parameters are in correct format.')
            raise ValueError('There was a problem with parsing parameters. ')

    def isPredictionStorageFilled(self):
        for param in self.predictionStorage.values():
            if isinstance(param, dict):
                for nestedParam in param.values():
                    if str(nestedParam) is '':
                        self.showMessageBox(severity='ERROR', message='One or more parameters is unfilled!')
                        return False
            else:
                if str(param) is '':
                    self.showMessageBox(severity='ERROR', message='One or more parameters is unfilled!')
                    return False
        return True

    def getEmptyPredictionMask(self):
        # TODO: Just clean up slices that are for predicting, no need to make empty mask
        # TODO: if prediction volume shape == mask volume shape ---> means that mask volume has been already created
        if self.ui.predictionMask.currentNode() is None:
            raise ValueError('Prediction mask is invalid')
        self.ui.predictionMask.currentNode().CopyOrientation(self.ui.predictionVolume.currentNode())
        mask_voxels = np.zeros(slicer.util.arrayFromVolume(self.ui.predictionVolume.currentNode()).shape)
        slicer.util.updateVolumeFromArray(self.ui.predictionMask.currentNode(), mask_voxels)
        return slicer.util.arrayFromVolume(self.ui.predictionMask.currentNode())

    def isPredictionSetFilled(self):
        return (self.ui.predictionVolume.currentNode() is not None
                and self.ui.predictionMask.currentNode() is not None
                and self.ui.featureStoragePathText.toPlainText() is not '')

    def isValidationSetFilled(self):
        return (self.ui.validationVolume.currentNode()
                and self.ui.validationMask.currentNode()
                and self.ui.validationSliceNumbersText.toPlainText() is not '')

    # endregion PREDICTING SECTION  ====================================================================================

    # region UTILITY METHODS ===========================================================================================

    def getDataFromVolumeNode(self, volumeNode, slices):
        if volumeNode is None:
            raise ValueError('Volume is invalid')
        voxels = slicer.util.arrayFromVolume(volumeNode)
        selected_voxels = voxels[slices][:][:]
        return selected_voxels

    def getParsedSlices(self, volumeNode, slices):
        if not slices:
            self.showMessageBox(severity='ERROR', message='No Slice Numbers have been selected!')
            raise ValueError('No Slice Numbers have been selected!')
        try:
            if '-' in slices:
                borderSlices = slices.replace(' ', '').split('-')
                borderSlices = [int(i) for i in borderSlices]
                if borderSlices[0] > borderSlices[1]:
                    self.showMessageBox(severity='ERROR',
                                        message='Starting slice number must be smaller than ending slice number!')
                    raise ValueError('Slice Numbers are not in correct format!')
                slices = [*range(borderSlices[0], borderSlices[1] + 1)]
            else:
                slices = slices.replace(' ', '').split(',')
                slices = [int(i) for i in slices]
        except Exception as e:
            self.showMessageBox(severity='ERROR', message='Slice Numbers are not in correct format!')
            raise ValueError('Slice Numbers are not in correct format!')
        max_slice_number = slicer.util.arrayFromVolume(volumeNode).shape[0]
        for s in slices:
            if s < 0:
                self.showMessageBox(severity='ERROR', message='Slice Number cannot be negative!')
                raise ValueError('Slice Number cannot be negative!')
            if s > max_slice_number:
                self.showMessageBox(severity='ERROR',
                                    message='Slice Number cannot be larger than total number of slices! Number has to '
                                            'be between 0 - ' + str(max_slice_number) + '.')
                raise ValueError(
                    'Slice number cannot be larger than total number of slices! Number has to be between 0 - ' + str(
                        max_slice_number) + '.')
        return slices

    def getParsedArrayOfFloats(self, text):
        text = text.replace(' ', '').split(',')
        return [float(i) for i in text]

    def showMessageBox(self, severity, message):
        # INFO severity is default, INFO is not appended to the message
        if severity == 'WARNING':
            message = 'Warning: ' + message
        elif severity == 'ERROR':
            message = 'Error: ' + message
        self.messageBox = qt.QMessageBox()
        self.messageBox.setText(message)
        self.messageBox.setModal(False)
        self.messageBox.show()

    def loadStorage(self, pathToFeatureStorage):
        return pickle.load(open(pathToFeatureStorage, 'rb'))

    def saveFeatureStorage(self, pathToFeatureStorage):
        with open(pathToFeatureStorage, 'wb') as f:
            pickle.dump(self.featureStorage, f, pickle.HIGHEST_PROTOCOL)

    def savePredictionStorage(self, pathToPredictionStorage):
        with open(pathToPredictionStorage, 'wb') as f:
            pickle.dump(self.predictionStorage, f, pickle.HIGHEST_PROTOCOL)

    def orientVolumeNodeToSagittal(self, volumeNode):
        """
        The internal logic of the module operates in the sagittal view,
        if volumes are not correctly oriented there could be errors connected
        to index out of bound exception when logic tries to parse chosen slices.
        """
        if volumeNode is None:
            return
        self.deactivateSliceViewDisplayUpdating()
        parameters = {'inputVolume1': volumeNode,
                      'outputVolume': volumeNode,
                      'orientation': 'Sagittal'}
        cliNode = slicer.cli.run(slicer.modules.orientscalarvolume, None, parameters)
        cliNode.AddObserver('ModifiedEvent', self.checkOrientingStatus)

    def checkOrientingStatus(self, caller, event):
        if caller.IsA('vtkMRMLCommandLineModuleNode'):
            if caller.GetStatusString() == 'Completed':
                self.activateSliceViewDisplayUpdating()

    def deactivateSliceViewDisplayUpdating(self):
        """
        This is an utility method that avoids blinking after orienting volume. Otherwise script that orients volumes
        would update each slice view, currently displayed volume would change to oriented one and that would
        cause an unpleasant blink.
        """
        layoutManager = slicer.app.layoutManager()
        sliceWidgetNames = ['Red', 'Green', 'Yellow']
        for sliceWidgetName in sliceWidgetNames:
            sliceView = layoutManager.sliceWidget(sliceWidgetName)
            if sliceView is None:
                continue
            sliceLogic = sliceView.sliceLogic()
            compositeNode = sliceLogic.GetSliceCompositeNode()
            compositeNode.SetDoPropagateVolumeSelection(False)

    def activateSliceViewDisplayUpdating(self):
        layoutManager = slicer.app.layoutManager()
        sliceWidgetNames = ['Red', 'Green', 'Yellow']
        for sliceWidgetName in sliceWidgetNames:
            sliceView = layoutManager.sliceWidget(sliceWidgetName)
            if sliceView is None:
                continue
            sliceLogic = sliceView.sliceLogic()
            compositeNode = sliceLogic.GetSliceCompositeNode()
            compositeNode.SetDoPropagateVolumeSelection(True)

    def validateParametersValues(self, parameters):
        corruptedParameters = []
        for i in parameters.keys():
            if isinstance(parameters[i], str):
                if parameters[i] is '':
                    corruptedParameters.append(i)
            elif type(parameters[i]) is not list:
                # for proper cross validation there has to be at least two sets
                if i == 'crossValidations' and parameters[i] < 2:
                    corruptedParameters.append(i)
                elif parameters[i] <= 0:
                    # value of paralellJobs can be -1 for maximum
                    if i == 'parallelJobs' and parameters[i] == -1:
                        continue
                    else:
                        corruptedParameters.append(i)
            else:
                for j in parameters[i]:
                    if j <= 0:
                        corruptedParameters.append(i)

        if len(corruptedParameters):
            self.showMessageBox(severity='ERROR',
                                message='Values of following parameters are not in correct format:\n '
                                        + str(corruptedParameters))
            return False
        else:
            return True

    def isShapeOfVolumesEqual(self, volumeA, volumeB):
        return volumeA.shape == volumeB.shape

    # endregion UTILITY METHODS ========================================================================================


class SVMClassifierTest(ScriptedLoadableModuleTest):

    def setUp(self):
        # Clear scene, called initially by Slicer
        slicer.mrmlScene.Clear(0)

    def createTestingVolumeNode(self, data):
        volumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode')
        volumeNode.CreateDefaultDisplayNodes()
        slicer.util.updateVolumeFromArray(volumeNode, data)
        return volumeNode

    def getTrainingData(self):
        return np.array([[[0.2, 1, 1, 1, 0.1],
                          [0.2, 1, 1, 1, 0.1],
                          [0.2, 1, 1, 1, 0.1]],
                         [[0.1, 1, 1, 1, 0.1],
                          [0.1, 1, 1, 1, 0.1],
                          [0.2, 1, 1, 1, 0.1]]])

    def getTrainingMask(self):
        return np.array([[[0, 1, 1, 1, 0],
                          [0, 1, 1, 1, 0],
                          [0, 1, 1, 1, 0]],
                         [[0, 1, 1, 1, 0],
                          [0, 1, 1, 1, 0],
                          [0, 1, 1, 1, 0]]])

    def getValidationData(self):
        return np.array([[[0.1, 0.9, 0.1, 0.8, 0.1],
                          [0.2, 0.6, 0.9, 0.8, 0.1],
                          [0.2, 0.5, 0.9, 0.9, 0.1]],
                         [[0.1, 0.6, 0.6, 0.8, 0.2],
                          [0.1, 0.7, 1.0, 0.9, 0.2],
                          [0.2, 0.9, 0.9, 0.9, 0.2]]])

    def getValidationMask(self):
        return np.array([[[0, 1, 1, 1, 0],
                          [0, 1, 1, 1, 0],
                          [0, 1, 1, 1, 0]],
                         [[0, 1, 1, 1, 0],
                          [0, 1, 1, 1, 0],
                          [0, 1, 1, 1, 0]]])

    def getPredictionData(self):
        return np.array([[[0.1, 1, 1, 1, 0.2],
                          [0.1, 1, 1, 1, 0.3],
                          [0.3, 1, 1, 1, 0.3]],
                         [[0.1, 1, 1, 1, 0.1],
                          [0.1, 1, 1, 1, 0.2],
                          [0.2, 1, 1, 1, 0.2]]])

    def getEmptyPredictionMask(self):
        return np.zeros([2, 3, 5])

    def getTrainingFeatureVector(self):
        return np.array([[0.222], [1.], [1.], [1.], [0.111], [0.222], [1.], [1.], [1.], [0.111], [0.222], [1.],
                         [1.], [1.], [0.111], [0.111], [1.], [1.], [1.], [0.111], [0.111], [1.], [1.], [1.], [0.111],
                         [0.222], [1.], [1.], [1.], [0.111]])

    def getTrainingMaskVector(self):
        return np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0])

    def getAllFeatures(self):
        return {
            'voxelValue': True,
            'mean': True,
            'variance': True,
            'gaussianFilter': True,
            'medianFilter': True,
            'sobelOperator': True,
            'gradientMatrix': True,
            'laplacian': True
        }

    def getFeatureStorage(self):
        return {
            'trainingData': self.getTrainingData(),
            'trainingMask': self.getTrainingMask(),
            'trainingFeatureVector': None,
            'trainingMaskVector': None,
            'features': self.getAllFeatures()
        }

    def getFeatureStorageWithVoxelValueEnabled(self):
        return {
            'trainingFeatureVector': self.getTrainingFeatureVector(),
            'trainingMaskVector': self.getTrainingMaskVector(),
            'features': {
                'voxelValue': True,
                'mean': False,
                'variance': False,
                'gaussianFilter': False,
                'medianFilter': False,
                'sobelOperator': False,
                'gradientMatrix': False,
                'laplacian': False
            }
        }

    def getPredictionStorage(self):
        return {
            'predictionData': self.getPredictionData(),
            'predictionMask': self.getEmptyPredictionMask(),
            'predictionSlices': [0, 1],
            'TRAINING_TYPE': 'DEFAULT_TRAINING',
            'parameters': {
                'c': 10,
                'gamma': 1,
                'parallelJobs': 1,
                'estimators': 1,
                'cacheSize': 500,
                'tolerance': 1,
                'maxIterations': 5000}
        }

    def runTest(self):
        self.setUp()
        self.test_prepareFeatureVectorWhenAllFeaturesAllEnabled()
        self.test_runClassifierWithValidationSet()
        self.test_runClassifierWithoutValidationSet()
        self.delayDisplay('ALL TESTS PASSED!')

    def test_prepareFeatureVectorWhenAllFeaturesAllEnabled(self):
        self.assertIsNotNone(slicer.modules.background_classifier)
        self.delayDisplay("STARTING TEST: prepareFeatureVectorWhenAllFeaturesAllEnabled")

        expectedLengthOfFeatureVector = self.getTrainingData().size
        # 56 is length of Feature Vector row when all features are enabled
        ExpectedLengthOfFeatureVectorRow = 56

        TESTING_PATH = os.path.join(os.path.abspath(os.getcwd()), 'testing_of_preparation_feature_storage.pickle')
        featureStorage = self.getFeatureStorage()
        with open(TESTING_PATH, 'wb') as f:
            pickle.dump(featureStorage, f, pickle.HIGHEST_PROTOCOL)
        cliNodeParameters = {
            'operationType': 'PREPARING_TRAINING_SET',
            'pathToFeatureStorage': TESTING_PATH
        }
        slicer.cli.runSync(slicer.modules.background_classifier, parameters=cliNodeParameters)
        featureStorage = pickle.load(open(TESTING_PATH, 'rb'))
        os.remove(TESTING_PATH)
        vectorLength = featureStorage['trainingFeatureVector'].shape[0]
        rowLength = featureStorage['trainingFeatureVector'].shape[1]
        self.assertEqual(expectedLengthOfFeatureVector, vectorLength)
        self.assertEqual(ExpectedLengthOfFeatureVectorRow, rowLength)
        self.delayDisplay('TEST PASSED: prepareFeatureVectorWhenAllFeaturesAllEnabled')

    def test_runClassifierWithValidationSet(self):
        # Runs just with voxel value feature enabled -> for speed
        self.assertIsNotNone(slicer.modules.background_classifier)
        self.delayDisplay("STARTING TEST: runClassifierWithValidationSet")

        predictionMask = self.getPredictionData()
        initial_numberOfSlices = predictionMask.shape[0]
        initial_rows = predictionMask.shape[1]
        initial_cols = predictionMask.shape[2]

        FEATURE_STORAGE_PATH = os.path.join(os.path.abspath(os.getcwd()), 'testing_classifying_feature_storage.pickle')
        PREDICTION_STORAGE_PATH = os.path.join(os.path.abspath(os.getcwd()),
                                               'testing_classifying_prediction_storage.pickle')

        featureStorage = self.getFeatureStorageWithVoxelValueEnabled()
        with open(FEATURE_STORAGE_PATH, 'wb') as f:
            pickle.dump(featureStorage, f, pickle.HIGHEST_PROTOCOL)

        predictionStorage = self.getPredictionStorage()
        predictionStorage['validationData'] = self.getValidationData()
        predictionStorage['validationMask'] = self.getValidationMask()
        predictionStorage['validationSlices'] = [0, 1]
        with open(PREDICTION_STORAGE_PATH, 'wb') as f:
            pickle.dump(predictionStorage, f, pickle.HIGHEST_PROTOCOL)

        cliNodeParameters = {
            'operationType': 'CLASSIFYING',
            'pathToFeatureStorage': FEATURE_STORAGE_PATH,
            'pathToPredictionStorage': PREDICTION_STORAGE_PATH
        }
        slicer.cli.runSync(slicer.modules.background_classifier, parameters=cliNodeParameters)
        predictionStorage = pickle.load(open(PREDICTION_STORAGE_PATH, 'rb'))
        os.remove(FEATURE_STORAGE_PATH)
        os.remove(PREDICTION_STORAGE_PATH)
        numberOfPredictedSlices = predictionStorage['predictionMask'].shape[0]
        rows = predictionStorage['predictionMask'].shape[1]
        cols = predictionStorage['predictionMask'].shape[2]
        # If there is pixel with value 1, we can assume that mask was predicted
        isMaskPredicted = np.max(predictionStorage['predictionMask'])
        self.assertEqual(initial_numberOfSlices, numberOfPredictedSlices)
        self.assertEqual(initial_rows, rows)
        self.assertEqual(initial_cols, cols)
        self.assertEquals(1, isMaskPredicted)
        self.assertTrue('DICE' in predictionStorage)
        self.delayDisplay('TEST PASSED: runClassifierWithValidationSet')

    def test_runClassifierWithoutValidationSet(self):
        self.assertIsNotNone(slicer.modules.background_classifier)
        self.delayDisplay("STARTING TEST: runClassifierWithoutValidationSet")
        predictionMask = self.getPredictionData()

        initial_numberOfSlices = predictionMask.shape[0]
        initial_rows = predictionMask.shape[1]
        initial_cols = predictionMask.shape[2]

        FEATURE_STORAGE_PATH = os.path.join(os.path.abspath(os.getcwd()), 'testing_classifying_feature_storage.pickle')
        PREDICTION_STORAGE_PATH = os.path.join(os.path.abspath(os.getcwd()),
                                               'testing_classifying_prediction_storage.pickle')

        featureStorage = self.getFeatureStorageWithVoxelValueEnabled()
        with open(FEATURE_STORAGE_PATH, 'wb') as f:
            pickle.dump(featureStorage, f, pickle.HIGHEST_PROTOCOL)

        predictionStorage = self.getPredictionStorage()
        with open(PREDICTION_STORAGE_PATH, 'wb') as f:
            pickle.dump(predictionStorage, f, pickle.HIGHEST_PROTOCOL)

        cliNodeParameters = {
            'operationType': 'CLASSIFYING',
            'pathToFeatureStorage': FEATURE_STORAGE_PATH,
            'pathToPredictionStorage': PREDICTION_STORAGE_PATH
        }
        slicer.cli.runSync(slicer.modules.background_classifier, parameters=cliNodeParameters)
        predictionStorage = pickle.load(open(PREDICTION_STORAGE_PATH, 'rb'))
        os.remove(FEATURE_STORAGE_PATH)
        os.remove(PREDICTION_STORAGE_PATH)
        numberOfPredictedSlices = predictionStorage['predictionMask'].shape[0]
        rows = predictionStorage['predictionMask'].shape[1]
        cols = predictionStorage['predictionMask'].shape[2]
        isMaskProperlyPredicted = np.max(predictionStorage['predictionMask'])

        self.assertEqual(initial_numberOfSlices, numberOfPredictedSlices)
        self.assertEqual(initial_rows, rows)
        self.assertEqual(initial_cols, cols)
        self.assertEquals(1, isMaskProperlyPredicted)
        self.assertFalse('DICE' in predictionStorage)
        self.delayDisplay('TEST PASSED: runClassifierWithoutValidationSet')
