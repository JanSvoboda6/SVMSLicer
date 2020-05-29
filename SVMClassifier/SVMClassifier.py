import os
import sys
import unittest
import logging
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import time

# TODO: Uncomment this!

# Dependency modules check ==============================================
# logging.info('-' * 30)
# logging.info('SVM Module: There is check needed whether all depency modules are installed...')
# dependencyModules = ['scipy', 'scikit-learn', 'numpy', 'joblib', 'pickle']
# for module in dependencyModules:
#     try:
#         module_obj = __import__(module)
#         logging.info('Module {0} was succesfully imported.'.format(module))
#     except ImportError:
#         logging.info("Module {0} was not found.\n Attempting to install {0}...".format(module))
#         slicer.util.pip_install(module)
# logging.info('SVM Module: All dependency modules should be installed! To be sure, check logging messages above.')
# logging.info('-' * 30)
# # Import dependency modules used in this file
import numpy as np
import pickle


# TODO: Delete this comment, if matplotlib is not used
# Importing plt after matplotlib.use('Agg') is not accidental, otherwise there is an error thrown during loading script


# TODO: Make method for validation of imports
# TODO: Make user friendly errors
# TODO: Delete module for preparing features
# TODO: Naming!!!
# TODO: Check IJKtoRAS Orientation

class SVMClassifier(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "SVMClassifier"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["Jan Svoboda (BUT Brno)"]
        self.parent.helpText = """
        SVM Classifier
        """
        self.parent.helpText += self.getDefaultModuleDocumentationLink()
        self.parent.acknowledgementText = """
        BUT Brno
        """


class SVMClassifierWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        # TODO:What about these self variables, should they be here or in setup?
        # TODO: Inner variables should start with _
        self.logic = None
        # TODO: unify file dialogs
        self.FileDialog = None
        self.featureStorage = None
        # TODO: is it used?
        self.file = None
        # TODO: Delete
        self.data = None
        # TODO: Rename featureVector to StorageNode?
        self.featureStoragePath = None
        self.predictionStoragePath = None
        self.total_time = None
        self.messageBox = None
        # self.pathToPredictionTemporaryStorage = None
        self.checkedIdx = 0

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        # TODO: What is this function, should it be somewhere else?
        ScriptedLoadableModuleWidget.setup(self)

        # Setup variables =================================================================
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/SVMClassifier.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        self.logic = SVMClassifierLogic()
        self.FileDialog = qt.QFileDialog()

        self.ui.trainingVolume.setMRMLScene(slicer.mrmlScene)
        self.ui.trainingMask.setMRMLScene(slicer.mrmlScene)
        self.ui.predictionVolume.setMRMLScene(slicer.mrmlScene)
        self.ui.predictionMask.setMRMLScene(slicer.mrmlScene)
        self.ui.validationVolume.setMRMLScene(slicer.mrmlScene)
        self.ui.validationMask.setMRMLScene(slicer.mrmlScene)

        # TODO: refactor self.ui.trainingVolume to save to self.trainingVolume...
        # Training Data =================================================================
        self.ui.trainingVolume.connect('currentNodeChanged(vtkMRMLNode*)', self.updateStateOfPrepareFeaturesButton)
        self.ui.trainingVolume.connect('currentNodeChanged(vtkMRMLNode*)', self.orientVolumeNodeToSagittal)
        self.ui.trainingMask.connect('currentNodeChanged(vtkMRMLNode*)', self.updateStateOfPrepareFeaturesButton)
        self.ui.trainingMask.connect('currentNodeChanged(vtkMRMLNode*)', self.orientVolumeNodeToSagittal)
        self.ui.createFeatureStorageButton.connect('clicked(bool)', self.onFeatureStorageButtons)
        self.ui.addToFeatureStorageButton.connect('clicked(bool)', self.onFeatureStorageButtons)
        self.ui.loadFeatureStorageButton.connect('clicked(bool)', self.onFeatureStorageButtons)
        self.ui.featureStorageFileBrowserButton.connect('clicked(bool)', self.onfeatureStorageFileBrowserButton)
        self.ui.prepareFeaturesButton.connect('clicked(bool)', self.onPrepareFeatures)

        # Features ========================================================================
        self.ui.voxelValueCheckBox.connect('clicked(bool)', self.onFeaturesCheckBoxes)
        self.ui.meanCheckBox.connect('clicked(bool)', self.onFeaturesCheckBoxes)
        self.ui.varianceCheckBox.connect('clicked(bool)', self.onFeaturesCheckBoxes)
        self.ui.gaussianFilterCheckBox.connect('clicked(bool)', self.onFeaturesCheckBoxes)
        self.ui.medianFilterCheckBox.connect('clicked(bool)', self.onFeaturesCheckBoxes)
        self.ui.sobelOperatorCheckBox.connect('clicked(bool)', self.onFeaturesCheckBoxes)
        self.ui.gradientMatrixCheckBox.connect('clicked(bool)', self.onFeaturesCheckBoxes)
        self.ui.laplacianCheckBox.connect('clicked(bool)', self.onFeaturesCheckBoxes)

        # Training =======================================================================
        self.ui.validationVolume.connect('currentNodeChanged(vtkMRMLNode*)', self.orientVolumeNodeToSagittal)
        self.ui.validationMask.connect('currentNodeChanged(vtkMRMLNode*)', self.orientVolumeNodeToSagittal)
        self.ui.defaultTrainingButton.connect('clicked(bool)', self.onTrainingButtons)
        self.ui.gridSearchTrainingButton.connect('clicked(bool)', self.onTrainingButtons)
        self.ui.noSavingButton.connect('clicked(bool)', self.onSavingLoadingClassifierButtons)
        self.ui.saveClassifierButton.connect('clicked(bool)', self.onSavingLoadingClassifierButtons)
        self.ui.loadClassifierButton.connect('clicked(bool)', self.onSavingLoadingClassifierButtons)
        self.ui.classifierFileBrowseButton.connect('clicked(bool)', self.onClassifierFileBrowserButton)
        # Default values
        self.ui.cParameterText.setText('10')
        self.ui.gammaParameterText.setText('1')
        self.ui.parallelJobsText.setText('-1')
        self.ui.estimatorsText.setText('16')
        self.ui.cacheSizeText.setText('500')
        self.ui.toleranceText.setText('0.5')
        self.ui.maxIterationsText.setText('15000')

        # Predicting =====================================================================
        self.ui.predictionVolume.connect('currentNodeChanged(vtkMRMLNode*)', self.updateStateOfApplyButton)
        self.ui.predictionVolume.connect('currentNodeChanged(vtkMRMLNode*)', self.orientVolumeNodeToSagittal)
        self.ui.predictionMask.connect('currentNodeChanged(vtkMRMLNode*)', self.updateStateOfApplyButton)
        # TODO: Rename applyButton
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

        # Initial GUI update
        # self.updateGUIFromParameterNode()

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def updateStateOfApplyButton(self):
        # TODO: Ugly condition
        # TODO: More conditions?
        # TODO: Not working when feature vector selected after prediction volumes
        if self.ui.predictionVolume.currentNode() is not None and self.ui.predictionMask.currentNode() is not None and self.ui.featureStoragePathText.toPlainText() is not '':
            if self.ui.saveClassifierButton.checked or self.ui.loadClassifierButton.checked:
                if self.ui.classifierPathText.toPlainText() is not "":
                    self.ui.applyButton.enabled = True
                    return
                else:
                    self.ui.applyButton.enabled = False
                    return
            else:
                self.ui.applyButton.enabled = True
                return
        self.ui.applyButton.enabled = False

    def onFeatureStorageButtons(self):
        self.ui.featureStoragePathText.setText('')
        self.updateStateOfFeatureCheckBoxes()
        self.updateStateOfPrepareFeaturesButton()
        self.updateStateOfApplyButton()
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

    def showMessageBox(self, severity, message):
        # TODO: Use Slicer built in warning
        if severity == 'INFO':
            message == 'INFO: ' + message
        elif severity == 'WARNING':
            message = 'Warning: ' + message
        elif severity == 'ERROR':
            message = 'Error: ' + message
        self.messageBox = qt.QMessageBox()
        self.messageBox.setText(message)
        self.messageBox.setModal(False)
        self.messageBox.show()

    def onfeatureStorageFileBrowserButton(self):
        valid, filePath = self.getFeatureStorageFile()
        if not valid:
            self.ui.featureStoragePathText.setText('')
            self.showMessageBox(severity='WARNING', message='No file has been selected.')
        else:
            logging.info('path:' + filePath)
            self.ui.featureStoragePathText.setText(filePath)
        self.updateStateOfFeatureCheckBoxes()
        self.updateStateOfPrepareFeaturesButton()
        self.updateStateOfApplyButton()

    def getFeatureStorageFile(self):
        if self.ui.createFeatureStorageButton.checked:
            filePath = self.FileDialog.getSaveFileName(None, 'Create New File', 'feature_storage', '*.pickle')
            if not filePath:
                return False, None
            # TODO: Extract this
            return True, filePath

        elif self.ui.addToFeatureStorageButton.checked:
            valid, filePath = self.getStorageNodeFileName()
            if not valid:
                return False, None
            # TODO: Is it necessary? Doesn't change occur somewhere else?
            return True, filePath

        elif self.ui.loadFeatureStorageButton.checked:
            valid, filePath = self.getStorageNodeFileName()
            if not valid:
                return False, None
            # TODO: Is it necessary? Doesn't change occur somewhere else?
            return True, filePath

    def updateStateOfFeatureCheckBoxes(self):
        if self.ui.featureStoragePathText.toPlainText() is not '':
            if self.ui.addToFeatureStorageButton.checked or self.ui.loadFeatureStorageButton.checked:
                self.checkActiveFeaturesCheckBoxes()
                self.enableCheckBoxes(False)
                return
        self.enableCheckBoxes(True)

    def checkActiveFeaturesCheckBoxes(self):
        path = self.ui.featureStoragePathText.toPlainText()
        if path is '':
            return
        features = self.loadfeatureStorage(path)['features']
        self.ui.voxelValueCheckBox.checked = features['voxelValue']
        self.ui.meanCheckBox.checked = features['mean']
        self.ui.varianceCheckBox.checked = features['variance']
        self.ui.gaussianFilterCheckBox.checked = features['gaussianFilter']
        self.ui.medianFilterCheckBox.checked = features['medianFilter']
        self.ui.sobelOperatorCheckBox.checked = features['sobelOperator']
        self.ui.gradientMatrixCheckBox.checked = features['gradientMatrix']
        self.ui.laplacianCheckBox.checked = features['laplacian']

    def enableCheckBoxes(self, isEnabled):
        for index in range(self.ui.featuresCheckBoxesLayout.count()):
            item = self.ui.featuresCheckBoxesLayout.itemAt(index)
            widget = item.widget()
            widget.enabled = isEnabled

    def updateStateOfPrepareFeaturesButton(self):
        # TODO: if load feature button enabled - prepare feature button is on
        if self.ui.featureStoragePathText.toPlainText() is not "":
            if self.ui.loadFeatureStorageButton.checked:
                self.ui.prepareFeaturesButton.enabled = False
                return
            elif self.ui.trainingVolume.currentNode() and self.ui.trainingMask.currentNode():
                self.ui.prepareFeaturesButton.enabled = True
                return
        self.ui.prepareFeaturesButton.enabled = False

    def getEmptyFeatureStorage(self):
        return {
            'trainingData': None,
            'trainingMask': None,
            'trainingFeatureVector': None,
            'trainingMaskVector': None,
            'features': {}
        }

    # TODO: this method is not used
    def getEmptyPredictionStorage(self):
        return {
            'TRAINING_TYPE': None,
            'predictionData': None,
            'parameters': {
                'c': None,
                'gamma': None,
                'crossValidations': None,
                'parallelJobs': None,
                'estimators': None,
                'cacheSize': None,
                'tolerance': None,
                'maxIterations': None,
                'classifierPath': None
            }
        }

    def loadfeatureStorage(self, pathToFeatureStorage):
        return pickle.load(open(pathToFeatureStorage, 'rb'))

    def saveFeatureStorage(self, pathToFeatureStorage):
        with open(pathToFeatureStorage, 'wb') as f:
            pickle.dump(self.featureStorage, f, pickle.HIGHEST_PROTOCOL)

    def loadPredictionStorage(self, pathToPredictionStorage):
        return pickle.load(open(pathToPredictionStorage, 'rb'))

    def savePredictionStorage(self, pathToPredictionStorage):
        with open(pathToPredictionStorage, 'wb') as f:
            pickle.dump(self.predictionStorage, f, pickle.HIGHEST_PROTOCOL)

    def validateBeforePreparation(self):
        # TODO: Complete validations
        if not self.ui.trainingVolume.currentNode() and not self.ui.trainingMask.currentNode():
            self.showMessageBox(severity='ERROR', message='Training Volume or Training Mask is not selected!')
            return False
        return True

    def onPrepareFeatures(self):
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
                self.featureStorage = self.loadfeatureStorage(self.ui.featureStoragePathText.toPlainText())

            slices = self.getParsedSlices(self.ui.trainingVolume.currentNode(),
                                          self.ui.trainingSliceNumbersText.toPlainText())
            self.fillFeatureStorageWithTrainingData(slices)
            logging.info('training vector: ' + str(self.featureStorage['trainingFeatureVector']))
            self.saveFeatureStorage(self.ui.featureStoragePathText.toPlainText())
            # TODO: implement different types - already done?
            self.featureStoragePath = self.ui.featureStoragePathText.toPlainText()
            cliNodeParameters = {
                'operationType': 'PREPARING_TRAINING_SET',
                'pathToFeatureStorage': self.featureStoragePath
            }
            # TODO: Rename cliNode?
            # TODO: Make XML for background_feature_preparation pretty
            cliNode = slicer.cli.run(slicer.modules.background_classifier, parameters=cliNodeParameters)
            self.ui.featurePreparationProgressBar.setCommandLineModuleNode(cliNode)
            logging.info(str(cliNode.GetErrorText()))
            logging.info(str(cliNode.GetOutputText()))
            cliNode.AddObserver('ModifiedEvent', self.checkFeaturePreparationStatus)
        except Exception as e:
            self.updateStateOfPrepareFeaturesButton()
            raise ValueError('Failed to prepare features!')

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
        This is an utility function that avoids blinking after orienting volume. Otherwise script that orients volumes
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

    def checkFeaturePreparationStatus(self, caller, event):
        # TODO: Refactor!
        # logging.info('Got a %s from a %s' % (event, caller.GetClassName()))
        if caller.IsA('vtkMRMLCommandLineModuleNode'):
            logging.info('Status is %s' % caller.GetStatusString())

            if caller.GetStatusString() == 'Completed':
                try:
                    # TODO: Maybe save classifier path before running, in case of change while running
                    self.featureStorage = self.loadfeatureStorage(self.featureStoragePath)
                    # TODO: clean logging
                    # TODO: get inner parameter of path, while running self.ui.textboxes can be changed
                    logging.info('Shape of result vector: ' + str(self.featureStorage['trainingFeatureVector'].shape))
                    logging.info('Shape of result mask vector: ' + str(self.featureStorage['trainingMaskVector'].shape))
                    # TODO: Maybe made this class without parameters, or just name it savePickleFile? to reuse it for temporary storage nodes
                    # At this point, we can delete training data and mask, we don't need them anymore
                    self.featureStorage['trainingData'] = None
                    self.featureStorage['trainingMask'] = None
                    self.saveFeatureStorage(self.featureStoragePath)
                    self.showMessageBox(severity='INFO', message=self.getAfterPreparationMessage())
                    self.updateStateOfPrepareFeaturesButton()
                except Exception as e:
                    caller.GetErrorText()
                    caller.GetOutputText()
                    self.updateStateOfPrepareFeaturesButton()
                    raise Exception()

            elif caller.GetStatusString() != 'Running':
                # No freeze in case of error
                logging.info(caller.GetErrorText())
                logging.info(caller.GetOutputText())
                self.updateStateOfPrepareFeaturesButton()

    def getAfterPreparationMessage(self):
        shapeOfFeatures = self.featureStorage['trainingFeatureVector'].shape
        message = '''\
        Features have been sucesfully prepared!
        Total time: ~ {} sec
        Size of whole feature vector: {} 
        Size of a feature row: {}
        For more information please look at full log message.\
        '''.format(int(time.time() - self.total_time), shapeOfFeatures[0], shapeOfFeatures[1])
        return message

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

    # TODO: These two methods should have same interface?
    def fillFeatureStorageWithTrainingData(self, slices):
        # TODO: Validate if shape of data and mask is same, rename these variables
        # TODO: Proper name of trainingMask to masks?
        self.featureStorage['trainingData'] = self.getTrainingData(slices)
        self.featureStorage['trainingMask'] = self.getTrainingMask(slices)

    def setupTemporaryStorageOfPredictionSet(self, slices):
        # TODO: Rename these variables
        temporaryStorageOfPredictingSet = {
            'predictionSlices': slices,
            'predictionData': self.getPredictionData(slices),
            'type': 'PREDICTION_PREPARE_FEATURES',
            'preparedFeatureVector': [],
            'predictionMask': self.getEmptyPredictionMask()
        }
        self.pathToPredictionTemporaryStorage = os.path.join(os.path.abspath(os.getcwd()),
                                                             'temporaryStorageOfPredictingSet.pickle')
        with open(self.pathToPredictionTemporaryStorage, 'wb') as f:
            pickle.dump(temporaryStorageOfPredictingSet, f, pickle.HIGHEST_PROTOCOL)

    def getStorageNodeFileName(self):
        filePath = self.FileDialog.getOpenFileName(None, 'Open File', '', '*.pickle')
        if not filePath:
            return False, None
        return True, filePath

    def getParsedSlices(self, volumeNode, slices):
        # TODO: Parse slices in format 20-100
        if not slices:
            self.showMessageBox(severity='ERROR', message='No Slice Numbers have been selected!')
            raise ValueError('No input slices has been selected!')
        try:
            if '-' in slices:
                borderSlices = [int(i) for i in slices.split('-')]
                if borderSlices[0] > borderSlices[1]:
                    self.showMessageBox(severity='ERROR',
                                        message='Starting slice number must be smaller than ending slice number!')
                    raise ValueError('Slice Numbers are not in correct format!')
                slices = [*range(borderSlices[0], borderSlices[1] + 1)]
            else:
                slices = slices.replace(' ', '').split(',')
                slices = [int(i) for i in slices]
        # TODO: What is the proper way of handling exceptions? Alternative to Value Error?
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
                                    message='Slice Number cannot be larger than total number of slices! Number has to be between 0 - ' + str(
                                        max_slice_number) + '.')
                raise ValueError(
                    'Slice number cannot be larger than total number of slices! Number has to be between 0 - ' + str(
                        max_slice_number) + '.')
        return slices

    def getTrainingData(self, slices):
        if self.ui.trainingVolume.currentNode() is None:
            raise ValueError('Training volume is invalid')
        voxels = slicer.util.arrayFromVolume(self.ui.trainingVolume.currentNode())
        logging.info('Voxels' + str(voxels.shape))
        selected_voxels = voxels[slices][:][:]
        logging.info('selected voxels' + str(selected_voxels.shape))
        return selected_voxels

    def getTrainingMask(self, slices):
        if self.ui.trainingMask.currentNode() is None:
            raise ValueError('Training mask is invalid')
        mask_labels = slicer.util.arrayFromVolume(self.ui.trainingMask.currentNode())
        selected_mask_labels = mask_labels[slices][:][:]
        logging.info('selected mask_labels' + str(selected_mask_labels.shape))
        return selected_mask_labels

    def onTrainingButtons(self):
        # TODO: Make better enabling - extract common
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

    def onSavingLoadingClassifierButtons(self):
        self.ui.classifierPathText.setText("")
        if self.ui.noSavingButton.checked:
            self.updateParametersState(True)
            self.ui.classifierPathText.enabled = False
            self.ui.classifierFileBrowseButton.enabled = False

        elif self.ui.saveClassifierButton.checked:
            self.updateParametersState(True)
            self.ui.classifierPathText.enabled = True
            self.ui.classifierFileBrowseButton.enabled = True

        elif self.ui.loadClassifierButton.checked:
            self.updateParametersState(False)
            self.ui.classifierPathText.enabled = True
            self.ui.classifierFileBrowseButton.enabled = True

        self.updateStateOfApplyButton()

    def updateParametersState(self, isEnabled):
        for index in range(self.ui.parametersLayout.count()):
            item = self.ui.parametersLayout.itemAt(index)
            widget = item.widget()
            widget.enabled = isEnabled
        self.ui.defaultTrainingButton.enabled = isEnabled
        self.ui.gridSearchTrainingButton.enabled = isEnabled
        self.onTrainingButtons()

    def getClassifierFile(self):
        if self.ui.saveClassifierButton.checked:
            filePath = self.FileDialog.getSaveFileName(None, 'Create New File', 'classifier', '*.joblib')
            if not filePath:
                return False, None
            return True, filePath

        elif self.ui.loadClassifierButton.checked:
            filePath = self.FileDialog.getOpenFileName(None, 'Load File', 'classifier', '*.joblib')
            if not filePath:
                return False, None
            # TODO: Is it necessary? Doesn't change occur somewhere else?
            return True, filePath

    def onClassifierFileBrowserButton(self):
        valid, path = self.getClassifierFile()
        if not valid:
            self.ui.classifierPathText.setText('')
            self.showMessageBox(severity='WARNING', message='No file has been selected.')
        else:
            self.ui.classifierPathText.setText(path)
        self.updateStateOfApplyButton()

    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        # TODO: Dissable applybutton while running
        # TODO: Move code to run method
        # TODO: Make pretty, extract to run method ?
        try:
            self.total_time = time.time()
            self.ui.applyButton.enabled = False
            self.fillPredictionStorageWithData()
            if not self.doValidationBeforeRunning():
                return
            self.featureStoragePath = self.ui.featureStoragePathText.toPlainText()
            # predictionStorage is used just for saving data temporarily to be able to send data to cliNode,
            # after predicting will be predictionStorage deleted. Current working directory will be used as path.
            self.predictionStoragePath = os.path.join(os.path.abspath(os.getcwd()), 'temporaryPredictionStorage.pickle')
            self.savePredictionStorage(self.predictionStoragePath)
            cliNodeParameters = {
                'operationType': 'CLASSIFYING',
                'pathToFeatureStorage': self.featureStoragePath,
                'pathToPredictionStorage': self.predictionStoragePath
            }
            # TODO: Rename cliNode?
            # TODO: Make XML for background_classifier pretty
            cliNode = slicer.cli.run(slicer.modules.background_classifier, parameters=cliNodeParameters)
            cliNode.AddObserver('ModifiedEvent', self.checkClassifierStatus)
            self.ui.classifyingProgressBar.setCommandLineModuleNode(cliNode)
            # TODO: what does except mean?
        except Exception as e:
            self.updateStateOfApplyButton()
            raise ValueError('Failed to compute results!')

    def doValidationBeforeRunning(self):
        # TODO: Make more validations?
        for param in self.predictionStorage.values():
            if isinstance(param, dict):
                for nestedParam in param.values():
                    logging.info('Nested:' + str(nestedParam))
                    if str(nestedParam) is '':
                        self.showMessageBox(severity='ERROR', message='One or more parameters is unfilled!')
                        return False

            else:
                if str(param) is '':
                    self.showMessageBox(severity='ERROR', message='One or more parameters is unfilled!')
                    return False
        return True

    def checkClassifierStatus(self, caller, event):
        # logging.info('Got a %s from a %s' % (event, caller.GetClassName()))
        if caller.IsA('vtkMRMLCommandLineModuleNode'):
            logging.info('Status is %s' % caller.GetStatusString())
            if caller.GetStatusString() == 'Completed':
                try:
                    # TODO: not get empty
                    # TODO: Do not create empty mask, just change slices
                    self.getEmptyPredictionMask()
                    self.predictionStorage = self.loadPredictionStorage(self.predictionStoragePath)
                    slicer.util.updateVolumeFromArray(self.ui.predictionMask.currentNode(),
                                                      self.predictionStorage['predictionMask'])
                    logging.info('Should be done!')
                    slicer.util.setSliceViewerLayers(self.ui.predictionMask.currentNode())
                    slicer.util.resetSliceViews()
                    self.showMessageBox(severity='INFO', message=self.getAfterPredictionMessage())
                    self.updateStateOfApplyButton()
                    os.remove(self.predictionStoragePath)
                    self.predictionStorage = None
                except Exception as e:
                    self.updateStateOfApplyButton()
                    raise Exception()

            elif caller.GetStatusString() != 'Running':
                # No freeze in case of error
                self.updateStateOfApplyButton()

    def getAfterPredictionMessage(self):
        message = 'Prediction Volume was successfully classified!\nTotal time: ~ {} sec'.format(
            int(time.time() - self.total_time))
        if 'DICE' in self.predictionStorage:
            message += '\nDICE: {:.3f}'.format(self.predictionStorage['DICE'])
        if self.predictionStorage['TRAINING_TYPE'] == 'GRID_SEARCH':
            message += '\nBEST PARAMETERS:  C: {}   Gamma: {}'.format(
                self.predictionStorage['bestParameters']['base_estimator__C'],
                self.predictionStorage['bestParameters']['base_estimator__gamma'])
        message += '\nFor more information please look at full log message.'
        return message

    def getFeatureStorage(self):
        return np.load(self.ui.featureStoragePathText.toPlainText())

    def getMasks(self):
        return np.load(self.ui.featureStoragePathText.toPlainText()[:-4] + '_masks.npy')

    def getPredictionSlices(self):
        if not self.ui.predictionSliceNumbersText.toPlainText():
            raise ValueError('No prediction slices has been selected!')
        slices = self.ui.predictionSliceNumbersText.toPlainText().replace(' ', '').split(',')
        slices = [int(i) for i in slices]
        logging.info(slices)
        return slices

    def getPredictionData(self, slices):
        if self.ui.predictionVolume.currentNode() is None:
            raise ValueError('Prediction volume is invalid')
        voxels = slicer.util.arrayFromVolume(self.ui.predictionVolume.currentNode())
        selected_voxels = voxels[slices][:][:]
        logging.info('selected prediction voxels' + str(selected_voxels.shape))
        return selected_voxels

    def getEmptyPredictionMask(self):
        # TODO: Just clean up slices that are for predicting, no need to make empty mask
        # TODO: if predicting volume shape == mask volume shape ---> that means mask volume has been already created
        if self.ui.predictionMask.currentNode() is None:
            raise ValueError('Prediction mask is invalid')
        self.ui.predictionMask.currentNode().CopyOrientation(self.ui.predictionVolume.currentNode())
        mask_voxels = np.zeros(slicer.util.arrayFromVolume(self.ui.predictionVolume.currentNode()).shape)
        slicer.util.updateVolumeFromArray(self.ui.predictionMask.currentNode(), mask_voxels)
        logging.info('prediction mask ' + str(slicer.util.arrayFromVolume(self.ui.predictionMask.currentNode()).shape))
        return slicer.util.arrayFromVolume(self.ui.predictionMask.currentNode())

    # TODO: can be general for other types of volumes
    def getValidationData(self, slices):
        if self.ui.validationVolume.currentNode() is None:
            raise ValueError('Validation volume is invalid')
        voxels = slicer.util.arrayFromVolume(self.ui.validationVolume.currentNode())
        selected_voxels = voxels[slices][:][:]
        logging.info('selected validation voxels' + str(selected_voxels.shape))
        return selected_voxels

    def getValidationMask(self, slices):
        if self.ui.validationMask.currentNode() is None:
            raise ValueError('Validation mask is invalid')
        mask_labels = slicer.util.arrayFromVolume(self.ui.validationMask.currentNode())
        selected_mask_labels = mask_labels[slices][:][:]
        logging.info('Validation mask_labels' + str(selected_mask_labels.shape))
        return selected_mask_labels

    def fillPredictionStorageWithData(self):
        try:
            predictionSlices = self.getParsedSlices(self.ui.predictionVolume.currentNode(),
                                                    self.ui.predictionSliceNumbersText.toPlainText())
            self.predictionStorage = {'predictionData': self.getPredictionData(predictionSlices),
                                      'predictionMask': self.getEmptyPredictionMask(),
                                      'predictionSlices': predictionSlices}
            if self.ui.validationVolume.currentNode() and self.ui.validationMask.currentNode():
                if self.ui.validationSliceNumbersText.toPlainText() is not '':
                    validationSlices = self.getParsedSlices(self.ui.validationVolume.currentNode(),
                                                            self.ui.validationSliceNumbersText.toPlainText())
                    self.predictionStorage['validationData'] = self.getValidationData(validationSlices)
                    self.predictionStorage['validationMask'] = self.getValidationMask(validationSlices)
                    self.predictionStorage['validationSlices'] = validationSlices

            if self.ui.loadClassifierButton.checked:
                self.predictionStorage['TRAINING_TYPE'] = 'FROM_SAVED_CLASSIFIER'
                # TODO: unify the behaviour of validation set and classifier path when they are not selected
                self.predictionStorage['parameters'] = {'classifierPath': self.ui.classifierPathText.toPlainText()}
                return

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
                logging.info(parameters['classifierPath'])

            if self.ui.defaultTrainingButton.checked:
                TRAINING_TYPE = 'DEFAULT_TRAINING'
                logging.info(type(TRAINING_TYPE))
                parameters['c'] = float(self.ui.cParameterText.toPlainText())
                parameters['gamma'] = float(self.ui.gammaParameterText.toPlainText())

            elif self.ui.gridSearchTrainingButton.checked:
                # TODO: Maybe extract common lines to method
                c = self.ui.cParametersText.toPlainText().replace(' ', '').split(',')
                c = [float(i) for i in c]
                gamma = self.ui.gammaParametersText.toPlainText().replace(' ', '').split(',')
                gamma = [float(i) for i in gamma]
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
            # TODO: REFACTOR
            self.predictionStorage['parameters'] = parameters

        except Exception as e:
            self.showMessageBox(severity='ERROR',
                                message='There was a problem with parsing parameters. '
                                        'Please ensure that all parameters are in correct format.')
            raise ValueError('There was a problem with parsing parameters. ')

        self.validateParameters(parameters)

    def validateParameters(self, parameters):
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
            raise ValueError('Parameters values must be positive!')

    # TODO: Unify the place of features, now it is "declared" in two or three methods
    # TODO: Is it possible?
    def getCheckedFeatures(self):
        return {'voxelValue': self.ui.voxelValueCheckBox.checked,
                'mean': self.ui.meanCheckBox.checked,
                'variance': self.ui.varianceCheckBox.checked,
                'gaussianFilter': self.ui.gaussianFilterCheckBox.checked,
                'medianFilter': self.ui.medianFilterCheckBox.checked,
                'sobelOperator': self.ui.sobelOperatorCheckBox.checked,
                'gradientMatrix': self.ui.gradientMatrixCheckBox.checked,
                'laplacian': self.ui.laplacianCheckBox.checked
                }


# SVMClassifierLogic
#

class SVMClassifierLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        # if not parameterNode.GetParameter("Threshold"):
        #     parameterNode.SetParameter("Threshold", "50.0")
        # if not parameterNode.GetParameter("Invert"):
        #     parameterNode.SetParameter("Invert", "false")

    def run(self, featureVector, trainingMasks, predictionData, predictionMask, predictionSlices, TRAINING_TYPE,
            parameters):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """
        prediction = self.classify(featureVector, trainingMasks, predictionData, predictionSlices, TRAINING_TYPE,
                                   parameters)
        prediction = prediction.reshape(-1, predictionMask.shape[1], predictionMask.shape[2])
        logging.info("Prediction after reshaping " + str(prediction.shape))
        idx = 0
        for i in predictionSlices:
            predictionMask[i] = prediction[idx]
            idx += 1
        return predictionMask


#
# SVMClassifierTest
#

class SVMClassifierTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    # TODO: Refactor test!
    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
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

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_prepareFeatureVectorWhenAllFeaturesAllEnabled()
        self.test_runClassifierWithValidationSet()
        self.test_runClassifierWithoutValidationSet()
        self.delayDisplay('ALL TESTS PASSED!')

    def test_prepareFeatureVectorWhenAllFeaturesAllEnabled(self):
        self.assertIsNotNone(slicer.modules.background_classifier)
        self.delayDisplay("STARTING TEST: prepareFeatureVectorWhenAllFeaturesAllEnabled")
        EXPECTED_LENGTH_OF_FEATURE_VECTOR = self.getTrainingData().size
        EXPECTED_LENGTH_OF_ROW = 56
        TESTING_PATH = os.path.join(os.path.abspath(os.getcwd()), 'testing_of_preparation_feature_storage.pickle')
        featureStorage = self.getFeatureStorage()
        with open(TESTING_PATH, 'wb') as f:
            pickle.dump(featureStorage, f, pickle.HIGHEST_PROTOCOL)
        cliNodeParameters = {
            'operationType': 'PREPARING_TRAINING_SET',
            'pathToFeatureStorage': TESTING_PATH
        }

        cliNode = slicer.cli.runSync(slicer.modules.background_classifier, parameters=cliNodeParameters)
        featureStorage = pickle.load(open(TESTING_PATH, 'rb'))
        os.remove(TESTING_PATH)
        vectorLength = featureStorage['trainingFeatureVector'].shape[0]
        rowLength = featureStorage['trainingFeatureVector'].shape[1]
        self.assertEqual(EXPECTED_LENGTH_OF_FEATURE_VECTOR, vectorLength)
        self.assertEqual(EXPECTED_LENGTH_OF_ROW, rowLength)
        self.delayDisplay('TEST PASSED: prepareFeatureVectorWhenAllFeaturesAllEnabled')

    def test_runClassifierWithValidationSet(self):
        # Just with voxel value feature enabled for speed
        self.assertIsNotNone(slicer.modules.background_classifier)
        self.delayDisplay("STARTING TEST: runClassifierWithValidationSet")
        predictionMask = self.getPredictionData()
        initial_numberOfSlices = predictionMask.shape[0]
        initial_rows = predictionMask.shape[1]
        initial_cols = predictionMask.shape[2]
        FEATURE_STORAGE_PATH = os.path.join(os.path.abspath(os.getcwd()), 'testing_classifying_feature_storage.pickle')
        PREDICTION_STORAGE_PATH = os.path.join(os.path.abspath(os.getcwd()),
                                               'testing_classifying_prediction_storage.pickle')
        featureStorage = {
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
        with open(FEATURE_STORAGE_PATH, 'wb') as f:
            pickle.dump(featureStorage, f, pickle.HIGHEST_PROTOCOL)

        predictionStorage = {'predictionData': self.getPredictionData(),
                             'predictionMask': self.getEmptyPredictionMask(),
                             'predictionSlices': [0, 1],
                             'validationData': self.getValidationData(),
                             'validationMask': self.getValidationMask(),
                             'validationSlices': [0, 1],
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

        with open(PREDICTION_STORAGE_PATH, 'wb') as f:
            pickle.dump(predictionStorage, f, pickle.HIGHEST_PROTOCOL)

        cliNodeParameters = {
            'operationType': 'CLASSIFYING',
            'pathToFeatureStorage': FEATURE_STORAGE_PATH,
            'pathToPredictionStorage': PREDICTION_STORAGE_PATH
        }
        cliNode = slicer.cli.runSync(slicer.modules.background_classifier, parameters=cliNodeParameters)
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
        # Just with voxel value feature enabled for speed
        self.assertIsNotNone(slicer.modules.background_classifier)
        self.delayDisplay("STARTING TEST: runClassifierWithoutValidationSet")
        predictionMask = self.getPredictionData()
        initial_numberOfSlices = predictionMask.shape[0]
        initial_rows = predictionMask.shape[1]
        initial_cols = predictionMask.shape[2]
        FEATURE_STORAGE_PATH = os.path.join(os.path.abspath(os.getcwd()), 'testing_classifying_feature_storage.pickle')
        PREDICTION_STORAGE_PATH = os.path.join(os.path.abspath(os.getcwd()),
                                               'testing_classifying_prediction_storage.pickle')
        featureStorage = {
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
        with open(FEATURE_STORAGE_PATH, 'wb') as f:
            pickle.dump(featureStorage, f, pickle.HIGHEST_PROTOCOL)

        predictionStorage = {'predictionData': self.getPredictionData(),
                             'predictionMask': self.getEmptyPredictionMask(),
                             'predictionSlices': [0, 1],
                             'TRAINING_TYPE': 'DEFAULT_TRAINING',
                             'parameters': {
                                 'c': 10,
                                 'gamma': 1,
                                 'parallelJobs': 1,
                                 'estimators': 4,
                                 'cacheSize': 500,
                                 'tolerance': 1,
                                 'maxIterations': 5000}
                             }

        with open(PREDICTION_STORAGE_PATH, 'wb') as f:
            pickle.dump(predictionStorage, f, pickle.HIGHEST_PROTOCOL)

        cliNodeParameters = {
            'operationType': 'CLASSIFYING',
            'pathToFeatureStorage': FEATURE_STORAGE_PATH,
            'pathToPredictionStorage': PREDICTION_STORAGE_PATH
        }
        cliNode = slicer.cli.runSync(slicer.modules.background_classifier, parameters=cliNodeParameters)
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
