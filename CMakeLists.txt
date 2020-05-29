cmake_minimum_required(VERSION 3.13.4)

project(SVM)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/SVM")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Jan Svoboda (BUT Brno)")
set(EXTENSION_DESCRIPTION "SVM Classifier")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/SVM.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/SVM/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SVMClassifier)
add_subdirectory(background_classifier)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
