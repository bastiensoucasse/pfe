cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(CustomRegistration)

#-----------------------------------------------------------------------------
set(EXTENSION_HOMEPAGE "https://github.com/bastiensoucasse/pfe")
set(EXTENSION_CATEGORY "PFE")
set(EXTENSION_CONTRIBUTORS "Wissam Boussella (Université de Bordeaux), Iantsa Provost (Université de Bordeaux), Bastien Soucasse (Université de Bordeaux), Tony Wolff (Université de Bordeaux)")
set(EXTENSION_DESCRIPTION "The Custom Registration module for Slicer provides the features for 3D images registration, based on the ITK library.")
set(EXTENSION_ICONURL "")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "NA")

#-----------------------------------------------------------------------------
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
add_subdirectory(CustomRegistration)
add_subdirectory(registration)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
