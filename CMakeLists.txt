cmake_minimum_required (VERSION 2.8)
project(diptera_track)

include(ExternalProject)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-deprecated-declarations")
set(THREADS_PREFER_PTHREAD_FLAG ON)
#set(VTK_DIR "./.local/lib/python2.7/site-packages")
#set(VTK_DIR "/opt/VTK-build/lib/cmake/vtk-8.1/Modules/")
#set(VTK_DIR "/usr/local/include/vtk-8.1/")
#set(VTK_DIR "/usr/local/bin/")
#set(VTK_DIR "/opt/VTK-build/lib" CACHE PATH "VTK directory override" FORCE)
#set(VTK_DIR "/usr/bin")

# Adding customized cmake module for building boost
list(APPEND CMAKE_MODULE_PATH  "${CMAKE_SOURCE_DIR}/cmake/Modules/")
list(APPEND CMAKE_MODULE_PATH "/usr/include/python2.7/")
#list(APPEND CMAKE_MODULE_PATH "/usr/local/lib/")
#list(APPEND CMAKE_MODULE_PATH "/usr/local/include/vtk-8.1/")
#list(APPEND CMAKE_MODULE_PATH "/usr/local/lib/cmake/vtk-8.1/")
#list(APPEND CMAKE_MODULE_PATH "/opt/VTK-build/lib/cmake/vtk-8.1/Modules/")
#list(APPEND CMAKE_LIBRARY_PATH "/usr/include/vtk-8.1/")
#list(APPEND CMAKE_LIBRARY_PATH "/usr/local/lib/")

# Find default python libraries and interpreter
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
#find_package(VTK 8.1 REQUIRED COMPONENTS vtkCommonComputationalGeometry vtkFiltersGeometry NO_MODULE)
#find_package(VTK 8.1 REQUIRED COMPONENTS vtkCommonCore vtkCommonMath vtkCommonDataModel vtkCommonSystem vtkFiltersCore vtkCommonComputationalGeometry vtkFiltersGeometry vtkIOGeometry NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(Armadillo REQUIRED)
find_package(PNG REQUIRED)
find_package(CGAL REQUIRED)
find_package(VTK REQUIRED)
#include(vtkExternalModuleMacros)
include(${VTK_USE_FILE})
#include(vtkExternalModuleMacros)
#include(vtkWrapping)
#include(vtkTargetLinkLibrariesWithDynamicLookup)
#include(${CGAL_USE_FILE})
#include(vtkObjectFactory)
#include(vtkCommonCore)
#include(vtkCommonMath)
#include(vtkCommonDataModel)
#include(vtkCommonComputationalGeometry)
#include(vtkCommonSystem)
#include(vtkFiltersCore)
#include(vtkFiltersGeometry)
#include(vtkIOGeometry)
include(BuildBoost) # Custom module
include(${CGAL_USE_FILE})
#include(vtkExternalModuleMacros)
#include(${VTK_USE_FILE})

SET(VTK_LIBS vtkGraphics vtkIO vtkCommon vtkImaging vtkFiltering)

message(STATUS "Boost_INCLUDE_DIR = " ${Boost_INCLUDE_DIR})
message(STATUS "Python_INCLUDE_DIR = " ${PYTHON_INCLUDE_DIRS})
message(STATUS "VTK_INCLUDE_DIRS = " ${VTK_INCLUDE_DIRS})
message(STATUS "VTK_DIR = " ${VTK_DIR})

include_directories(${Boost_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS} ${VTK_DIR})
link_directories(${Boost_LIBRARY_DIR})

# Build and link the pylib module
add_library(FlightTracker_lib SHARED FlightTracker_lib.cpp flight_tracker_class.cpp frame_loader.cpp focal_grid.cpp model_class.cpp image_segmentation.cpp initial_state.cpp contour_optimization.cpp)
target_link_libraries(FlightTracker_lib Threads::Threads)
target_link_libraries(FlightTracker_lib ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${OpenCV_LIBS} ${ARMADILLO_LIBRARIES} ${PNG_LIBRARY} ${VTK_LIBRARIES} nlopt gmp mpfr)
#target_link_libraries(FlightTracker_lib "/usr/local/lib/libnlopt.so")
add_dependencies(FlightTracker_lib Boost)

# Tweaks the name of the library to match what Python expects
set_target_properties(FlightTracker_lib PROPERTIES SUFFIX .so)
set_target_properties(FlightTracker_lib PROPERTIES PREFIX "")
