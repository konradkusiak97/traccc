# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Set up the project.
cmake_minimum_required( VERSION 3.9 )
project( traccc VERSION 0.1.0 LANGUAGES CXX )

# Standard CMake include(s).
include( GNUInstallDirs )

# Explicitly set the output directory for the binaries. Such that if this
# project is included by another project, the main project's configuration would
# win out.
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY
   "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}" CACHE PATH
   "Directory for the built binaries" )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY
   "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH
   "Directory for the built libraries" )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY
   "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH
   "Directory for the built static libraries" )

# Include the traccc CMake code.
list( APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" )
include( traccc-compiler-options )

# Check if CUDA is available.
include( CheckLanguage )
check_language( CUDA )
set( TRACCC_BUILD_CUDA_DEFAULT FALSE )
if( CMAKE_CUDA_COMPILER )
   set( TRACCC_BUILD_CUDA_DEFAULT TRUE )
endif()

# Flags controlling which parts of traccc to build.
option( TRACCC_BUILD_CUDA "Build the CUDA sources included in traccc"
   ${TRACCC_BUILD_CUDA_DEFAULT} )
option( TRACCC_BUILD_TESTING "Build the (unit) tests of traccc" TRUE )

# Clean up.
unset( TRACCC_BUILD_CUDA_DEFAULT )

# Set up VecMem.
option( TRACCC_SETUP_VECMEM
   "Set up the VecMem target(s) explicitly" TRUE )
option( TRACCC_USE_SYSTEM_VECMEM
   "Pick up an existing installation of VecMem from the build environment"
   FALSE )
if( TRACCC_SETUP_VECMEM )
   if( TRACCC_USE_SYSTEM_VECMEM )
      find_package( vecmem REQUIRED )
   else()
      add_subdirectory( extern/vecmem )
   endif()
endif()

<<<<<<< HEAD
# Set up Eigen3.
option( TRACCC_SETUP_EIGEN3
   "Set up the Eigen3 target(s) explicitly" TRUE )
option( TRACCC_USE_SYSTEM_EIGEN3
   "Pick up an existing installation of Eigen3 from the build environment"
   FALSE )
if( TRACCC_SETUP_EIGEN3 )
   if( TRACCC_USE_SYSTEM_EIGEN3 )
      find_package( Eigen3 REQUIRED )
   else()
      add_subdirectory( extern/eigen3 )
   endif()
endif()

if ( TRACCC_BUILD_SYCL )
  add_subdirectory( device/sycl )
=======
# Build the necessary externals.
include(traccc-dependencies)
add_subdirectory(extern)

# Set GPU-related options
if( TRACCC_BUILD_CUDA )
  add_subdirectory( device/cuda )
>>>>>>> 1676121e928ca5a8252e799f56d59168f117385c
endif()

if ( TRACCC_BUILD_SYCL )
  list( APPEND CMAKE_MODULE_PATH "${vecmem_SOURCE_DIR}/cmake/sycl" )
  enable_language(SYCL)
  add_subdirectory( device/sycl )
endif()

# Build the traccc code.
add_subdirectory( core )
if( TRACCC_BUILD_CUDA )
   add_subdirectory( device/cuda )
endif()
add_subdirectory( examples )
add_subdirectory( io )
add_subdirectory( plugins )

<<<<<<< HEAD
# Set up the test(s).
include( CTest )
if( BUILD_TESTING AND TRACCC_BUILD_TESTING )
   add_subdirectory( tests )
endif()
=======

>>>>>>> 1676121e928ca5a8252e799f56d59168f117385c
