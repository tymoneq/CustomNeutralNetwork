cmake_minimum_required(VERSION 3.10)
project(NeuralNetworkCpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)



# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Different configurations
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
endif()


# Try to find Eigen3
find_package(Eigen3 QUIET)

if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found via find_package, trying pkg-config...")
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(EIGEN3 eigen3)
    endif()
    
    if(NOT EIGEN3_FOUND)
        message(FATAL_ERROR "Eigen3 not found. Please install it or set EIGEN3_INCLUDE_DIR manually.")
    endif()
endif()

add_executable(myapp main.cpp)

# Link Eigen3
if(TARGET Eigen3::Eigen)
    target_link_libraries(myapp Eigen3::Eigen)
else()
    target_include_directories(myapp PRIVATE ${EIGEN3_INCLUDE_DIRS})
    target_compile_options(myapp PRIVATE ${EIGEN3_CFLAGS_OTHER})
endif()