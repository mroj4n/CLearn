cmake_minimum_required(VERSION 3.5)

# Linux enviorment
if (UNIX)
    set(CMAKE_C_COMPILER "/usr/bin/gcc")
    set(CMAKE_CXX_COMPILER "/usr/bin/g++")
endif (UNIX)

project(CLearn VERSION 1.0 LANGUAGES CXX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(BOOST_INCLUDE_LIBRARIES serialization filesystem)
set(BOOST_ENABLE_CMAKE ON)

include(FetchContent)
FetchContent_Declare(
  Boost
  GIT_REPOSITORY https://github.com/boostorg/boost.git
  GIT_TAG boost-1.80.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Boost)


# Add subdirectories
add_subdirectory(Metrics)
add_subdirectory(Models)
add_subdirectory(Utils)
add_subdirectory(Validators)
add_subdirectory(Helpers)
# Add executable
add_executable(${PROJECT_NAME} main.cpp)

# Link libraries with the executable
target_link_libraries(${PROJECT_NAME} Metrics Models Utils Helpers Validators Boost::filesystem Boost::serialization)
