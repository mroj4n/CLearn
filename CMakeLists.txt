cmake_minimum_required(VERSION 3.5)
project(CLearn VERSION 1.0 LANGUAGES CXX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Boost library
find_package(Boost REQUIRED COMPONENTS serialization filesystem)
include_directories(${Boost_INCLUDE_DIRS})

# Add subdirectories
add_subdirectory(Metrics)
add_subdirectory(Models)
add_subdirectory(Utils)
add_subdirectory(Validators)
add_subdirectory(Helpers)
# Add executable
add_executable(${PROJECT_NAME} main.cpp)

# Link libraries with the executable
target_link_libraries(${PROJECT_NAME} Metrics Models Utils Helpers Validators ${Boost_LIBRARIES})
