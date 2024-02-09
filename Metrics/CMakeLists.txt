# Collect source files
file(GLOB SOURCES "src/*.cpp")

# Include header files
include_directories(include)

# Specify that this is a static library
add_library(Metrics STATIC ${SOURCES})

target_include_directories(Metrics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include) 

