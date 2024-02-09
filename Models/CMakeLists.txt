# Collect source files
file(GLOB SOURCES "src/*.cpp")

# Include header files
include_directories(include)

# Specify that this is a static library
add_library(Models STATIC ${SOURCES})

# Link with Boost if necessary
target_link_libraries(Models ${Boost_LIBRARIES})

target_include_directories(Models PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
