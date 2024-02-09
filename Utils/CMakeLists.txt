# Collect source files
file(GLOB SOURCES "src/*.cpp")

# Include header files
include_directories(include)

# Specify that this is a static library
add_library(Utils STATIC ${SOURCES})

# Link with Boost
target_link_libraries(Utils ${Boost_LIBRARIES})

target_include_directories(Utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)