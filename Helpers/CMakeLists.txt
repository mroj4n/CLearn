# Collect source files
file(GLOB SOURCES "src/*.cpp")

# Include header files
include_directories(include)

# Specify that this is a static library
add_library(Helpers STATIC ${SOURCES})

target_link_libraries(Helpers Utils)

target_include_directories(Helpers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
