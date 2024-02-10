# Collect source files
file(GLOB SOURCES "src/*.cpp")

# Include header files
include_directories(include)

# Specify that this is a static library
add_library(Validators STATIC ${SOURCES})

target_link_libraries(Validators Helpers)

target_include_directories(Validators PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
