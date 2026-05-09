cmake_minimum_required(VERSION 3.10)

# 1. Project name and language
project(aruco2 LANGUAGES CXX)

# 2. Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# 3. Handle Build Type Portability (Release mode)
# ---------------------------------------------------------------------------

# For Linux/Unix (Single-config): Default to Release if nothing is specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# 4. Find Dependencies
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED)
#include opencv dir
include_directories(${OpenCV_INCLUDE_DIRS})
SET(SRCARUCO2 aruco2.hpp aruco2.cpp aruco_dictionary.hpp aruco_dictionary.cpp aruco2_fractal.cpp)

# ---------------------------------------------------------------------------
# 5. Compiler warnings — pedantic, treat warnings as errors
# ---------------------------------------------------------------------------
# if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
#     add_compile_options(-Wall -Wextra -Wpedantic -Werror)
# elseif(MSVC)
#     add_compile_options(/W4 /WX)
# endif()

# 5. Add Executable
add_executable(tests tests.cpp ${SRCARUCO2})
target_link_libraries(tests PRIVATE ${OpenCV_LIBS})

add_executable(camera_calibration camera_calibration.cpp ${SRCARUCO2})
target_link_libraries(camera_calibration PRIVATE ${OpenCV_LIBS})



# ---------------------------------------------------------------------------
# 8. Status Printout
# ---------------------------------------------------------------------------
message(STATUS "---------------------------------------------------")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler ID:    ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "---------------------------------------------------")
