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
include_directories(src)
include_directories(opencv2/objdetect)
SET(SRCARUCO2  opencv2/objdetect/aruco2.hpp src/aruco2.cpp src/aruco_dictionary.hpp src/aruco_dictionary.cpp src/aruco2_fractal.cpp)

# ---------------------------------------------------------------------------
# 5. Compiler warnings — pedantic, treat warnings as errors
# ---------------------------------------------------------------------------
# if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
#     add_compile_options(-Wall -Wextra -Wpedantic -Werror)
# elseif(MSVC)
#     add_compile_options(/W4 /WX)
# endif()

include_directories(.)
# 5. Add Executable
OPTION(BUILD_DEV OFF)
IF(BUILD_DEV)
add_executable(devtest1 dev/devtest1.cpp ${SRCARUCO2})
target_link_libraries(devtest1 PRIVATE ${OpenCV_LIBS})
add_executable(devtest2 dev/devtest2.cpp ${SRCARUCO2})
target_link_libraries(devtest2 PRIVATE ${OpenCV_LIBS})
ENDIF()
add_executable(aruco2_camera_calibration utils/aruco2_camera_calibration.cpp ${SRCARUCO2})
target_link_libraries(aruco2_camera_calibration PRIVATE ${OpenCV_LIBS})

add_executable(aruco2_print utils/aruco2_print.cpp ${SRCARUCO2})
target_link_libraries(aruco2_print PRIVATE ${OpenCV_LIBS})

add_executable(aruco2_detect_markers utils/aruco2_detect_markers.cpp ${SRCARUCO2})
target_link_libraries(aruco2_detect_markers PRIVATE ${OpenCV_LIBS})

add_executable(aruco2_detect_diamonds utils/aruco2_detect_diamonds.cpp ${SRCARUCO2})
target_link_libraries(aruco2_detect_diamonds PRIVATE ${OpenCV_LIBS})

add_executable(aruco2_detect_fractal utils/aruco2_detect_fractal.cpp ${SRCARUCO2})
target_link_libraries(aruco2_detect_fractal PRIVATE ${OpenCV_LIBS})

add_executable(aruco2_testdictspeedup aruco2_testdictspeedup.cpp ${SRCARUCO2})
target_link_libraries(aruco2_testdictspeedup PRIVATE ${OpenCV_LIBS})

# ---------------------------------------------------------------------------
# 8. Status Printout
# ---------------------------------------------------------------------------
message(STATUS "---------------------------------------------------")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler ID:    ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "---------------------------------------------------")
