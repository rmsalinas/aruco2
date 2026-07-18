# CMake script to generate C++ wrappers from OpenCL kernels
# Usage:
#   cmake -DCL_PATH=<input.cl> -DHPP_PATH=<output.hpp> -DCPP_PATH=<output.cpp> -P cl2cpp.cmake

if(NOT CL_PATH OR NOT HPP_PATH OR NOT CPP_PATH)
    message(FATAL_ERROR "Usage: cmake -DCL_PATH=<input.cl> -DHPP_PATH=<output.hpp> -DCPP_PATH=<output.cpp> -P cl2cpp.cmake")
endif()

if(NOT EXISTS "${CL_PATH}")
    message(FATAL_ERROR "Error: Input file ${CL_PATH} does not exist.")
endif()

# Read the OpenCL file content
file(READ "${CL_PATH}" cl_content)

# Split the OpenCL content into chunks of less than 16380 characters to prevent MSVC C2026 error.
# MSVC has a limit of 16380 characters per string literal before concatenation.
string(LENGTH "${cl_content}" cl_length)
set(chunk_size 4000)
set(cpp_string_literals "")
set(offset 0)

while(offset LESS cl_length)
    # Calculate length to extract
    math(EXPR remaining "${cl_length} - ${offset}")
    if(remaining LESS chunk_size)
        set(extract_len ${remaining})
    else()
        set(extract_len ${chunk_size})
    endif()
    
    # Extract chunk
    string(SUBSTRING "${cl_content}" ${offset} ${extract_len} chunk)
    
    # Escape the chunk for C++ string literal (order is important: escape backslash first!)
    string(REPLACE "\\" "\\\\" chunk "${chunk}")
    string(REPLACE "\"" "\\\"" chunk "${chunk}")
    string(REPLACE "\n" "\\n" chunk "${chunk}")
    string(REPLACE "\r" "" chunk "${chunk}")
    string(REPLACE "\t" "\\t" chunk "${chunk}")
    
    set(cpp_string_literals "${cpp_string_literals}\"${chunk}\"\n")
    
    math(EXPR offset "${offset} + ${chunk_size}")
endwhile()

get_filename_component(HPP_NAME "${HPP_PATH}" NAME)

# Generate header file content
set(hpp_content "#pragma once
#include <opencv2/core/ocl.hpp>

namespace cv {
namespace ocl {
namespace objdetect {
    extern const ::cv::ocl::ProgramSource arucodetect_oclsrc;
}
}
}
")

# Generate source file content
set(cpp_content "#include \"${HPP_NAME}\"

namespace cv {
namespace ocl {
namespace objdetect {
    const ::cv::ocl::ProgramSource arucodetect_oclsrc(
${cpp_string_literals}    );
}
}
}
")

# Write the files
file(WRITE "${HPP_PATH}" "${hpp_content}")
file(WRITE "${CPP_PATH}" "${cpp_content}")

message(STATUS "Successfully generated ${HPP_PATH} and ${CPP_PATH} using CMake")
