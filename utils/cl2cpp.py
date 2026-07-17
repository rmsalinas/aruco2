import sys
import os

def main():
    if len(sys.argv) < 4:
        print("Usage: python cl2cpp.py <input.cl> <output.hpp> <output.cpp>")
        sys.exit(1)

    cl_path = sys.argv[1]
    hpp_path = sys.argv[2]
    cpp_path = sys.argv[3]

    if not os.path.exists(cl_path):
        print(f"Error: Input file {cl_path} does not exist.")
        sys.exit(1)

    # Read the OpenCL file content
    with open(cl_path, "r", encoding="utf-8") as f:
        cl_content = f.read()

    # Generate the .hpp file
    hpp_content = """#pragma once
#include <opencv2/core/ocl.hpp>

namespace cv {
namespace ocl {
namespace objdetect {
    extern const ::cv::ocl::ProgramSource arucodetect_oclsrc;
}
}
}
"""

    # Generate the .cpp file using C++11 raw string literal
    cpp_content = f"""#include "{os.path.basename(hpp_path)}"

namespace cv {{
namespace ocl {{
namespace objdetect {{
    const ::cv::ocl::ProgramSource arucodetect_oclsrc(
R"arucodetect_cl(
{cl_content}
)arucodetect_cl"
    );
}}
}}
}}
"""

    # Ensure parent directories exist
    os.makedirs(os.path.dirname(hpp_path), exist_ok=True)
    os.makedirs(os.path.dirname(cpp_path), exist_ok=True)

    # Write the files
    with open(hpp_path, "w", encoding="utf-8") as f:
        f.write(hpp_content)

    with open(cpp_path, "w", encoding="utf-8") as f:
        f.write(cpp_content)

    print(f"Successfully generated {hpp_path} and {cpp_path}")

if __name__ == "__main__":
    main()
