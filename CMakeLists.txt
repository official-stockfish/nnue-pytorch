cmake_minimum_required(VERSION 3.0)

project(training_data_loader CXX)

if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O3 -march=native -DNDEBUG")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED 17)

include(CheckCXXCompilerFlag)

# Function to check if the CPU is AMD and not older than family 25
function(check_amd_family_25_or_newer)
    execute_process(
        COMMAND awk "/^vendor_id/{{print \$3; exit}}" /proc/cpuinfo
        OUTPUT_VARIABLE VENDOR_ID
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND awk "/^cpu family/{{print \$4; exit}}" /proc/cpuinfo
        OUTPUT_VARIABLE CPU_FAMILY
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(VENDOR_ID STREQUAL "AuthenticAMD" AND CPU_FAMILY GREATER_EQUAL "25")
        set(CPU_AMD_FAMILY_25_OR_NEWER TRUE PARENT_SCOPE)
    else()
        set(CPU_AMD_FAMILY_25_OR_NEWER FALSE PARENT_SCOPE)
    endif()
endfunction()

# Call the function to check if the CPU is AMD and not older than family 25
check_amd_family_25_or_newer()

# Check for BMI2 support
check_cxx_compiler_flag("-mbmi2" COMPILER_SUPPORTS_BMI2)
if(COMPILER_SUPPORTS_BMI2)
    if(CPU_AMD_FAMILY_25_OR_NEWER OR NOT VENDOR_ID STREQUAL "AuthenticAMD")
        message(STATUS "Adding BMI2 support")
        add_definitions(-DHAS_BMI2)
    else()
        message(STATUS "CPU is older than AMD family 25, not adding BMI2 support")
    endif()
else()
    message(FATAL_ERROR "Compiler does not support BMI2")
endif()

add_library(training_data_loader SHARED training_data_loader.cpp)

find_package(Threads REQUIRED)

target_link_libraries(training_data_loader Threads::Threads)

install(
  TARGETS training_data_loader
  RUNTIME DESTINATION .
  LIBRARY DESTINATION .)
