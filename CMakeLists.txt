cmake_minimum_required(VERSION 3.10)

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

# Function to check if the CPU supports bmi2
function(check_bmi2_support)
    execute_process(
        COMMAND bash -c "awk '/^vendor_id/{{print \$3; exit}}' /proc/cpuinfo"
        OUTPUT_VARIABLE VENDOR_ID
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND bash -c "awk '/^cpu family/{{print \$4; exit}}' /proc/cpuinfo"
        OUTPUT_VARIABLE CPU_FAMILY
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND bash -c "grep -m1 -o 'bmi2' /proc/cpuinfo"
        OUTPUT_VARIABLE CPU_BMI2
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(VENDOR_ID STREQUAL "AuthenticAMD")
        if(CPU_FAMILY GREATER_EQUAL "23" AND CPU_BMI2 STREQUAL "bmi2")
            set(CPU_SUPPORTS_BMI2 TRUE PARENT_SCOPE)
        endif()
    elseif(CPU_BMI2 STREQUAL "bmi2")
        set(CPU_SUPPORTS_BMI2 TRUE PARENT_SCOPE)
    else()
        set(CPU_SUPPORTS_BMI2 FALSE PARENT_SCOPE)
    endif()
endfunction()

check_bmi2_support()

if(CPU_SUPPORTS_BMI2)
    message(STATUS "Adding BMI2 support")
    add_definitions(-DHAS_BMI2)
else()
    message(STATUS "No BMI2 support")
endif()

add_library(training_data_loader SHARED training_data_loader.cpp)

find_package(Threads REQUIRED)

target_link_libraries(training_data_loader Threads::Threads)

install(
  TARGETS training_data_loader
  RUNTIME DESTINATION .
  LIBRARY DESTINATION .)
