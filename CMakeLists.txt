cmake_minimum_required(VERSION 3.10)

project(training_data_loader CXX)

if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O3 -march=native -DNDEBUG")

# PGO Flags
set(PGO_PROFILE_DATA_DIR "${CMAKE_BINARY_DIR}/pgo_data" CACHE PATH "Directory for PGO profile data")
set(CMAKE_CXX_FLAGS_PGO_GENERATE "${CMAKE_CXX_FLAGS_RELEASE} -DPGO_BUILD -fprofile-generate=${PGO_PROFILE_DATA_DIR}")
set(CMAKE_EXE_LINKER_FLAGS_PGO_GENERATE "-fprofile-generate=${PGO_PROFILE_DATA_DIR}")
set(CMAKE_CXX_FLAGS_PGO_USE "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fprofile-use=${PGO_PROFILE_DATA_DIR} -fprofile-correction")

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

find_package(Threads REQUIRED)

if(CMAKE_BUILD_TYPE STREQUAL "PGO_Generate")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_PGO_GENERATE}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS_PGO_GENERATE}")
    add_executable(training_data_loader_benchmark training_data_loader.cpp)
    target_compile_definitions(training_data_loader_benchmark PRIVATE BENCH)
    target_link_libraries(training_data_loader_benchmark PRIVATE Threads::Threads)
    file(MAKE_DIRECTORY ${PGO_PROFILE_DATA_DIR})
else()
    if(CMAKE_BUILD_TYPE STREQUAL "PGO_USE")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_PGO_USE}")
    endif()
    add_library(training_data_loader SHARED training_data_loader.cpp)
    target_link_libraries(training_data_loader Threads::Threads)
    install(
      TARGETS training_data_loader
      RUNTIME DESTINATION .
      LIBRARY DESTINATION .)
endif()
