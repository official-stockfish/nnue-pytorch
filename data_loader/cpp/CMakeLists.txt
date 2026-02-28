cmake_minimum_required(VERSION 3.13)
project(training_data_loader CXX)

if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# Normalize build type for safe, case-insensitive comparisons
string(TOUPPER "${CMAKE_BUILD_TYPE}" UPPER_BUILD_TYPE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O3 -march=native -DNDEBUG")

# ---- PGO Configuration ----
set(PGO_PROFILE_DATA_DIR "${CMAKE_BINARY_DIR}/pgo_data" CACHE PATH "Directory for PGO profile data")
set(PGO_INPUT "" CACHE STRING "Input file passed to training_data_loader_bench for profile generation")

if(UPPER_BUILD_TYPE STREQUAL "PGO_GENERATE" AND NOT PGO_INPUT)
  message(FATAL_ERROR "PGO_GENERATE build requires a non-empty PGO_INPUT.\nRe-run CMake with -DPGO_INPUT=<path-to-input-file>.")
endif()

# ---- BMI2 Detection ----
include(CheckCXXCompilerFlag)
execute_process(COMMAND bash -c "awk '/^vendor_id/{{print \$3; exit}}' /proc/cpuinfo" OUTPUT_VARIABLE VENDOR_ID OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND bash -c "awk '/^cpu family/{{print \$4; exit}}' /proc/cpuinfo" OUTPUT_VARIABLE CPU_FAMILY OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND bash -c "grep -m1 -o 'bmi2' /proc/cpuinfo" OUTPUT_VARIABLE CPU_BMI2 OUTPUT_STRIP_TRAILING_WHITESPACE)

if(VENDOR_ID STREQUAL "AuthenticAMD" AND CPU_FAMILY GREATER_EQUAL "23" AND CPU_BMI2 STREQUAL "bmi2")
    set(CPU_SUPPORTS_BMI2 TRUE)
elseif(CPU_BMI2 STREQUAL "bmi2")
    set(CPU_SUPPORTS_BMI2 TRUE)
else()
    set(CPU_SUPPORTS_BMI2 FALSE)
endif()

if(CPU_SUPPORTS_BMI2)
    message(STATUS "Adding BMI2 support")
    add_definitions(-DHAS_BMI2)
else()
    message(STATUS "No BMI2 support")
endif()

find_package(Threads REQUIRED)

# ---- Targets ----

if(UPPER_BUILD_TYPE MATCHES "^PGO_")
    add_library(training_data_loader STATIC training_data_loader.cpp)
    set_property(TARGET training_data_loader PROPERTY POSITION_INDEPENDENT_CODE ON)
else()
    add_library(training_data_loader SHARED training_data_loader.cpp)
    # Force the linker to fail at compile-time if symbols are missing
    target_link_options(training_data_loader PRIVATE "-Wl,--no-undefined")
endif()

target_link_libraries(training_data_loader PRIVATE Threads::Threads)

set_target_properties(training_data_loader PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_executable(training_data_loader_bench training_data_loader_bench.cpp)
target_link_libraries(training_data_loader_bench PRIVATE training_data_loader Threads::Threads)

set_target_properties(training_data_loader_bench PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)

# ---- Apply Target-Specific Build Flags ----

if(UPPER_BUILD_TYPE STREQUAL "PGO_GENERATE")
  file(MAKE_DIRECTORY ${PGO_PROFILE_DATA_DIR})

  target_compile_options(training_data_loader PRIVATE -O3 -march=native -DPGO_BUILD -fprofile-generate=${PGO_PROFILE_DATA_DIR})
  target_link_options(training_data_loader PRIVATE -fprofile-generate=${PGO_PROFILE_DATA_DIR})

  target_compile_options(training_data_loader_bench PRIVATE -O3 -march=native -DPGO_BUILD -fprofile-generate=${PGO_PROFILE_DATA_DIR})
  target_link_options(training_data_loader_bench PRIVATE -fprofile-generate=${PGO_PROFILE_DATA_DIR})

  add_custom_target(pgo_run
    COMMAND $<TARGET_FILE:training_data_loader_bench> ${PGO_INPUT}
    DEPENDS training_data_loader_bench
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running training_data_loader_bench to generate PGO profiles in ${PGO_PROFILE_DATA_DIR}"
    VERBATIM
  )

elseif(UPPER_BUILD_TYPE STREQUAL "PGO_USE")
  target_compile_options(training_data_loader PRIVATE -g -O3 -march=native -fprofile-use=${PGO_PROFILE_DATA_DIR} -fprofile-correction)
  target_link_options(training_data_loader PRIVATE -fprofile-use=${PGO_PROFILE_DATA_DIR} -fprofile-correction)

  target_compile_options(training_data_loader_bench PRIVATE -g -O3 -march=native -fprofile-use=${PGO_PROFILE_DATA_DIR} -fprofile-correction)
  target_link_options(training_data_loader_bench PRIVATE -fprofile-use=${PGO_PROFILE_DATA_DIR} -fprofile-correction)
endif()