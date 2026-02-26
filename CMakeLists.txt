cmake_minimum_required(VERSION 3.10)
project(training_data_loader CXX)

if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O3 -march=native -DNDEBUG")

# PGO
set(PGO_PROFILE_DATA_DIR "${CMAKE_BINARY_DIR}/pgo_data"
    CACHE PATH "Directory for PGO profile data")
set(PGO_INPUT "" CACHE STRING "Input file passed to training_data_loader_bench for profile generation")

set(CMAKE_CXX_FLAGS_PGO_GENERATE
    "${CMAKE_CXX_FLAGS_RELEASE} -DPGO_BUILD -fprofile-generate=${PGO_PROFILE_DATA_DIR}")
set(CMAKE_EXE_LINKER_FLAGS_PGO_GENERATE
    "-fprofile-generate=${PGO_PROFILE_DATA_DIR}")
set(CMAKE_CXX_FLAGS_PGO_USE
    "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fprofile-use=${PGO_PROFILE_DATA_DIR} -fprofile-correction")

include(CheckCXXCompilerFlag)

# BMI2 detection

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
        set(CPU_SUPPORTS_BMI2 TRUE)
    endif()
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

add_library(training_data_loader SHARED training_data_loader.cpp)
target_compile_options(training_data_loader PRIVATE)
target_link_libraries(training_data_loader PRIVATE Threads::Threads)

set_target_properties(training_data_loader PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_executable(training_data_loader_bench training_data_loader_bench.cpp)
target_compile_options(training_data_loader_bench PRIVATE)
target_link_libraries(training_data_loader_bench
  PRIVATE training_data_loader Threads::Threads)

set_target_properties(training_data_loader_bench PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)

# ---- PGO ----

if(CMAKE_BUILD_TYPE STREQUAL "PGO_Generate")
  file(MAKE_DIRECTORY ${PGO_PROFILE_DATA_DIR})

  target_compile_options(training_data_loader PRIVATE
    -fprofile-generate=${PGO_PROFILE_DATA_DIR})
  target_link_options(training_data_loader PRIVATE
    -fprofile-generate=${PGO_PROFILE_DATA_DIR})

  target_compile_options(training_data_loader_bench PRIVATE
    -DPGO_BUILD -fprofile-generate=${PGO_PROFILE_DATA_DIR})
  target_link_options(training_data_loader_bench PRIVATE
    -fprofile-generate=${PGO_PROFILE_DATA_DIR})

  add_custom_target(pgo_run
    COMMAND $<TARGET_FILE:training_data_loader_bench> ${PGO_INPUT}
    DEPENDS training_data_loader_bench
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running training_data_loader_bench to generate PGO profiles in ${PGO_PROFILE_DATA_DIR}"
    VERBATIM
  )

elseif(CMAKE_BUILD_TYPE STREQUAL "PGO_Use" OR CMAKE_BUILD_TYPE STREQUAL "PGO_USE")
  target_compile_options(training_data_loader PRIVATE
    -fprofile-use=${PGO_PROFILE_DATA_DIR} -fprofile-correction)
  target_link_options(training_data_loader PRIVATE
    -fprofile-use=${PGO_PROFILE_DATA_DIR} -fprofile-correction)

  target_compile_options(training_data_loader_bench PRIVATE
    -fprofile-use=${PGO_PROFILE_DATA_DIR} -fprofile-correction)
  target_link_options(training_data_loader_bench PRIVATE
    -fprofile-use=${PGO_PROFILE_DATA_DIR} -fprofile-correction)
endif()