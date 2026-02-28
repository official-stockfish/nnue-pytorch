#!/usr/bin/env bash
set -euo pipefail

# Builds the training_data_loader with profile-guided optimization (PGO).
#
# Workflow:
# 1. Configure and build with PGO instrumentation (PGO_Generate).
# 2. Run the bench target to collect profile data.
# 3. Reconfigure and build the shared library with the collected profile data (PGO_Use).
#
# Usage:
#   ./compile_data_loader.sh [path/to/pgo_input]
# If no argument is provided the default is `.pgo/small.binpack` in the repo root.

ROOT_DIR=$(pwd)
BUILD_DIR=${BUILD_DIR:-build}
PGO_DIR=${PGO_DIR:-pgo_data}
PGO_INPUT=${1:-$ROOT_DIR/.pgo/small.binpack}

echo "ROOT_DIR: $ROOT_DIR"
echo "BUILD_DIR: $BUILD_DIR"
echo "PGO_DIR: $PGO_DIR"
echo "PGO_INPUT: $PGO_INPUT"

echo "Cleaning previous build and profile data..."
rm -rf "$BUILD_DIR" "$PGO_DIR"

echo "Configuring PGO_Generate build (instrumentation)..."
cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=PGO_Generate \
  -DPGO_PROFILE_DATA_DIR="$ROOT_DIR/$PGO_DIR" \
  -DPGO_INPUT="$PGO_INPUT"

echo "Building instrumented targets..."
cmake --build "$BUILD_DIR"

echo "Running bench to generate profile data (pgo_run)..."
cmake --build "$BUILD_DIR" --target pgo_run

echo "Re-configuring for PGO_Use (use collected profiles)..."
cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=PGO_Use \
  -DPGO_PROFILE_DATA_DIR="$ROOT_DIR/$PGO_DIR" \
  -DCMAKE_INSTALL_PREFIX="./"

echo "Building shared library with profile data (training_data_loader)..."
cmake --build "$BUILD_DIR" --target training_data_loader

echo "PGO build complete."

rm -rf "$PGO_DIR"