cmake -S . -B build-pgo-generate -DCMAKE_BUILD_TYPE=PGO_Generate
cmake --build ./build-pgo-generate --config PGO_Generate

./build-pgo-generate/training_data_loader_benchmark .pgo/small.binpack

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=PGO_Use \
    -DPGO_PROFILE_DATA_DIR=build-pgo-generate/pgo_data \
    -DCMAKE_INSTALL_PREFIX="./"
cmake --build ./build --config PGO_Use --target install

rm -rf build-pgo-generate