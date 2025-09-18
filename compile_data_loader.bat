cmake -S . -B build-pgo-generate -DCMAKE_BUILD_TYPE=PGO_Generate
cmake --build ./build-pgo-generate --config PGO_Generate

./build-pgo-generate/training_data_loader_benchmark /home/shawn/Documents/Development/Python/nnue-pytorch/test77-dec2021-16tb7p.no-db.min.binpack

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=PGO_Use \
    -DPGO_PROFILE_DATA_DIR=build-pgo-generate/pgo_data \
    -DCMAKE_INSTALL_PREFIX="./"
cmake --build ./build --config PGO_Use --target install