cmake ./cpp/ -B ./cpp/build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX="./bin/"
cmake --build ./cpp/build --config RelWithDebInfo --target install