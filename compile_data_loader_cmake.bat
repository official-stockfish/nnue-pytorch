cmake . -Bbuild -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX="./"
cmake --build ./build --config RelWithDebInfo --target install