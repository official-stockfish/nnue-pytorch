cmake . -Bbuild -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="./"
cmake --build ./build --config Release --target install