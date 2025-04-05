cmake . -Bbuild -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX="./" -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build ./build --config RelWithDebInfo --target install
