:; # Linux/Bash section
:; rm -rf build
:; cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="./"
:; cmake --build ./build --config Release --target install
:; exit $?

@echo off
:: Windows/Batch section
if exist build rmdir /s /q build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="./"
cmake --build ./build --config Release --target install