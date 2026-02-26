import subprocess
import shutil
import os

def run():
    # 1. Clean previous build
    if os.path.exists("build"):
        shutil.rmtree("build")

    # 2. Configure
    subprocess.run(["cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_INSTALL_PREFIX=./"], check=True)

    # 3. Build & Install
    subprocess.run(["cmake", "--build", "./build", "--config", "Release", "--target", "install"], check=True)

if __name__ == "__main__":
    run()