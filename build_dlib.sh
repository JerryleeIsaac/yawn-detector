#/usr/bin/bash

git clone git@github.com:davisking/dlib.git dlib
cd dlib
git checkout v19.22
mkdir build; cd build; cmake .. -DUSE_AVX_INSTRUCTIONS=1 -DDLIB_USE_CUDA=1; cmake --build .