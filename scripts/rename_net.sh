#!/bin/bash

name=nn-$(sha256sum quantized.nnue | cut -c1-12).nnue
echo ${name}
mv quantized.nnue ${name}
