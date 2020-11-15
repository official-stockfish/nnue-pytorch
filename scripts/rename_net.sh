#!/bin/bash

# Renames file to SF net format.
name=nn-$(sha256sum $1 | cut -c1-12).nnue
echo ${name}
mv $1 ${name}
