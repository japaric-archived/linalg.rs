#!/bin/bash

set -e
set -x

pushd ..
mkdir rust
cd rust
wget $URL
tar xf *.tar.gz
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
export PATH=$PWD/bin:$PATH
rustc -V

cd ..
mkdir cargo
cd cargo
wget http://static.rust-lang.org/cargo-dist/cargo-nightly-x86_64-unknown-linux-gnu.tar.gz
tar xf *.tar.gz --strip-components 1
export LD_LIBRARY_PATH=$PWD/cargo/lib:$LD_LIBRARY_PATH
export PATH=$PWD/cargo/bin:$PATH
popd

set +e
set +x
