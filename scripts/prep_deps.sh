#!/bin/bash
# This file prepares all the necessary dependencies for the web build.
set -euxo pipefail

emcc --version
npm --version

TVM_SOURCE_DIR_SET="${TVM_SOURCE_DIR:-}"

if [[ -z ${TVM_SOURCE_DIR_SET} ]]; then
    if [[ ! -d "3rdparty/tvm-unity" ]]; then
        echo "Do not find TVM_SOURCE_DIR env variable, cloning a version as source".
        git clone https://github.com/mlc-ai/relax 3rdparty/tvm-unity --recursive
    fi
    export TVM_SOURCE_DIR="${TVM_SOURCE_DIR:-3rdparty/tvm-unity}"
fi

cd ${TVM_SOURCE_DIR}/web && make && npm install && npm run build && cd -
rm -rf tvm_home
ln -s ${TVM_SOURCE_DIR} tvm_home
npm install
