#!/bin/bash
# This file prepares all the necessary dependencies for the web build.
set -euxo pipefail

emcc --version
npm --version

TVM_HOME_SET="${TVM_HOME:-}"

if [[ -z ${TVM_HOME_SET} ]]; then
    if [[ ! -d "3rdparty/tvm" ]]; then
        echo "Do not find TVM_HOME env variable, cloning a version as source".
        git clone https://github.com/apache/tvm 3rdparty/tvm --branch unity --recursive
    fi
    export TVM_HOME="${TVM_HOME:-3rdparty/tvm}"
fi

export TOKENIZERS_CPP_HOME="3rdparty/tokenizers-cpp/web"

mkdir -p dist
cd ${TVM_HOME}/web && make && npm install && npm run bundle && cd -
git submodule update --init --recursive
cd ${TOKENIZERS_CPP_HOME} && npm install && npm run build && cd -
git submodule update --init --recursive
rm -rf dist/tokenizers-cpp
cp -r ${TOKENIZERS_CPP_HOME}/dist dist/tokenizers-cpp

echo "Exporting tvmjs runtime dist files"
python -c "from tvm.contrib import tvmjs; tvmjs.export_runtime(\"dist\")"
