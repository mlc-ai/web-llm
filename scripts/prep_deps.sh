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

export SENTENCEPIECE_JS_HOME="3rdparty/sentencepiece-js"

mkdir -p dist
cd ${TVM_HOME}/web && make && npm install && npm run bundle && cd -
git submodule update --init --recursive
cd ${SENTENCEPIECE_JS_HOME} && npm install && npm run build && cd -
git submodule update --init --recursive
rm -rf dist/sentencepiece
cp -r ${SENTENCEPIECE_JS_HOME}/dist dist/sentencepiece

echo "Exporting tvmjs runtime dist files"
python -c "from tvm.contrib import tvmjs; tvmjs.export_runtime(\"dist\")"
