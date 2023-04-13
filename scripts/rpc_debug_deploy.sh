#!/bin/bash
set -euxo pipefail

TVM_HOME_SET="${TVM_HOME:-}"


if [[ -z ${TVM_HOME_SET} ]]; then
    echo "Require TVM_HOME to be set"
    exit 255
fi

echo "Copy files..."
mkdir -p ${TVM_HOME}/web/dist/www/dist/
cp web/llm_chat.html ${TVM_HOME}/web/dist/www/rpc_plugin.html
cp web/llm_chat.js ${TVM_HOME}/web/dist/www/dist/
cp web/llm_chat.css ${TVM_HOME}/web/dist/www/dist/
cp web/local-config.json ${TVM_HOME}/web/dist/www/llm-chat-config.json

rm -rf ${TVM_HOME}/web/dist/www/dist/sentencepiece
cp -rf dist/sentencepiece ${TVM_HOME}/web/dist/www/dist/

if [ -d "dist/vicuna-7b/params" ]; then
    mkdir -p ${TVM_HOME}/web/dist/www/dist/vicuna-7b
    cp -rf dist/models/vicuna-7b/tokenizer.model ${TVM_HOME}/web/dist/www/dist/vicuna-7b/
    cp -rf dist/vicuna-7b/vicuna-7b_webgpu.wasm ${TVM_HOME}/web/dist/www/dist/vicuna-7b/
    rm -rf ${TVM_HOME}/web/.ndarray_cache/vicuna-7b-params
    ln -s `pwd`/dist/vicuna-7b/params ${TVM_HOME}/web/.ndarray_cache/vicuna-7b-params
fi
