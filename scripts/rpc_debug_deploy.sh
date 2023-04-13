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

cp dist/tokenizer.model ${TVM_HOME}/web/dist/www/tokenizer.model
rm -rf ${TVM_HOME}/web/dist/www/dist/sentencepiece
cp -rf dist/sentencepiece ${TVM_HOME}/web/dist/www/dist/

# rm -rf ${TVM_HOME}/web/.ndarray_cache/chat-llm-shards
# ln -s `pwd`/dist/params ${TVM_HOME}/web/.ndarray_cache/web-sd-shards-v1-5
