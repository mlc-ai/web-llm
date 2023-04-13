#!/bin/bash
set -euxo pipefail

if [[ ! -f $1 ]]; then
    echo "cannot find config" $1
fi

rm -rf site/dist
mkdir -p site/dist site/_inlcudes

echo "Copy local configurations.."
cp $1 site/llm-chat-config.json
echo "Copy files..."
cp web/llm_chat.html site/_includes
cp web/llm_chat.js site/dist/
cp web/llm_chat.css site/dist/

cp dist/tvmjs_runtime.wasi.js site/dist
cp dist/tvmjs.bundle.js site/dist
cp -r dist/sentencepiece site/dist

if [ -d "dist/vicuna-7b/params" ]; then
    mkdir -p site/dist/vicuna-7b
    cp -rf dist/models/vicuna-7b/tokenizer.model site/dist/vicuna-7b/
    cp -rf dist/vicuna-7b/vicuna-7b_webgpu.wasm site/dist/vicuna-7b/
fi

cd site && jekyll b && cd ..
