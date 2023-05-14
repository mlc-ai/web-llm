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

cp mlc-llm/dist/tvmjs_runtime.wasi.js site/dist
cp mlc-llm/dist/tvmjs.bundle.js site/dist
cp -r mlc-llm/dist/sentencepiece site/dist

if [ -d "mlc-llm/dist/vicuna-v1-7b-q4f32_0/params" ]; then
    mkdir -p site/dist/vicuna-v1-7b-q4f32_0
    cp -rf mlc-llm/dist/vicuna-v1-7b-q4f32_0/tokenizer.model site/dist/vicuna-v1-7b-q4f32_0/
    cp -rf mlc-llm/dist/vicuna-v1-7b-q4f32_0/vicuna-v1-7b-q4f32_0-webgpu.wasm site/dist/vicuna-v1-7b-q4f32_0/
fi
if [ -d "mlc-llm/dist/wizardlm-7b/params" ]; then
    mkdir -p site/dist/wizardlm-7b
    cp -rf mlc-llm/dist/wizardlm-7b/tokenizer.model site/dist/wizardlm-7b/
    cp -rf mlc-llm/dist/wizardlm-7b/wizardlm-7b-webgpu.wasm site/dist/wizardlm-7b/
fi

cd site && jekyll b && cd ..
