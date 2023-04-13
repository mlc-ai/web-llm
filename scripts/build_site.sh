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
cp web/llm_chat.js site/dist
cp web/llm_chat.css site/dist/

cp dist/tokenizer.model site/dist
cp dist/tvmjs_runtime.wasi.js site/dist
cp dist/tvmjs.bundle.js site/dist
cp -r dist/sentencepiece site/dist

cd site && jekyll b && cd ..
