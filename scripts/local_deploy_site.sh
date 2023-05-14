#!/bin/bash
set -euxo pipefail

scripts/build_site.sh web/global_config.json

echo "symlink parameter location to site.."

if [ -d "mlc-llm/dist/vicuna-v1-7b-q4f32_0/params" ]; then
    rm -rf site/_site/dist/vicuna-v1-7b-q4f32_0-params
    ln -s `pwd`/mlc-llm/dist/vicuna-v1-7b-q4f32_0/params site/_site/dist/vicuna-v1-7b-q4f32_0/params
    ls site/_site/dist/vicuna-v1-7b-q4f32_0
fi
if [ -d "mlc-llm/dist/wizardlm-7b/params" ]; then
    rm -rf site/_site/dist/wizardlm-7b-params
    ln -s `pwd`/mlc-llm/dist/wizardlm-7b/params site/_site/dist/wizardlm-7b-params
fi



cd site && jekyll serve  --skip-initial-build --host localhost --baseurl /web-llm --port 8888
