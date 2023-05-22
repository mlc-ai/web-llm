#!/bin/bash
set -euxo pipefail

MLC_LLM_HOME_SET="${MLC_LLM_HOME:-}"

if [ -z ${MLC_LLM_HOME_SET} ]; then
    export MLC_LLM_HOME="${MLC_LLM_HOME:-mlc-llm}"
fi
scripts/build_site.sh web/local-config.json

echo "symlink parameter location to site.."

if [ -d "$MLC_LLM_HOME/dist/vicuna-v1-7b-q4f32_0/params" ]; then
    rm -rf site/_site/dist/vicuna-v1-7b-q4f32_0
    mkdir -p site/_site/dist/vicuna-v1-7b-q4f32_0
    ln -s  "$(cd $MLC_LLM_HOME/dist/vicuna-v1-7b-q4f32_0/params && pwd)" site/_site/dist/vicuna-v1-7b-q4f32_0/params
    cp -rf $MLC_LLM_HOME/dist/vicuna-v1-7b-q4f32_0/vicuna-v1-7b-q4f32_0-webgpu.wasm site/_site/dist/vicuna-v1-7b-q4f32_0/
fi
if [ -d "$MLC_LLM_HOME/dist/RedPajama-INCITE-Chat-3B-v1-q4f32_0/params" ]; then
    rm -rf site/_site/dist/RedPajama-INCITE-Chat-3B-v1-q4f32_0
    mkdir -p site/_site/dist/RedPajama-INCITE-Chat-3B-v1-q4f32_0
    ln -s "$(cd $MLC_LLM_HOME/dist/RedPajama-INCITE-Chat-3B-v1-q4f32_0/params && pwd)" site/_site/dist/RedPajama-INCITE-Chat-3B-v1-q4f32_0/params
    cp -rf $MLC_LLM_HOME/dist/RedPajama-INCITE-Chat-3B-v1-q4f32_0/RedPajama-INCITE-Chat-3B-v1-q4f32_0-webgpu.wasm site/_site/dist/RedPajama-INCITE-Chat-3B-v1-q4f32_0/
fi
if [ -d "$MLC_LLM_HOME/dist/wizardlm-7b/params" ]; then
    rm -rf site/_site/dist/wizardlm-7b
    mkdir -p site/_site/dist/wizardlm-7b
    ln -s "$(cd $MLC_LLM_HOME/dist/wizardlm-7b/params && pwd)" site/_site/dist/wizardlm-7b/params
    cp -rf $MLC_LLM_HOME/dist/wizardlm-7b/wizardlm-7b-webgpu.wasm site/_site/dist/wizardlm-7b/
fi

cd site && jekyll serve  --skip-initial-build --host localhost --baseurl /web-llm --port 8888
