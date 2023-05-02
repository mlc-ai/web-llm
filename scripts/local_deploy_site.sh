#!/bin/bash
set -euxo pipefail

scripts/build_site.sh web/local-config.json

echo "symlink parameter location to site.."

if [ -d "dist/vicuna-7b-v1/params" ]; then
    rm -rf site/_site/vicuna-7b-v1-params
    ln -s `pwd`/dist/vicuna-7b-v1/params site/_site/vicuna-7b-v1-params
fi
if [ -d "dist/wizardlm-7b/params" ]; then
    rm -rf site/_site/wizardlm-7b-params
    ln -s `pwd`/dist/wizardlm-7b/params site/_site/wizardlm-7b-params
fi

cd site && jekyll serve  --skip-initial-build --host localhost --baseurl /web-llm --port 8888
