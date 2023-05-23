#!/bin/bash
# This file prepares all the necessary dependencies for the web build.
set -euxo pipefail

npm --version

MLC_LLM_HOME_SET="${MLC_LLM_HOME:-}"

if [[ -z ${MLC_LLM_HOME_SET} ]]; then
    echo "Do not find MLC_LLM_HOME env variable, need to set this to work".
fi
cd ${MLC_LLM_HOME}/dist
echo "Serving ${MLC_LLM_HOME}/dist for local debugging purposes"
npx http-server -p 8000 --cors
cd -
