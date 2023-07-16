#!/bin/bash
set -euxo pipefail

cd examples/simple-chat
rm -rf lib
npm run build
cd ../..

cp examples/simple-chat/lib/* site

cd site && jekyll serve  --host localhost --port 8888
