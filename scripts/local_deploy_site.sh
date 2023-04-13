#!/bin/bash
set -euxo pipefail

scripts/build_site.sh web/local-config.json

echo "symlink parameter location to site.."

# ln -s `pwd`/dist/params site/_site/web-sd-shards-v1-5
cd site && jekyll serve  --skip-initial-build --host localhost --baseurl /web-llm --port 8888
