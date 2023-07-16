#!/bin/bash
set -euxo pipefail

cd examples/simple-chat
rm -rf lib
npm run build
cd ../..

cp examples/simple-chat/lib/* site
cd site && jekyll b && cd ..

git fetch
git checkout -B gh-pages origin/gh-pages
rm -rf docs .gitignore
mkdir -p docs
cp -rf site/_site/* docs
touch docs/.nojekyll
echo "webllm.mlc.ai" >> docs/CNAME

DATE=`date`
git add docs && git commit -am "Build at ${DATE}"
git push origin gh-pages
git checkout main && git submodule update
echo "Finish deployment at ${DATE}"
