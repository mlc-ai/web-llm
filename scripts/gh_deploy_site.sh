#!/bin/bash
set -euxo pipefail

export PYTHONPATH=$PWD/python
cd docs && make html && cd ..
cd site && jekyll b && cd ..
rm -rf site/_site/docs
cp -r docs/_build/html site/_site/docs

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
