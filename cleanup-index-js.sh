# Remove instances of string "const{createRequire:createRequire}=await import('module');"
# This is required to allow background workers packaged with Parcel for the chrome extension
# to run the `ChatModule`.
sed -e s/"const{createRequire:createRequire}=await import('module');"//g -i .backup lib/index.js
sed -e s/"const{createRequire:createRequire}=await import('module');"//g -i .backup lib/index.js.map

# Replace string "new (require('u' + 'rl').URL)('file:' + __filename).href" with "MLC_DUMMY_PATH"
# This is required for building nextJS projects -- its compile time would complain about `require()`
# See https://github.com/mlc-ai/web-llm/issues/383 and the fixing PR's description for more.
sed -e s/"new (require('u' + 'rl').URL)('file:' + __filename).href"/"\"MLC_DUMMY_PATH\""/g -i .backup lib/index.js
sed -e s/"new (require('u' + 'rl').URL)('file:' + __filename).href"/"\"MLC_DUMMY_PATH\""/g -i .backup lib/index.js.map

# Cleanup backup files
rm lib/index.js.backup
rm lib/index.js.map.backup