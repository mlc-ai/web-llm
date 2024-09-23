# Remove instances of string "const{createRequire:createRequire}=await import('module');"
# This is required to allow background workers packaged with Parcel for the chrome extension
# to run the `ChatModule`.
sed -e s/"const{createRequire:createRequire}=await import('module');"//g -i .backup lib/index.js
sed -e s/"const{createRequire:createRequire}=await import('module');"//g -i .backup lib/index.js.map

# Replace string "new (require('u' + 'rl').URL)('file:' + __filename).href" with "MLC_DUMMY_PATH"
# This is required for building nextJS projects -- its compile time would complain about `require()`
# See https://github.com/mlc-ai/web-llm/issues/383 and the fixing PR's description for more.
sed -e s/"new (require('u' + 'rl').URL)('file:' + __filename).href"/"\"MLC_DUMMY_PATH\""/g -i .backup lib/index.js
# Replace with \"MLC_DUMMY_PATH\"
sed -e s/"new (require('u' + 'rl').URL)('file:' + __filename).href"/'\\\"MLC_DUMMY_PATH\\\"'/g -i .backup lib/index.js.map

# Replace "import require$$3 from 'perf_hooks';" with a string "const require$$3 = "MLC_DUMMY_REQUIRE_VAR""
# This is to prevent `perf_hooks` not found error
# For more see https://github.com/mlc-ai/web-llm/issues/258 and https://github.com/mlc-ai/web-llm/issues/127
sed -e s/"import require\$\$3 from 'perf_hooks';"/"const require\$\$3 = \"MLC_DUMMY_REQUIRE_VAR\""/g -i .backup lib/index.js
# Similarly replace `const performanceNode = require(\"perf_hooks\")` with `const performanceNode = \"MLC_DUMMY_REQUIRE_VAR\"`
sed -e s/'require(\\\"perf_hooks\\\")'/'\\\"MLC_DUMMY_REQUIRE_VAR\\\"'/g -i .backup lib/index.js.map

# Below is added when we include dependency @mlc-ai/web-runtime, rather than using local tvm_home
# Replace "import require$$4 from 'ws'" with a string "const require$$3 = "MLC_DUMMY_REQUIRE_VAR""
# This is to prevent error `Cannot find module 'ws'`
sed -e s/"import require\$\$4 from 'ws';"/"const require\$\$4 = \"MLC_DUMMY_REQUIRE_VAR\""/g -i .backup lib/index.js
# Similarly replace `const WebSocket = require(\"ws\")` with `const WebSocket = \"MLC_DUMMY_REQUIRE_VAR\"`
sed -e s/'require(\\\"ws\\\")'/'\\\"MLC_DUMMY_REQUIRE_VAR\\\"'/g -i .backup lib/index.js.map

# Cleanup backup files
rm lib/index.js.backup
rm lib/index.js.map.backup
