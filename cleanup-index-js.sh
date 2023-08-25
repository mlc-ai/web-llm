# Remove instances of string "const{createRequire:createRequire}=await import('module');"
# This is required to allow background workers packaged with Parcel for the chrome extension
# to run the `ChatModule`.
sed -e s/"const{createRequire:createRequire}=await import('module');"//g -i .backup lib/index.js
sed -e s/"const{createRequire:createRequire}=await import('module');"//g -i .backup lib/index.js.map

# Cleanup backup files
rm lib/index.js.backup
rm lib/index.js.map.backup