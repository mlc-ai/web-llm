#!/bin/bash
set -euxo pipefail

# Generate model_hashes.json by fetching WASM files
echo "Generating model hashes..."
node scripts/generate_model_hashes.js

# Apply the generated hashes to src/config.ts
echo "Applying model hashes..."
node scripts/apply_model_hashes.js

echo "Model hashes updated successfully."
