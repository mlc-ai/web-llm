import fs from 'fs';
import path from 'path';

// Applies the model_hashes.json to src/config.ts
const CONFIG_PATH = path.resolve('src/config.ts');
const HASHES_PATH = path.resolve('model_hashes.json');

function main() {
    console.log('Loading hashes...');
    const hashes = JSON.parse(fs.readFileSync(HASHES_PATH, 'utf-8'));
    const hashMap = new Map();
    hashes.forEach(h => {
        if (!h.error) {
            hashMap.set(h.model_id, h.hash);
        }
    });

    console.log('Reading config...');
    let configContent = fs.readFileSync(CONFIG_PATH, 'utf-8');
    let updatedCount = 0;

    for (const [modelId, hash] of hashMap.entries()) {
        // Pass 1: Replace existing integrity fields
        // We use slightly relaxed regex to match variations in spacing and any existing value inside quotes
        const replaceRegex = new RegExp(`(model_id:\\s*"${modelId}",[\\s\\S]*?model_lib_integrity:\\s*")([^"]*)(")`, 'g');
        if (replaceRegex.test(configContent)) {
            configContent = configContent.replace(replaceRegex, `$1${hash}$3`);
            updatedCount++;
            continue;
        }

        // Pass 2: Insert new integrity field if not present
        // We capture the prefix (indentation + potential comments) to preserve it
        const insertRegex = new RegExp(`([^\\n]*)(model_id:\\s*"${modelId}",[\\s\\S]*?model_lib:[\\s\\S]+?,)`, 'g');
        const checkRegex = new RegExp(`model_id:\\s*"${modelId}",[\\s\\S]*?model_lib_integrity`);

        if (!checkRegex.test(configContent)) {
            const oldContent = configContent;
            configContent = configContent.replace(insertRegex, (match, prefix, block) => {
                return `${prefix}${block}\n${prefix}model_lib_integrity: "${hash}",`;
            });
            if (configContent !== oldContent) {
                updatedCount++;
            }
        }
    }

    console.log(`Updated ${updatedCount} model records.`);
    fs.writeFileSync(CONFIG_PATH, configContent);
    console.log('Successfully updated src/config.ts');
}

main();
