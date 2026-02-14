import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import https from 'https';

const CONFIG_PATH = path.resolve('src/config.ts');
const OUTPUT_MD = path.resolve('model_hashes.md');
const OUTPUT_JSON = path.resolve('model_hashes.json');

async function fetchFile(url) {
    console.log(`Fetching ${url}`);
    return new Promise((resolve, reject) => {
        https.get(url, (res) => {
            if (res.statusCode !== 200) {
                // If it's a redirect, follow it? GitHub raw usually doesn't redirect unless repo moved.
                if (res.statusCode === 301 || res.statusCode === 302) {
                    return resolve(fetchFile(res.headers.location));
                }
                reject(new Error(`Failed to fetch ${url}: ${res.statusCode}`));
                return;
            }
            const chunks = [];
            res.on('data', (chunk) => chunks.push(chunk));
            res.on('end', () => resolve(Buffer.concat(chunks)));
            res.on('error', reject);
        }).on('error', reject);
    });
}

function computeSHA384(buffer) {
    const hashBuffer = crypto.createHash('sha384').update(buffer).digest();
    const hashBase64 = hashBuffer.toString('base64');
    return `sha384-${hashBase64}`;
}

async function main() {
    const configContent = fs.readFileSync(CONFIG_PATH, 'utf-8');

    // Extract modelVersion
    const versionMatch = configContent.match(/export const modelVersion = "(.+?)";/);
    if (!versionMatch) {
        throw new Error('Could not find modelVersion in config.ts');
    }
    const modelVersion = versionMatch[1];

    // Extract modelLibURLPrefix
    const prefixMatch = configContent.match(/export const modelLibURLPrefix =\s+"(.+?)";/);
    if (!prefixMatch) {
        throw new Error('Could not find modelLibURLPrefix in config.ts');
    }
    const modelLibURLPrefix = prefixMatch[1];

    console.log(`Version: ${modelVersion}`);
    console.log(`Prefix: ${modelLibURLPrefix}`);

    // Find all model records
    // We match model_id and the wasm file path.
    // We assume the wasm file path is the last part of the model_lib string concatenation.
    // Matches: model_id: "...", ... model_lib: ... + "/..."

    // Note: formatting might vary. 
    // We'll search for 'model_id' and then look ahead for 'model_lib'.

    const models = [];
    const lines = configContent.split('\n');
    let currentModelId = null;

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const idMatch = line.match(/model_id:\s+"(.+?)"/);
        if (idMatch) {
            currentModelId = idMatch[1];
        }

        if (currentModelId) {
            // Look for wasm file in subsequent lines
            const wasmMatch = line.match(/"\/(.+?\.wasm)"/);
            if (wasmMatch) {
                const wasmPath = wasmMatch[1];
                models.push({
                    id: currentModelId,
                    path: wasmPath,
                    // Construct URL using the prefix and version from config
                    url: `${modelLibURLPrefix}${modelVersion}/${wasmPath}`
                });
                currentModelId = null; // Reset
            }
        }
    }

    console.log(`Found ${models.length} models. Starting hash generation...`);

    const results = [];
    for (const model of models) {
        try {
            console.log(`Processing ${model.id}...`);
            const buffer = await fetchFile(model.url);
            const hash = computeSHA384(buffer);
            results.push({
                model_id: model.id,
                wasm_file: model.path.split('/').pop(),
                url: model.url,
                hash: hash
            });
            console.log(`Done: ${hash}`);
        } catch (err) {
            console.error(`Error processing ${model.id}: ${err.message}`);
            results.push({
                model_id: model.id,
                error: err.message
            });
        }
    }

    // Generate Markdown Table
    let mdContent = '# Model WASM Hashes\n\n| Model ID | WASM File | SHA-384 Hash |\n| --- | --- | --- |\n';
    models.forEach(m => {
        const res = results.find(r => r.model_id === m.id);
        if (res.error) {
            mdContent += `| ${m.id} | ERROR | ${res.error} |\n`;
        } else {
            mdContent += `| ${m.id} | ${res.wasm_file} | \`${res.hash}\` |\n`;
        }
    });

    fs.writeFileSync(OUTPUT_MD, mdContent);
    fs.writeFileSync(OUTPUT_JSON, JSON.stringify(results, null, 2));
    console.log(`Written results to ${OUTPUT_MD} and ${OUTPUT_JSON}`);
}

main().catch(console.error);
