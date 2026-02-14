import fs from 'fs';
import path from 'path';

const files = ['lib/index.js', 'lib/index.js.map'];

const replacements = [
    {
        pattern: /const\{createRequire:createRequire\}=await import\('module'\);/g,
        replacement: ''
    },
    {
        pattern: /require\("url"\)\.fileURLToPath\(new URL\("\.\/",import\.meta\.url\)\)/g,
        replacement: '"./"'
    },
    {
        pattern: /new \(require\('u' \+ 'rl'\)\.URL\)\('file:' \+ __filename\)\.href/g,
        replacement: '"MLC_DUMMY_PATH"'
    },
    {
        pattern: /import require\$\$3 from 'perf_hooks';/g,
        replacement: 'const require$$3 = "MLC_DUMMY_REQUIRE_VAR"'
    },
    {
        pattern: /require\("perf_hooks"\)/g,
        replacement: '"MLC_DUMMY_REQUIRE_VAR"'
    },
    {
        pattern: /import require\$\$4 from 'ws';/g,
        replacement: 'const require$$4 = "MLC_DUMMY_REQUIRE_VAR"'
    },
    {
        pattern: /require\("ws"\)/g,
        replacement: '"MLC_DUMMY_REQUIRE_VAR"'
    }
];

files.forEach(file => {
    const filePath = path.resolve(file);
    if (fs.existsSync(filePath)) {
        console.log(`Cleaning up ${file}...`);
        let content = fs.readFileSync(filePath, 'utf8');
        replacements.forEach(r => {
            content = content.replace(r.pattern, r.replacement);
        });
        fs.writeFileSync(filePath, content, 'utf8');
    } else {
        console.warn(`File ${file} not found.`);
    }
});
