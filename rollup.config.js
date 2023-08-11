import { nodeResolve } from '@rollup/plugin-node-resolve';
import typescript from 'rollup-plugin-typescript2'
import { defineConfig } from 'rollup';
import babel from '@rollup/plugin-babel'
import replace from '@rollup/plugin-replace'
import terser from '@rollup/plugin-terser'

const extensions = ['.ts']
const noDeclarationFiles = { compilerOptions: { declaration: false } }

const external = []

export default defineConfig([
    // es
    {
        input: 'src/index.ts',
        output: [
            {
                file: 'dist/es/index.js',
                format: 'es',
                indent: false,
            }],
        external,
        plugins: [
            nodeResolve({
                extensions
            }),
            typescript({
                tsconfigOverride: noDeclarationFiles
            }),
            babel({
                extensions,
                babelHelpers: 'bundled',

            })
        ]
    },
    // ES for Browsers
  {
    input: 'src/index.ts',
    output: { file: 'dist/es/redux.mjs', format: 'es', indent: false },
    plugins: [
      nodeResolve({
        extensions
      }),
      replace({
        preventAssignment: true,
        'process.env.NODE_ENV': JSON.stringify('production')
      }),
      typescript({ tsconfigOverride: noDeclarationFiles }),
      babel({
        extensions,
        exclude: 'node_modules/**',
        plugins: [['./scripts/mangleErrors.cjs', { minify: true }]],
        skipPreflightCheck: true,
        babelHelpers: 'bundled'
      }),
      terser({
        compress: {
          pure_getters: true,
          unsafe: true,
          unsafe_comps: true
        }
      })
    ]
  }
])
