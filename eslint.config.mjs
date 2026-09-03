import js from '@eslint/js'
import eslintConfigPrettier from 'eslint-config-prettier'
import * as importPlugin from 'eslint-plugin-import'
import unusedPlugin from 'eslint-plugin-unused-imports'
import globals from 'globals'
import tseslint from 'typescript-eslint'

export default tseslint.config(
    {
        files: ['./projects/**/*.{js,ts,mjs,mts,cjs,cts,jsx,tsx}'],
        ignores: [
            'scalable-application-delivery/**',
            '**/target/**',
            '**/dist/**',
        ],
    },
    js.configs.recommended,
    eslintConfigPrettier,
    ...tseslint.configs.recommended,
    {
        languageOptions: {
            globals: {
                ...globals.browser,
                ...globals.node,
                ...globals.es5,
            },
        },
    },
    {
        plugins: { import: importPlugin, 'unused-imports': unusedPlugin },
        rules: {
            'unused-imports/no-unused-imports': 'error',
            'unused-imports/no-unused-vars': [
                'warn',
                {
                    vars: 'all',
                    varsIgnorePattern: '^_',
                    args: 'all',
                    argsIgnorePattern: '^_',
                },
            ],
            'import/order': [
                'error',
                {
                    alphabetize: {
                        order: 'asc',
                    },
                    groups: [
                        ['builtin', 'external', 'internal'],
                        ['parent', 'sibling', 'index'],
                        ['object'],
                    ],
                    'newlines-between': 'always',
                },
            ],
        },
    },
    {
        rules: {
            '@typescript-eslint/no-unused-vars': 'off',
        },
    },
    {
        files: [
            './src/app.tsx',
            './src/application/**/*.{ts,tsx}',
            './src/components/**/*.{ts,tsx}',
            './src/hooks/**/*.{ts,tsx}',
            './src/lib/**/*.{ts,tsx}',
            './src/ui/**/*.{ts,tsx}',
            './src/platform/memory/**/*.{ts,tsx}',
            './src/entrypoints/web-preview.tsx',
        ],
        rules: {
            'no-restricted-imports': [
                'error',
                {
                    patterns: ['@tauri-apps/*'],
                },
            ],
        },
    },
)
