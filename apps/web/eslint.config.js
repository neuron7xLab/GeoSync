const next = require('eslint-config-next')
const prettierFlat = require('eslint-config-prettier/flat')
const prettierPlugin = require('eslint-plugin-prettier')
const tseslint = require('@typescript-eslint/eslint-plugin')

module.exports = [
  ...next,
  prettierFlat,
  {
    files: ['**/*.ts', '**/*.tsx'],
    plugins: { prettier: prettierPlugin, '@typescript-eslint': tseslint },
    rules: {
      'prettier/prettier': 'error',
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
      ],
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/explicit-function-return-type': 'off',
      '@typescript-eslint/explicit-module-boundary-types': 'off',
      'react/react-in-jsx-scope': 'off',
      'no-console': ['warn', { allow: ['warn', 'error'] }],
    },
  },
  {
    files: ['**/*.js', '**/*.cjs', '**/*.mjs'],
    plugins: { prettier: prettierPlugin },
    rules: {
      'prettier/prettier': 'error',
    },
  },
  {
    files: ['scripts/**'],
    rules: {
      'no-console': 'off',
    },
  },
  {
    ignores: [
      '.next/**',
      'next-env.d.ts',
      'out/**',
      'build/**',
      'dist/**',
      'playwright-report/**',
      'test-results/**',
      'coverage/**',
    ],
  },
]
