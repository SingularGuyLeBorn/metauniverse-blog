import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

// Configuration
const ROOT_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../docs/knowledge')
const TARGET_EXTENSIONS = ['.py', '.java', '.cpp', '.c', '.h', '.js', '.ts', '.go', '.rs', '.ipynb']
const IGNORE_DIRS = ['node_modules', '.git', 'dist', 'assets', 'images', 'img']

function scanAndGenerate(dir: string) {
  const entries = fs.readdirSync(dir, { withFileTypes: true })

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name)

    if (entry.isDirectory()) {
      if (IGNORE_DIRS.includes(entry.name)) continue
      scanAndGenerate(fullPath)
    } else if (entry.isFile()) {
      const ext = path.extname(entry.name).toLowerCase()
      if (TARGET_EXTENSIONS.includes(ext)) {
        createWrapperMarkdown(fullPath, entry.name, ext)
      }
    }
  }
}

function createWrapperMarkdown(filePath: string, fileName: string, ext: string) {
  const wrapperPath = filePath + '.md' // e.g., script.py -> script.py.md
  
  // If wrapper already exists, skip (or update if needed, but for now skip to verify)
  if (fs.existsSync(wrapperPath)) {
    return
  }

  // Calculate relative path for VitePress snippet inclusion
  // VitePress snippets are relative to the markdown file, or project root with @
  // Let's use relative path for simplicity: ./script.py
  // But wait, the wrapper is in the same folder, so `./${fileName}` works perfectly.
  
  const title = fileName
  
  // Special handling for ipynb: just treat as json for now, or use a specific language
  const lang = ext === '.ipynb' ? 'json' : ext.substring(1)

  const content = `---
title: ${title}
---

# ${title}

::: code-group

<<< ./${fileName}

:::

`

  fs.writeFileSync(wrapperPath, content, 'utf-8')
  console.log(`[Generated] ${wrapperPath}`)
}

console.log(`Scanning directory: ${ROOT_DIR}`)
scanAndGenerate(ROOT_DIR)
console.log('Done.')
