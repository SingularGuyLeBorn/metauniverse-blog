import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

// Configuration
const ROOT_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../docs/knowledge')
const TARGET_EXTENSIONS = ['.py', '.java', '.cpp', '.c', '.h', '.js', '.ts', '.go', '.rs', '.ipynb', '.pdf', '.ppt', '.pptx', '.doc', '.docx']
const IGNORE_DIRS = ['node_modules', '.git', 'dist', 'assets', 'images', 'img']

// ... (imports remain the same)

function scanAndGenerate(dir) {
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

function convertIpynbToMarkdown(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf-8')
    const json = JSON.parse(content)
    let md = ''

    if (json.cells) {
      for (const cell of json.cells) {
        if (cell.cell_type === 'markdown') {
          // Join lines and ensure proper spacing
          md += cell.source.map(line => line.endsWith('\n') ? line : line + '\n').join('') + '\n\n'
        } else if (cell.cell_type === 'code') {
          // Wrap code in python block
          const code = cell.source.map(line => line.endsWith('\n') ? line : line + '\n').join('')
          md += '```python\n' + code + '```\n\n'
        }
      }
    }
    return md
  } catch (e) {
    console.error(`Error parsing ipynb ${filePath}:`, e)
    return '> Error parsing notebook content. Please check the file format.'
  }
}

function createWrapperMarkdown(filePath, fileName, ext) {
  const wrapperPath = filePath + '.md' // e.g., script.py -> script.py.md
  
  // NOTE: Always overwrite to ensure updates

  const title = fileName
  let bodyContent = ''

  if (ext === '.ipynb') {
    // For Jupyter Notebooks, we parse and render them as actual Markdown
    bodyContent = convertIpynbToMarkdown(filePath)
  } else if (ext === '.pdf') {
    // For PDF, use embed/iframe
    // Use relative path to the pdf file
    bodyContent = `<iframe src="./${fileName}" width="100%" height="800px" style="border: none;"></iframe>`
  } else if (['.ppt', '.pptx', '.doc', '.docx'].includes(ext)) {
    // For Office files, use a download card style
    // Since we can't embed them locally without backend, provides a neat download/open option
    bodyContent = `
<div class="tip custom-block">
  <p class="custom-block-title">📑 附件文档</p>
  <p>此文件格式 (${ext}) 暂不支持在博客中直接预览。</p>
  <p><a href="./${fileName}" target="_blank" download>👉 点击下载 ${fileName}</a></p>
</div>
`
  } else {
    // For other code files, we use the snippet include syntax
    const lang = ext.substring(1)
    bodyContent = `::: code-group\n\n<<< ./${fileName}{${lang}}\n\n:::`
  }

  const content = `---
title: ${title}
---

# ${title}

${bodyContent}
`

  fs.writeFileSync(wrapperPath, content, 'utf-8')
  console.log(`[Generated] ${wrapperPath}`)
}

console.log(`Scanning directory: ${ROOT_DIR}`)
scanAndGenerate(ROOT_DIR)
console.log('Done.')
