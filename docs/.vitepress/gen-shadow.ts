
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import fg from 'fast-glob'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const docsRoot = path.resolve(__dirname, '..')

console.log('Scanning for source files to generate shadow markdown...')

const files = fg.sync(['**/*.py', '**/*.ipynb', '**/*.java', '**/*.cpp', '**/*.c', '**/*.cs', '**/*.go', '**/*.rs', '**/*.ts', '**/*.js'], { 
    cwd: docsRoot, 
    ignore: ['**/node_modules/**', '**/.vitepress/**', '**/public/**', '**/assets/**', '**/view/**'] 
})

let generatedCount = 0
let skippedCount = 0

files.forEach(file => {
    const fullPath = path.join(docsRoot, file)
    const shadowPath = fullPath + '.md'
    
    // Check if shadow file already exists
    if (fs.existsSync(shadowPath)) {
        skippedCount++
        return
    }

    const ext = path.extname(file).toLowerCase()
    const fileName = path.basename(file)
    let content = ''

    try {
        if (ext === '.ipynb') {
            const json = JSON.parse(fs.readFileSync(fullPath, 'utf-8'))
            const cells = Array.isArray(json.cells) ? json.cells : []
            content = cells.map((c: any) => {
                const source = Array.isArray(c.source) ? c.source.join('') : (c.source || '')
                if (c.cell_type === 'markdown') {
                    return source
                } else if (c.cell_type === 'code') {
                    return `\`\`\`python\n${source}\n\`\`\``
                }
                return ''
            }).join('\n\n')
        } else {
            // Source code files
            const langMap: Record<string, string> = {
                '.py': 'python',
                '.ts': 'typescript',
                '.js': 'javascript',
                '.java': 'java',
                '.c': 'c',
                '.cpp': 'cpp',
                '.h': 'cpp',
                '.cs': 'csharp',
                '.go': 'go',
                '.rs': 'rust'
            }
            const lang = langMap[ext] || ext.slice(1)
            // Use code-group to include the file
            content = `::: code-group\n\n<<< ./${fileName}{${lang}}\n\n:::`
        }

        const frontmatter = `---
title: ${fileName}
filePath: ${file}
aside: false
---

# ${fileName}

${content}
`
        fs.writeFileSync(shadowPath, frontmatter)
        console.log(`Generated: ${file}.md`)
        generatedCount++

    } catch (e) {
        console.error(`Error processing file ${file}:`, e)
    }
})

console.log(`Shadow file generation complete.`)
console.log(`Generated: ${generatedCount}`)
console.log(`Skipped (Already Exists): ${skippedCount}`)
