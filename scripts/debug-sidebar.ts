// Debug script - run with: npx vite-node scripts/debug-sidebar.ts

import { generateFullSidebar } from '../docs/.vitepress/utils/sidebar'
import path from 'path'

const docsDir = path.resolve(__dirname, '../docs')
const sidebar = generateFullSidebar(docsDir)

console.log('=== /knowledge/ sidebar ===')
console.log(JSON.stringify(sidebar['/knowledge/'], null, 2))
