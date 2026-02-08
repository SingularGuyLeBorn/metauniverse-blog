import fs from 'node:fs'
import path from 'node:path'
import matter from 'gray-matter'

export interface SidebarItem {
  text: string
  link?: string
  collapsed?: boolean
  items?: SidebarItem[]
  rel?: string
  target?: string
}

/**
 * 递归扫描文件夹生成 sidebar 配置
 * 
 * 规则（重构后）：
 * 1. 文件夹内容文件优先级: FolderName.md (同级) > index.md (内部)
 * 2. 层级始终保留：子文件夹无论有无内容，都正确嵌套
 * 3. 无内容文件：文件夹仅可折叠/展开，不可点击
 */
export function generateSidebar(
  baseDir: string, 
  basePath: string = '',
  depth: number = 0
): SidebarItem[] {
  const items: SidebarItem[] = []
  
  // 读取目录内容
  let entries: fs.Dirent[]
  try {
    entries = fs.readdirSync(baseDir, { withFileTypes: true })
  } catch {
    return items
  }
  
  const excludeDirs = ['assets', 'images', 'img', 'public', '.vitepress', 'node_modules']
  
  // 1. 分离目录和文件
  const directories = entries
    .filter(e => e.isDirectory() && !e.name.startsWith('.') && !excludeDirs.includes(e.name))
  const files = entries
    .filter(e => e.isFile() && e.name.endsWith('.md') && !e.name.startsWith('.'))
  
  // 2. 构建 "同名.md" 映射表: folderName -> mdFileEntry
  // 例如: { "rl-math-principle": Entry("rl-math-principle.md") }
  const folderContentMap = new Map<string, fs.Dirent>()
  for (const file of files) {
    const baseName = file.name.replace(/\.md$/, '')
    // 检查是否存在同名文件夹
    if (directories.some(d => d.name === baseName)) {
      folderContentMap.set(baseName, file)
    }
  }
  
  // 3. 收集所有节点 (文件夹 + 独立文件)，排序
  // 独立文件 = 不是 index.md 且不是某个文件夹的同名.md
  const standaloneMdFiles = files.filter(f => {
    if (f.name === 'index.md') return false
    const baseName = f.name.replace(/\.md$/, '')
    return !folderContentMap.has(baseName) // 不是同名内容文件
  })
  
  const allNodes = [...directories, ...standaloneMdFiles]
    .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' }))
  
  // 4. 处理每个节点
  for (const node of allNodes) {
    const nodeName = node.name
    
    if (node.isDirectory()) {
      // === 处理文件夹 ===
      const dirPath = path.join(baseDir, nodeName)
      const linkPath = `${basePath}/${nodeName}/`
      
      let title = formatDirName(nodeName)
      let link: string | undefined = undefined
      let contentFilePath: string | undefined = undefined
      
      // 优先级1: 同名.md (如 A.md 对应 A/)
      const folderContentFile = folderContentMap.get(nodeName)
      if (folderContentFile) {
        contentFilePath = path.join(baseDir, folderContentFile.name)
        // 链接指向同名文件 (不带 .md 后缀)
        link = `${basePath}/${nodeName}`
      } else {
        // 优先级2: index.md
        const indexPath = path.join(dirPath, 'index.md')
        if (fs.existsSync(indexPath)) {
          contentFilePath = indexPath
          link = linkPath
        }
      }
      
      // 读取内容文件获取标题
      if (contentFilePath && fs.existsSync(contentFilePath)) {
        try {
          const content = fs.readFileSync(contentFilePath, 'utf-8')
          const { data: frontmatter } = matter(content)
          if (frontmatter.title) {
            title = frontmatter.title
          }
          // 补全序号
          const match = nodeName.match(/^(\d+(\.\d+)*\.?\s+)/)
          if (match && !title.startsWith(match[1].trim()) && !title.startsWith(match[1])) {
            title = `${match[1]}${title}`
          }
        } catch {
          // ignore
        }
      }
      
      // 递归处理子项
      const children = generateSidebar(dirPath, linkPath.slice(0, -1), depth + 1)
      
      // 始终添加文件夹到 sidebar (层级保留)
      const item: SidebarItem = { text: `📁 ${title}` }
      
      // VitePress: 有 items 时会自动显示折叠箭头
      // 对于没有内容文件的文件夹，不设置 link，只允许折叠/展开
      if (link) {
        item.link = link
      }
      // 无 link 的文件夹：点击只会切换折叠状态，不会导航
      
      if (children.length > 0) {
        item.items = children
        item.collapsed = true
      }
      
      items.push(item)
      
    } else {
      // === 处理独立 .md 文件 ===
      const filePath = path.join(baseDir, nodeName)
      const linkPath = `${basePath}/${nodeName.replace(/\.md$/, '')}`
      
      let title: string
      try {
        const content = fs.readFileSync(filePath, 'utf-8')
        const { data: frontmatter } = matter(content)
        const rawName = nodeName.replace(/\.md$/, '')
        title = frontmatter.title || formatDirName(rawName)
        
        // 补全序号
        const match = nodeName.match(/^(\d+(\.\d+)*\.?\s+)/)
        if (match && !title.startsWith(match[1].trim()) && !title.startsWith(match[1])) {
          title = `${match[1]}${title}`
        }
      } catch {
        title = formatDirName(nodeName.replace(/\.md$/, ''))
      }
      
      // 文件图标
      const isWrapper = nodeName.endsWith('.md') && nodeName.split('.').length > 2
      const targetName = isWrapper ? nodeName : nodeName.replace(/\.md$/, '')
      const icon = getFileIcon(targetName)
      if (icon) {
        title = `${icon} ${title}`
      }
      
      const item: SidebarItem = {
        text: title,
        link: linkPath
      }
      
      // PDF 和其他附件：新标签页打开
      const isPdfOrAttachment = /\.(pdf|zip|rar|7z|tar|gz)\.md$/i.test(nodeName)
      if (isPdfOrAttachment) {
        item.rel = 'noreferrer'
        item.target = '_blank'
      }
      
      items.push(item)
    }
  }
  
  return items
}

/**
 * Get icon based on file extension
 */
function getFileIcon(filename: string): string {
  const ext = path.extname(filename).toLowerCase()
  // Check for combined extensions like .pdf.md which implies a wrapper
  if (filename.endsWith('.pdf.md')) return '📄'
  
  switch (ext) {
    case '.pdf': return '📄'
    case '.ppt':
    case '.pptx': return '📊'
    case '.doc':
    case '.docx': return '📝'
    case '.xls':
    case '.xlsx': return '📉'
    case '.py': return '🐍'
    case '.ipynb': return '📓'
    case '.java': return '☕'
    case '.c':
    case '.cpp':
    case '.h': return '🇨'
    case '.js':
    case '.ts': return '📜'
    case '.go': return '🐹'
    case '.rs': return '🦀'
    case '.zip':
    case '.rar':
    case '.7z': return '📦'
    case '.md': return '📝' // Default markdown icon
    default: return '📝' 
  }
}

/**
 * Check if the file is an attachment wrapper that should open in a new tab
 */
function isAttachmentWrapper(filename: string): boolean {
   return filename.endsWith('.pdf.md') || 
          filename.endsWith('.doc.md') || 
          filename.endsWith('.docx.md') || 
          filename.endsWith('.ppt.md') || 
          filename.endsWith('.pptx.md')
}

/**
 * 格式化目录名为标题
 * 01-hello-world -> Hello World
 */
function formatDirName(name: string): string {
  return name
}

/**
 * 为所有栏目生成完整的 sidebar 配置
 */
export function generateFullSidebar(docsDir: string): Record<string, SidebarItem[]> {
  const sections = ['posts', 'papers', 'essays', 'thoughts', 'yearly']
  const sidebar: Record<string, SidebarItem[]> = {}
  
  // 1. 处理普通栏目
  for (const section of sections) {
    const sectionPath = path.join(docsDir, section)
    if (fs.existsSync(sectionPath)) {
      const items = generateSidebar(sectionPath, `/${section}`)
      
      // 添加栏目首页链接
      sidebar[`/${section}/`] = [
        {
          text: getSectionTitle(section),
          items: [
            { text: '栏目首页', link: `/${section}/` },
            ...items
          ]
        }
      ]
    }
  }

  // 2. 特殊处理 Knowledge 知识库 (Sidebar Isolation)
  // 知识库下的每个子目录 (如 rl-math-principle, llm-mastery) 拥有独立的 Sidebar
  const knowledgeDir = path.join(docsDir, 'knowledge')
  if (fs.existsSync(knowledgeDir)) {
    const kbEntries = fs.readdirSync(knowledgeDir, { withFileTypes: true })
    
    // 过滤出知识库子目录
    const knowledgeBases = kbEntries
      .filter(e => e.isDirectory() && !['assets', 'img', 'images'].includes(e.name))
      .map(e => e.name)

    for (const kbName of knowledgeBases) {
      const kbPath = path.join(knowledgeDir, kbName)
      const kbLinkPath = `/knowledge/${kbName}`
      
      const items = generateSidebar(kbPath, kbLinkPath)
      
      // 读取知识库标题 (优先级: kbName.md > index.md)
      let title = formatDirName(kbName)
      const kbContentMdPath = path.join(knowledgeDir, `${kbName}.md`)
      const indexPath = path.join(kbPath, 'index.md')
      try {
        if (fs.existsSync(kbContentMdPath)) {
          const content = fs.readFileSync(kbContentMdPath, 'utf-8')
          const { data } = matter(content)
          if (data.title) title = data.title
        } else if (fs.existsSync(indexPath)) {
          const content = fs.readFileSync(indexPath, 'utf-8')
          const { data } = matter(content)
          if (data.title) title = data.title
        }
      } catch {}

      sidebar[`${kbLinkPath}/`] = [
        {
          text: title,
          items: [
            { text: '返回知识库首页', link: '/knowledge/' },
            { text: '📚 本库概览', link: `${kbLinkPath}/` },
            ...items
          ]
        }
      ]
    }

    // 3. 知识库首页自身的 Sidebar
    // 只显示知识库列表，不展开具体内容
    const knowledgeRootItems: SidebarItem[] = knowledgeBases.map(kbName => {
      let title = formatDirName(kbName)
      // 获取标题 (优先级: kbName.md > index.md)
      const kbContentMdPath = path.join(knowledgeDir, `${kbName}.md`)
      const indexPath = path.join(knowledgeDir, kbName, 'index.md')
      try {
        if (fs.existsSync(kbContentMdPath)) {
          const content = fs.readFileSync(kbContentMdPath, 'utf-8')
          const { data } = matter(content)
          if (data.title) title = data.title
        } else if (fs.existsSync(indexPath)) {
          const content = fs.readFileSync(indexPath, 'utf-8')
          const { data } = matter(content)
          if (data.title) title = data.title
        }
      } catch {}
      
      return {
        text: title,
        link: `/knowledge/${kbName}/`
      }
    })

    sidebar['/knowledge/'] = [
      {
        text: '🧠 知识库体系',
        items: [
          { text: '知识库首页', link: '/knowledge/' },
          ...knowledgeRootItems
        ]
      }
    ]
  }
  
  return sidebar
}

function getSectionTitle(section: string): string {
  const titles: Record<string, string> = {
    posts: '技术文章',
    papers: '论文阅读',
    knowledge: '知识库',
    essays: '杂谈',
    thoughts: '随想',
    yearly: '年度总结'
  }
  return titles[section] || section
}
