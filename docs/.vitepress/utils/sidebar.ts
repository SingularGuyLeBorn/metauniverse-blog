import fs from 'node:fs'
import path from 'node:path'
import matter from 'gray-matter'

export interface SidebarItem {
  text: string
  link?: string
  collapsed?: boolean
  items?: SidebarItem[]
}

/**
 * 递归扫描文件夹生成 sidebar 配置
 * 规则：
 * - 文件夹下有 index.md 则识别为一个文档节点
 * - 文档标题优先从 frontmatter.title 读取，否则用文件夹名
 * - 子文件夹自动成为子节点
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
  
  // 过滤出子目录和 .md 文件
  const excludeDirs = ['assets', 'images', 'img', 'public', '.vitepress', 'node_modules']
  const nodes = entries
    .filter(e => {
      if (e.name.startsWith('.')) return false
      if (e.isDirectory()) return !excludeDirs.includes(e.name)
      if (e.isFile()) return e.name.endsWith('.md') && e.name !== 'index.md'
      return false
    })
    .sort((a, b) => {
      // 使用 localeCompare 的 numeric 模式进行自然排序 (1, 2, 10 而不是 1, 10, 2)
      return a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' })
    })
  
  for (const node of nodes) {
    const nodeName = node.name
    
    if (node.isDirectory()) {
      // 处理文件夹节点 (A/index.md)
      const dirPath = path.join(baseDir, nodeName)
      const indexPath = path.join(dirPath, 'index.md')
      const linkPath = `${basePath}/${nodeName}/`
      
      let title = formatDirName(nodeName)
      let link: string | undefined = undefined

      // 尝试读取 index.md 获取标题和链接
      if (fs.existsSync(indexPath)) {
        try {
          const content = fs.readFileSync(indexPath, 'utf-8')
          const { data: frontmatter } = matter(content)
          title = frontmatter.title || title
          link = linkPath
        } catch (e) {
          // ignore error
        }
      }

      const children = generateSidebar(dirPath, linkPath.slice(0, -1), depth + 1)

      // 只有当存在 index.md 或者有子节点时才添加到 sidebar
      if (link || children.length > 0) {
        const item: SidebarItem = {
          text: title
        }

        if (link) {
          item.link = link
        }

        if (children.length > 0) {
          item.items = children
          item.collapsed = true // VS Code 风格：默认折叠，自动定位
        }
        
        items.push(item)
      }
    } else {
      // 处理独立文件节点 (A.md)
      const filePath = path.join(baseDir, nodeName)
      const linkPath = `${basePath}/${nodeName.replace(/\.md$/, '')}`
      
      const content = fs.readFileSync(filePath, 'utf-8')
      const { data: frontmatter } = matter(content)
      const title = frontmatter.title || formatDirName(nodeName.replace(/\.md$/, ''))
      
      items.push({
        text: title,
        link: linkPath
      })
    }
  }
  
  return items
}

/**
 * 格式化目录名为标题
 * 01-hello-world -> Hello World
 */
function formatDirName(name: string): string {
  return name
    .replace(/^\d+-/, '')  // 移除数字前缀
    .replace(/-/g, ' ')    // 横线转空格
    .replace(/\b\w/g, c => c.toUpperCase())  // 首字母大写
}

/**
 * 为所有栏目生成完整的 sidebar 配置
 */
export function generateFullSidebar(docsDir: string): Record<string, SidebarItem[]> {
  const sections = ['posts', 'papers', 'knowledge', 'essays', 'thoughts', 'yearly']
  const sidebar: Record<string, SidebarItem[]> = {}
  
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
