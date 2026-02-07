import { createContentLoader } from 'vitepress'
import path from 'path'

/**
 * Knowledge Graph Data Loader
 * 扫描所有文档，提取 [[WikiLinks]]，生成节点和边的数据
 */

export interface HelperNode {
  id: string
  name: string
  val: number // weight/size
  group?: number
  link: string
}

export interface HelperLink {
  source: string
  target: string
}

export declare const data: {
  nodes: HelperNode[]
  links: HelperLink[]
}

export default createContentLoader('**/*.md', {
  includeSrc: true, // 需要读取内容来解析链接
  render: true,     // Optional: 让 vitepress 先渲染 markdown (如果需要 html)
  transform(rawData) {
    const nodes: HelperNode[] = []
    const links: HelperLink[] = []
    const idMap = new Map<string, boolean>()

    // 1. First pass: Create Nodes
    rawData.forEach(page => {
      // 忽略 index.md (通常是目录页，或者只作为入口) - 视情况而定
      const isIndex = page.url.endsWith('/')
      
      const title = page.frontmatter.title || 
                    page.url.split('/').filter(Boolean).pop() || 
                    'Home'
      
      const id = page.url
      
      nodes.push({
        id,
        name: title,
        val: 1,
        link: page.url
      })
      
      idMap.set(id, true)
    })

    // 2. Second pass: Create Links
    const linkRegex = /\[\[([^\]|]+)(?:\|[^\]]+)?\]\]/g

    rawData.forEach(page => {
      const srcContent = page.src || ''
      let match
      
      // Reset regex state
      linkRegex.lastIndex = 0
      
      while ((match = linkRegex.exec(srcContent)) !== null) {
        const rawTarget = match[1].trim()
        // 尝试匹配目标 URL
        // 这里只是简单的匹配文件名或路径，实际逻辑可能更复杂
        // 假设 wiki link 只是文件名
        
        // 寻找匹配的节点
        // 1. 尝试完全匹配
        let targetNode = nodes.find(n => n.name === rawTarget)
        
        // 2. 尝试部分匹配 (如 "2.1 深度学习" 匹配 "深度学习")
        if (!targetNode) {
          targetNode = nodes.find(n => n.name.includes(rawTarget))
        }

        if (targetNode && targetNode.id !== page.url) {
          links.push({
            source: page.url,
            target: targetNode.id
          })
          
          // 增加节点权重
          targetNode.val += 0.5
          const sourceNode = nodes.find(n => n.id === page.url)
          if (sourceNode) sourceNode.val += 0.5
        }
      }
    })

    return { nodes, links }
  }
})
