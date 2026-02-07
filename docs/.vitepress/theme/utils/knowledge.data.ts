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
    // Optimize: Use Map for O(1) lookup instead of array.find O(N)
    const nodeMap = new Map<string, HelperNode>()
    const titleMap = new Map<string, HelperNode>()

    // 1. First pass: Create Nodes
    rawData.forEach(page => {
      // Safety check
      if (!page.url) return

      const title = page.frontmatter.title || 
                    page.url.split('/').filter(Boolean).pop() || 
                    'Home'
      
      const id = page.url
      
      const node: HelperNode = {
        id,
        name: title,
        val: 1,
        link: page.url
      }

      nodes.push(node)
      nodeMap.set(id, node)
      titleMap.set(title, node)
    })

    // 2. Second pass: Create Links
    // Optimize: simpler regex
    const linkRegex = /\[\[([^\]|]+)(?:\|[^\]]+)?\]\]/g

    rawData.forEach(page => {
      const srcContent = page.src || ''
      if (!srcContent) return

      let match
      
      while ((match = linkRegex.exec(srcContent)) !== null) {
        const rawTarget = match[1].trim()
        
        // 寻找匹配的节点
        // 1. 尝试完全匹配标题
        let targetNode = titleMap.get(rawTarget)
        
        // 2. 尝试部分匹配 (Fallback, slower but acceptable)
        if (!targetNode) {
           // Limit partial search to avoid O(N^2) on huge datasets if possible, 
           // but for now we keep it simple but maybe add a length check
           if (rawTarget.length > 1) {
             targetNode = nodes.find(n => n.name.includes(rawTarget))
           }
        }

        if (targetNode && targetNode.id !== page.url) {
          links.push({
            source: page.url,
            target: targetNode.id
          })
          
          // 增加节点权重
          targetNode.val += 0.5
          const sourceNode = nodeMap.get(page.url)
          if (sourceNode) sourceNode.val += 0.5
        }
      }
    })

    return { nodes, links }
  }
})
