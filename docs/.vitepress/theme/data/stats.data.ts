import { createContentLoader } from 'vitepress'

export interface StatsData {
  totalArticles: number
  totalWords: number
  lastUpdate: string
}

declare const data: StatsData
export { data }

export default createContentLoader('knowledge/**/*.md', {
  includeSrc: true,
  transform(rawData) {
    let totalWords = 0
    let articleCount = 0
    const ignoredFiles = ['index.md', 'README.md']

    // Filter out index files and non-article pages
    const articles = rawData.filter(page => {
      const filename = page.url.split('/').pop() || ''
      return !ignoredFiles.includes(filename) && page.frontmatter.layout !== 'home'
    })

    articles.forEach(page => {
      if (page.src) {
        // Simple estimation for Chinese/English mixed content
        // Remove code blocks
        const pureText = page.src.replace(/```[\s\S]*?```/g, '')
        // Remove links and html
        .replace(/\[([^\]]+)\]\([^\)]+\)/g, '$1')
        .replace(/<[^>]*>/g, '')
        // Remove markdown headers/formatting
        .replace(/[#*`\-|_]/g, '')
        .trim()
        
        totalWords += pureText.length
      }
    })

    return {
      totalArticles: articles.length,
      totalWords,
      lastUpdate: new Date().toLocaleDateString('zh-CN')
    }
  }
})
