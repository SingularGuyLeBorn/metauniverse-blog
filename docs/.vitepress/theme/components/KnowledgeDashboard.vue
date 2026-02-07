<template>
  <div class="knowledge-dashboard">
    <!-- Hero Section -->
    <section class="hero">
      <div class="hero-decoration">
        <div class="deco-circle c1"></div>
        <div class="deco-circle c2"></div>
        <div class="deco-circle c3"></div>
      </div>
      
      <div class="hero-inner">
        <span class="hero-label">
          <span class="label-dot"></span>
          持续更新中
        </span>
        
        <h1 class="hero-title">
          <span class="emoji">📚</span>
          <span>知识库</span>
        </h1>
        
        <p class="hero-subtitle">
          系统化的技术沉淀，从理论推导到工程实践
        </p>
        
        <div class="hero-metrics">
          <div class="metric">
            <span class="metric-value">{{ totalArticles }}</span>
            <span class="metric-label">篇文章</span>
          </div>
          <div class="metric-divider"></div>
          <div class="metric">
            <span class="metric-value">{{ totalSeries }}</span>
            <span class="metric-label">个系列</span>
          </div>
          <div class="metric-divider"></div>
          <div class="metric">
            <span class="metric-value">{{ totalWords }}</span>
            <span class="metric-label">总字数</span>
          </div>
        </div>
      </div>
    </section>

    <!-- Categories Section -->
    <section class="categories">
      <div class="section-title">
        <h2>🗂️ 知识系列</h2>
        <span class="section-desc">深入探索每个技术领域</span>
      </div>
      
      <div class="category-list">
        <a 
          v-for="category in categories" 
          :key="category.link"
          :href="category.link"
          class="category-card"
          :style="{ '--accent': category.color }"
        >
          <div class="card-icon">{{ category.icon }}</div>
          <div class="card-body">
            <div class="card-header">
              <h3>{{ category.title }}</h3>
              <span class="card-count">{{ category.count }} 篇</span>
            </div>
            <p class="card-desc">{{ category.description }}</p>
            <div class="card-tags">
              <span v-for="tag in category.tags?.slice(0, 3)" :key="tag">{{ tag }}</span>
            </div>
          </div>
          <div class="card-arrow">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M5 12h14M12 5l7 7-7 7"/>
            </svg>
          </div>
        </a>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useData } from 'vitepress'
import { data as statsData } from '../data/stats.data'

const { theme } = useData()

// Helper to count articles recursively
const countArticles = (items: any[]): number => {
  let count = 0
  for (const item of items) {
    if (item.link) count++
    if (item.items && Array.isArray(item.items)) {
      count += countArticles(item.items)
    }
  }
  return count
}

// Helper to get first link
const getFirstLink = (items: any[]): string => {
  for (const item of items) {
    if (item.link) return item.link
    if (item.items) {
      const link = getFirstLink(item.items)
      if (link) return link
    }
  }
  return '/knowledge/'
}

// Deterministic color generator
const colors = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16']
const getColor = (str: string) => {
  let hash = 0
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash)
  }
  return colors[Math.abs(hash) % colors.length]
}

// Icons mapping based on keywords
const getIcon = (title: string) => {
  if (title.includes('导论') || title.includes('基础')) return '🌱'
  if (title.includes('架构') || title.includes('原理')) return '🏗️'
  if (title.includes('预训练')) return '🏋️'
  if (title.includes('后训练') || title.includes('微调')) return '🔧'
  if (title.includes('模型')) return '🤖'
  if (title.includes('优化')) return '⚡'
  if (title.includes('应用')) return '🚀'
  if (title.includes('多模态')) return '👁️'
  if (title.includes('工程')) return '⚙️'
  if (title.includes('综述')) return '📑'
  if (title.includes('实践')) return '💻'
  if (title.includes('回顾')) return '📜'
  return '📚'
}

const categories = computed(() => {
  const sidebar = theme.value.sidebar
  let knowledgeItems: any[] = []
  
  if (sidebar['/knowledge/']) {
    knowledgeItems = sidebar['/knowledge/']
  } else {
    return []
  }
  
  // Try to find the group that contains the series
  // Structure: [ { text: '知识库', items: [ { text: 'Series 1' }, ... ] } ]
  let targetItems: any[] = []
  
  if (Array.isArray(knowledgeItems)) {
    // Flatten logic: Look for the first group that isn't empty
    for (const group of knowledgeItems) {
      if (group.items && group.items.length > 0) {
        // Filter out navigation items that aren't actual knowledge bases
        targetItems = group.items.filter((i: any) => 
          i.text !== '栏目首页' && 
          i.text !== '知识库首页' &&
          i.text !== '知识库所首页' &&
          i.link !== '/knowledge/' // Also filter by link to be safe
        )
        break;
      }
    }
  }
  
  return targetItems.map((item: any) => {
    // For each KB, lookup its own sidebar to count articles
    // E.g., for link="/knowledge/llm-mastery/", look for sidebar["/knowledge/llm-mastery/"]
    const kbPath = item.link?.endsWith('/') ? item.link : `${item.link}/`
    const kbSidebar = sidebar[kbPath]
    
    let articleCount = 0
    if (kbSidebar && Array.isArray(kbSidebar)) {
      // Count articles recursively from the KB's own sidebar
      for (const group of kbSidebar) {
        if (group.items) {
          articleCount += countArticles(group.items)
        }
      }
    }
    
    // Get tags from KB sidebar (first few sub-items)
    let tags: string[] = []
    if (kbSidebar && kbSidebar[0]?.items) {
      // Skip meta links like '返回知识库首页', '本库概览' 
      const contentItems = kbSidebar[0].items.filter((sub: any) => 
        !sub.text?.includes('返回') && 
        !sub.text?.includes('概览') &&
        !sub.text?.includes('首页')
      )
      tags = contentItems.slice(0, 3).map((sub: any) => 
        sub.text?.replace(/^[\d.]+\s*/, '').replace(/^📚\s*/, '') || ''
      ).filter(Boolean)
    }
    
    return {
      title: item.text,
      icon: getIcon(item.text),
      description: `包含 ${articleCount} 篇文章`,
      link: item.link || getFirstLink(item.items || []),
      count: articleCount,
      color: getColor(item.text),
      tags
    }
  })
})

// Use build-time stats for global metrics
const totalArticles = computed(() => statsData.totalArticles)
const totalWords = computed(() => (statsData.totalWords / 10000).toFixed(1) + 'w')
// Series count from sidebar logic
const totalSeries = computed(() => categories.value.length)
</script>

<style scoped>
/* ============ Base ============ */
.knowledge-dashboard {
  max-width: 960px;
  margin: 0 auto;
  padding: 3rem 1.5rem 5rem;
}

/* ============ Hero ============ */
.hero {
  position: relative;
  padding: 4rem 2rem;
  margin-bottom: 3rem;
  text-align: center;
  background: var(--vp-c-bg-soft);
  border-radius: 24px;
  border: 1px solid var(--vp-c-divider);
  overflow: hidden;
}

.hero-decoration {
  position: absolute;
  inset: 0;
  pointer-events: none;
  overflow: hidden;
}

.deco-circle {
  position: absolute;
  border-radius: 50%;
  opacity: 0.4;
  filter: blur(80px);
}

.c1 { width: 400px; height: 400px; background: #6366f1; top: -200px; left: -100px; }
.c2 { width: 300px; height: 300px; background: #10b981; bottom: -150px; right: -50px; }
.c3 { width: 200px; height: 200px; background: #f59e0b; top: 50%; right: 10%; }

.hero-inner { position: relative; z-index: 1; }

.hero-label {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 16px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 100px;
  font-size: 13px;
  color: var(--vp-c-text-2);
  margin-bottom: 24px;
}

.label-dot {
  width: 8px;
  height: 8px;
  background: #10b981;
  border-radius: 50%;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}

.hero-title {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
  margin: 0 0 16px;
  font-size: 48px;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--vp-c-text-1);
}

.hero-title .emoji { font-size: 42px; }

.hero-subtitle {
  margin: 0 0 32px;
  font-size: 18px;
  color: var(--vp-c-text-2);
  line-height: 1.6;
}

.hero-metrics {
  display: inline-flex;
  align-items: center;
  gap: 24px;
  padding: 16px 32px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 16px;
}

.metric {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}

.metric-value {
  font-size: 24px;
  font-weight: 700;
  color: var(--vp-c-brand-1);
}

.metric-label { font-size: 13px; color: var(--vp-c-text-2); }
.metric-divider { width: 1px; height: 32px; background: var(--vp-c-divider); }

/* ============ Section Title ============ */
.section-title {
  display: flex;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 20px;
}

.section-title h2 {
  margin: 0;
  font-size: 22px;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.section-desc { font-size: 14px; color: var(--vp-c-text-3); }

/* ============ Categories ============ */
.categories { margin-bottom: 48px; }
.category-list { display: flex; flex-direction: column; gap: 12px; }

.category-card {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 20px 24px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 16px;
  text-decoration: none !important;
  transition: all 0.2s ease;
}

.category-card:hover {
  border-color: var(--accent);
  transform: translateX(4px);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
}

.card-icon {
  flex-shrink: 0;
  width: 56px;
  height: 56px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 28px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 14px;
  transition: transform 0.2s;
}

.category-card:hover .card-icon { transform: scale(1.05); }

.card-body { flex: 1; min-width: 0; }

.card-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 6px;
}

.card-header h3 {
  margin: 0;
  font-size: 17px;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.card-count {
  padding: 2px 10px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 100px;
  font-size: 12px;
  color: var(--vp-c-text-2);
}

.card-desc {
  margin: 0 0 8px;
  font-size: 14px;
  color: var(--vp-c-text-2);
  line-height: 1.5;
}

.card-tags { display: flex; gap: 6px; }

.card-tags span {
  padding: 3px 10px;
  background: var(--vp-c-bg);
  border-radius: 6px;
  font-size: 12px;
  color: var(--vp-c-text-3);
}

.card-arrow {
  flex-shrink: 0;
  width: 24px;
  height: 24px;
  color: var(--vp-c-text-3);
  transition: all 0.2s;
}

.category-card:hover .card-arrow {
  color: var(--accent);
  transform: translateX(4px);
}

/* ============ Responsive ============ */
@media (max-width: 768px) {
  .knowledge-dashboard { padding: 1.5rem 1rem 3rem; }
  .hero { padding: 2.5rem 1.5rem; border-radius: 16px; }
  .hero-title { font-size: 32px; gap: 10px; flex-wrap: wrap; }
  .hero-title .emoji { font-size: 28px; }
  .hero-subtitle { font-size: 15px; }
  .hero-metrics { flex-wrap: wrap; gap: 16px; padding: 16px 20px; }
  .metric-divider { display: none; }
  .category-card { padding: 16px; gap: 14px; }
  .card-icon { width: 48px; height: 48px; font-size: 24px; }
}
</style>
