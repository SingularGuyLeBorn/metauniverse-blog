<template>
  <div class="posts-dashboard">
    <!-- Hero -->
    <section class="hero">
      <div class="hero-bg">
        <div class="bg-gradient"></div>
        <div class="code-pattern">
          <span v-for="i in 8" :key="i" class="code-line" :style="{ '--i': i }">
            {{ codeLines[(i - 1) % codeLines.length] }}
          </span>
        </div>
      </div>
      
      <div class="hero-inner">
        <span class="hero-label">
          <span class="label-icon">🔥</span>
          持续产出中
        </span>
        
        <h1 class="hero-title">
          <span class="emoji">✍️</span>
          <span>技术文章</span>
        </h1>
        
        <p class="hero-subtitle">
          <span class="typing">{{ displayText }}</span>
          <span class="cursor">|</span>
        </p>
        
        <div class="hero-metrics">
          <div class="metric">
            <span class="metric-value">{{ articles.length }}</span>
            <span class="metric-label">篇文章</span>
          </div>
          <div class="metric-divider"></div>
          <div class="metric">
            <span class="metric-value">{{ allTags.length }}</span>
            <span class="metric-label">个标签</span>
          </div>
        </div>
      </div>
    </section>

    <!-- Tag Filter -->
    <section class="filter" v-if="allTags.length > 0">
      <button 
        v-for="tag in ['全部', ...allTags]" 
        :key="tag"
        class="filter-tag"
        :class="{ active: activeTag === tag || (tag === '全部' && !activeTag) }"
        @click="activeTag = tag === '全部' ? '' : tag"
      >
        {{ tag }}
      </button>
    </section>

    <!-- Articles -->
    <section class="articles">
      <div class="section-title">
        <h2>📚 全部文章</h2>
        <span class="section-desc">{{ filteredArticles.length }} 篇干货</span>
      </div>
      
      <div class="article-grid" v-if="filteredArticles.length > 0">
        <a 
          v-for="article in filteredArticles" 
          :key="article.link"
          :href="article.link"
          class="article-card"
          :style="{ '--accent': article.color || '#0ea5e9' }"
        >
          <div class="card-accent"></div>
          <div class="card-icon">{{ article.icon || '📄' }}</div>
          <div class="card-body">
            <h3>{{ article.title }}</h3>
            <p>{{ article.description }}</p>
            <div class="card-meta">
              <span class="read-time">{{ article.readTime || '5 分钟' }}</span>
              <div class="card-tags">
                <span v-for="tag in article.tags?.slice(0, 2)" :key="tag">{{ tag }}</span>
              </div>
            </div>
          </div>
        </a>
      </div>
      
      <div class="empty" v-else>
        <span class="empty-icon">📝</span>
        <h3>{{ activeTag ? `没有 "${activeTag}" 相关文章` : '暂无文章' }}</h3>
        <p>{{ activeTag ? '试试其他标签' : '敬请期待...' }}</p>
        <button v-if="activeTag" class="reset-btn" @click="activeTag = ''">查看全部</button>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'

interface Article {
  title: string
  link: string
  description: string
  icon?: string
  color?: string
  tags?: string[]
  readTime?: string
}

const articles = ref<Article[]>([
  {
    title: 'Markdown 展示 Demo',
    link: '/posts/markdown-demo/',
    description: '展示 VitePress 支持的 Markdown 语法和扩展功能',
    icon: '📝',
    color: '#0ea5e9',
    tags: ['Markdown', 'VitePress'],
    readTime: '8 分钟'
  }
])

const codeLines = [
  'def train(model):',
  'loss = criterion()',
  'optimizer.step()',
  'model.forward(x)',
  'torch.cuda.empty()',
  'attention @ value',
  'grads = backward()',
  'checkpoint.save()'
]

const phrases = [
  '深度解析 Transformer 架构原理...',
  '从零实现 RLHF 训练流程...',
  '显存优化的艺术与实践...',
  '端侧推理部署全攻略...'
]

const displayText = ref('')
const activeTag = ref('')

let phraseIndex = 0
let charIndex = 0
let isDeleting = false
let typingTimeout: ReturnType<typeof setTimeout> | null = null

const allTags = computed(() => {
  const tagSet = new Set<string>()
  articles.value.forEach(a => a.tags?.forEach(t => tagSet.add(t)))
  return Array.from(tagSet)
})

const filteredArticles = computed(() => {
  if (!activeTag.value) return articles.value
  return articles.value.filter(a => a.tags?.includes(activeTag.value))
})

const typeEffect = () => {
  const currentPhrase = phrases[phraseIndex]
  
  if (isDeleting) {
    displayText.value = currentPhrase.substring(0, charIndex - 1)
    charIndex--
  } else {
    displayText.value = currentPhrase.substring(0, charIndex + 1)
    charIndex++
  }

  let delay = isDeleting ? 30 : 60

  if (!isDeleting && charIndex === currentPhrase.length) {
    delay = 2000
    isDeleting = true
  } else if (isDeleting && charIndex === 0) {
    isDeleting = false
    phraseIndex = (phraseIndex + 1) % phrases.length
    delay = 500
  }

  typingTimeout = setTimeout(typeEffect, delay)
}

onMounted(() => typeEffect())
onUnmounted(() => { if (typingTimeout) clearTimeout(typingTimeout) })
</script>

<style scoped>
/* ============ Base ============ */
.posts-dashboard {
  max-width: 960px;
  margin: 0 auto;
  padding: 3rem 1.5rem 5rem;
}

/* ============ Hero ============ */
.hero {
  position: relative;
  padding: 4rem 2rem;
  margin-bottom: 2rem;
  text-align: center;
  background: linear-gradient(135deg, #0369a1, #0ea5e9);
  border-radius: 24px;
  overflow: hidden;
}

.hero-bg {
  position: absolute;
  inset: 0;
  overflow: hidden;
}

.bg-gradient {
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at 30% 20%, rgba(255,255,255,0.1) 0%, transparent 50%);
}

.code-pattern {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  justify-content: space-around;
  padding: 20px 40px;
  pointer-events: none;
}

.code-line {
  font-family: 'Fira Code', monospace;
  font-size: 12px;
  color: rgba(255, 255, 255, 0.08);
  white-space: nowrap;
  animation: float 8s ease-in-out infinite;
  animation-delay: calc(var(--i) * 0.5s);
}

.code-line:nth-child(even) {
  text-align: right;
}

@keyframes float {
  0%, 100% { transform: translateX(0); }
  50% { transform: translateX(10px); }
}

.hero-inner {
  position: relative;
  z-index: 1;
}

.hero-label {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 16px;
  background: rgba(255, 255, 255, 0.15);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 100px;
  font-size: 13px;
  color: white;
  margin-bottom: 24px;
}

.label-icon {
  animation: pulse-icon 2s infinite;
}

@keyframes pulse-icon {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.2); }
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
  color: white;
}

.hero-title .emoji {
  font-size: 42px;
}

.hero-subtitle {
  margin: 0 0 32px;
  font-size: 18px;
  color: rgba(255, 255, 255, 0.9);
  min-height: 1.5em;
}

.cursor {
  animation: blink 1s step-end infinite;
  margin-left: 2px;
}

@keyframes blink {
  50% { opacity: 0; }
}

.hero-metrics {
  display: inline-flex;
  align-items: center;
  gap: 24px;
  padding: 16px 32px;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 16px;
  backdrop-filter: blur(10px);
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
  color: white;
}

.metric-label {
  font-size: 13px;
  color: rgba(255, 255, 255, 0.7);
}

.metric-divider {
  width: 1px;
  height: 32px;
  background: rgba(255, 255, 255, 0.2);
}

/* ============ Filter ============ */
.filter {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
  margin-bottom: 2rem;
  padding: 16px;
  background: var(--vp-c-bg-soft);
  border-radius: 16px;
  border: 1px solid var(--vp-c-divider);
}

.filter-tag {
  padding: 8px 16px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 100px;
  font-size: 14px;
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition: all 0.2s;
}

.filter-tag:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}

.filter-tag.active {
  background: var(--vp-c-brand-soft);
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
  font-weight: 500;
}

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

.section-desc {
  font-size: 14px;
  color: var(--vp-c-text-3);
}

/* ============ Articles ============ */
.article-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 16px;
}

.article-card {
  display: flex;
  gap: 16px;
  padding: 20px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 16px;
  text-decoration: none !important;
  transition: all 0.2s;
  position: relative;
  overflow: hidden;
}

.article-card:hover {
  transform: translateY(-4px);
  border-color: var(--accent);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
}

.card-accent {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--accent);
  transform: scaleX(0);
  transform-origin: left;
  transition: transform 0.3s;
}

.article-card:hover .card-accent {
  transform: scaleX(1);
}

.card-icon {
  flex-shrink: 0;
  width: 48px;
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
}

.card-body {
  flex: 1;
  min-width: 0;
}

.card-body h3 {
  margin: 0 0 8px;
  font-size: 16px;
  font-weight: 600;
  color: var(--vp-c-text-1);
  line-height: 1.4;
}

.card-body p {
  margin: 0 0 12px;
  font-size: 14px;
  color: var(--vp-c-text-2);
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.card-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.read-time {
  font-size: 12px;
  color: var(--vp-c-text-3);
}

.card-tags {
  display: flex;
  gap: 6px;
}

.card-tags span {
  padding: 2px 8px;
  background: var(--vp-c-bg);
  border-radius: 4px;
  font-size: 11px;
  color: var(--vp-c-text-3);
}

/* ============ Empty ============ */
.empty {
  text-align: center;
  padding: 4rem 2rem;
  background: var(--vp-c-bg-soft);
  border-radius: 16px;
  border: 2px dashed var(--vp-c-divider);
}

.empty-icon {
  font-size: 48px;
  display: block;
  margin-bottom: 16px;
}

.empty h3 {
  margin: 0 0 8px;
  color: var(--vp-c-text-1);
}

.empty p {
  margin: 0 0 20px;
  color: var(--vp-c-text-2);
}

.reset-btn {
  padding: 10px 24px;
  background: var(--vp-c-brand-1);
  color: white;
  border: none;
  border-radius: 100px;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s;
}

.reset-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

/* ============ Responsive ============ */
@media (max-width: 768px) {
  .posts-dashboard {
    padding: 1.5rem 1rem 3rem;
  }
  
  .hero {
    padding: 2.5rem 1.5rem;
    border-radius: 16px;
  }
  
  .hero-title {
    font-size: 32px;
    gap: 10px;
  }
  
  .hero-title .emoji {
    font-size: 28px;
  }
  
  .hero-subtitle {
    font-size: 15px;
  }
  
  .hero-metrics {
    padding: 12px 20px;
    gap: 16px;
  }
  
  .article-grid {
    grid-template-columns: 1fr;
  }
  
  .article-card {
    flex-direction: column;
    gap: 12px;
  }
  
  .card-icon {
    width: 40px;
    height: 40px;
    font-size: 20px;
  }
}
</style>
