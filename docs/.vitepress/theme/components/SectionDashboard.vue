<script setup lang="ts">
import { computed } from 'vue'
import { useData } from 'vitepress'

interface Article {
  title: string
  link: string
  description?: string
  date?: string
  icon?: string
}

interface Props {
  title: string
  description: string
  icon?: string
  accentColor?: string
  articles?: Article[]
}

const props = withDefaults(defineProps<Props>(), {
  icon: '📂',
  accentColor: '#0ea5e9',
  articles: () => []
})

const { isDark } = useData()

const articleCount = computed(() => props.articles.length)
</script>

<template>
  <div class="section-dashboard">
    <!-- Hero Section -->
    <div class="dashboard-hero" :style="{ '--accent': accentColor }">
      <div class="hero-bg"></div>
      <div class="hero-content">
        <span class="hero-icon">{{ icon }}</span>
        <h1 class="hero-title">{{ title }}</h1>
        <p class="hero-desc">{{ description }}</p>
        <div class="hero-stats">
          <span class="stat-item">
            <span class="stat-value">{{ articleCount }}</span>
            <span class="stat-label">篇文章</span>
          </span>
        </div>
      </div>
    </div>

    <!-- Article Grid -->
    <div class="articles-section" v-if="articles.length > 0">
      <h2 class="section-label">内容目录</h2>
      <div class="article-grid">
        <a
          v-for="article in articles"
          :key="article.link"
          :href="article.link"
          class="article-card"
        >
          <div class="card-icon">{{ article.icon || '📄' }}</div>
          <div class="card-body">
            <h3 class="card-title">{{ article.title }}</h3>
            <p v-if="article.description" class="card-desc">{{ article.description }}</p>
          </div>
          <span class="card-arrow">→</span>
        </a>
      </div>
    </div>

    <!-- Empty State -->
    <div class="empty-state" v-else>
      <div class="empty-icon">🚧</div>
      <p class="empty-text">暂无内容，敬请期待...</p>
    </div>
  </div>
</template>

<style scoped>
.section-dashboard {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1.5rem 2rem;
}

/* Hero Section */
.dashboard-hero {
  position: relative;
  padding: 3rem 2rem;
  margin: -1.5rem -1.5rem 2rem;
  border-radius: 0 0 1.5rem 1.5rem;
  overflow: hidden;
}

.hero-bg {
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, var(--accent) 0%, color-mix(in srgb, var(--accent) 60%, #8b5cf6) 100%);
  opacity: 0.1;
}

.dark .hero-bg {
  opacity: 0.2;
}

.hero-content {
  position: relative;
  z-index: 1;
  text-align: center;
}

.hero-icon {
  font-size: 3.5rem;
  display: block;
  margin-bottom: 1rem;
}

.hero-title {
  font-size: 2.5rem;
  font-weight: 700;
  margin: 0 0 0.75rem;
  color: var(--vp-c-text-1);
}

.hero-desc {
  font-size: 1.1rem;
  color: var(--vp-c-text-2);
  max-width: 600px;
  margin: 0 auto 1.5rem;
  line-height: 1.6;
}

.hero-stats {
  display: flex;
  justify-content: center;
  gap: 2rem;
}

.stat-item {
  display: flex;
  align-items: baseline;
  gap: 0.25rem;
}

.stat-value {
  font-size: 1.75rem;
  font-weight: 700;
  color: var(--accent);
}

.stat-label {
  font-size: 0.9rem;
  color: var(--vp-c-text-3);
}

/* Articles Section */
.articles-section {
  margin-top: 2rem;
}

.section-label {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--vp-c-text-2);
  margin-bottom: 1.25rem;
  padding-left: 0.5rem;
  border-left: 3px solid var(--accent, var(--vp-c-brand-1));
}

.article-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 1.25rem;
}

.article-card {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1.25rem 1.5rem;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 1rem;
  text-decoration: none !important;
  transition: all 0.25s ease;
}

.article-card:hover {
  transform: translateY(-4px);
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 12px 24px -8px rgba(0, 0, 0, 0.1);
}

.card-icon {
  font-size: 2rem;
  flex-shrink: 0;
}

.card-body {
  flex: 1;
  min-width: 0;
}

.card-title {
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin: 0 0 0.25rem;
}

.card-desc {
  font-size: 0.875rem;
  color: var(--vp-c-text-3);
  margin: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.card-arrow {
  font-size: 1.25rem;
  color: var(--vp-c-text-3);
  transition: transform 0.2s ease;
}

.article-card:hover .card-arrow {
  transform: translateX(4px);
  color: var(--vp-c-brand-1);
}

/* Empty State */
.empty-state {
  text-align: center;
  padding: 4rem 2rem;
}

.empty-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
}

.empty-text {
  font-size: 1.1rem;
  color: var(--vp-c-text-3);
}

/* Responsive */
@media (max-width: 768px) {
  .hero-title { font-size: 2rem; }
  .article-grid { grid-template-columns: 1fr; }
}
</style>
