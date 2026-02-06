<template>
  <div class="papers-dashboard">
    <!-- Hero with Particle Effect -->
    <div class="dashboard-hero">
      <div class="hero-bg">
        <div class="particle" v-for="i in 20" :key="i" :style="getParticleStyle(i)"></div>
      </div>
      <div class="hero-content">
        <h1 class="dashboard-title">
          <span class="icon">📄</span>
          论文阅读
        </h1>
        <p class="dashboard-desc">站在巨人的肩膀上，解读前沿 AI 论文的核心思想</p>
      </div>
    </div>

    <!-- Reading Stats -->
    <div class="stats-bar">
      <div class="stat-item">
        <span class="stat-value">{{ papers.length }}</span>
        <span class="stat-label">篇论文</span>
      </div>
      <div class="stat-item">
        <span class="stat-value">{{ categories.length }}</span>
        <span class="stat-label">个方向</span>
      </div>
      <div class="stat-item">
        <span class="stat-value">📚 深度阅读</span>
        <span class="stat-label">风格</span>
      </div>
    </div>

    <!-- Category Tags -->
    <div class="category-tags">
      <span 
        v-for="cat in categories" 
        :key="cat.name"
        class="cat-tag"
        :class="{ active: activeCategory === cat.name }"
        @click="activeCategory = cat.name"
      >
        {{ cat.icon }} {{ cat.name }}
      </span>
    </div>

    <!-- Paper Cards -->
    <div class="paper-grid">
      <a 
        v-for="paper in filteredPapers" 
        :key="paper.link"
        :href="paper.link"
        class="paper-card"
      >
        <div class="paper-header">
          <span class="paper-year">{{ paper.year }}</span>
          <span class="paper-venue">{{ paper.venue }}</span>
        </div>
        <h3 class="paper-title">{{ paper.title }}</h3>
        <p class="paper-authors">{{ paper.authors }}</p>
        <p class="paper-tldr">{{ paper.tldr }}</p>
        <div class="paper-footer">
          <span class="read-more">阅读笔记 →</span>
        </div>
      </a>
    </div>

    <!-- Empty State -->
    <div class="empty-prompt" v-if="filteredPapers.length === 0">
      <div class="empty-icon">📚</div>
      <p>该方向暂无论文笔记，正在研读中...</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

const categories = [
  { name: '全部', icon: '🔖' },
  { name: 'Transformer', icon: '🤖' },
  { name: 'RLHF', icon: '🎯' },
  { name: 'MoE', icon: '🧩' },
  { name: 'Efficient Training', icon: '⚡' }
]

const activeCategory = ref('全部')

const papers: Array<{
  title: string
  link: string
  year: string
  venue: string
  authors: string
  tldr: string
  category: string
}> = []

const filteredPapers = computed(() => {
  if (activeCategory.value === '全部') return papers
  return papers.filter(p => p.category === activeCategory.value)
})

const getParticleStyle = (i: number) => ({
  '--x': `${Math.random() * 100}%`,
  '--y': `${Math.random() * 100}%`,
  '--size': `${Math.random() * 4 + 2}px`,
  '--delay': `${Math.random() * 5}s`,
  '--duration': `${Math.random() * 10 + 10}s`
})
</script>

<style scoped>
.papers-dashboard {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1.5rem 3rem;
}

/* Hero */
.dashboard-hero {
  position: relative;
  padding: 4rem 2rem;
  border-radius: 1.5rem;
  margin-bottom: 2rem;
  overflow: hidden;
  background: linear-gradient(135deg, #4c1d95 0%, #7c3aed 50%, #a78bfa 100%);
}

.hero-bg {
  position: absolute;
  inset: 0;
  overflow: hidden;
}

.particle {
  position: absolute;
  left: var(--x);
  top: var(--y);
  width: var(--size);
  height: var(--size);
  background: rgba(255, 255, 255, 0.3);
  border-radius: 50%;
  animation: particle-float var(--duration) var(--delay) infinite ease-in-out;
}

@keyframes particle-float {
  0%, 100% { transform: translateY(0) scale(1); opacity: 0.3; }
  50% { transform: translateY(-30px) scale(1.2); opacity: 0.6; }
}

.hero-content {
  position: relative;
  z-index: 1;
  text-align: center;
}

.dashboard-title {
  font-size: 2.5rem;
  font-weight: 800;
  margin: 0 0 1rem;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
}

.dashboard-title .icon {
  font-size: 2.5rem;
}

.dashboard-desc {
  font-size: 1.1rem;
  color: rgba(255, 255, 255, 0.9);
  margin: 0;
}

/* Stats */
.stats-bar {
  display: flex;
  justify-content: center;
  gap: 3rem;
  padding: 1.5rem;
  background: var(--vp-c-bg-soft);
  border-radius: 1rem;
  margin-bottom: 2rem;
  border: 1px solid var(--vp-c-divider);
}

.stat-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.25rem;
}

.stat-value {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--vp-c-brand-1);
}

.stat-label {
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
}

/* Category Tags */
.category-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin-bottom: 2rem;
  justify-content: center;
}

.cat-tag {
  padding: 0.5rem 1rem;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 999px;
  cursor: pointer;
  font-size: 0.9rem;
  transition: all 0.2s ease;
}

.cat-tag:hover {
  border-color: var(--vp-c-brand-1);
}

.cat-tag.active {
  background: var(--vp-c-brand-soft);
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
  font-weight: 600;
}

/* Paper Grid */
.paper-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: 1.5rem;
}

.paper-card {
  padding: 1.5rem;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 1rem;
  text-decoration: none !important;
  transition: all 0.3s ease;
}

.paper-card:hover {
  transform: translateY(-4px);
  border-color: #7c3aed;
  box-shadow: 0 16px 32px -8px rgba(124, 58, 237, 0.15);
}

.paper-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.75rem;
}

.paper-year {
  font-size: 0.8rem;
  font-weight: 600;
  color: #7c3aed;
}

.paper-venue {
  font-size: 0.75rem;
  background: var(--vp-c-bg-mute);
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  color: var(--vp-c-text-2);
}

.paper-title {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin: 0 0 0.5rem;
  line-height: 1.4;
}

.paper-authors {
  font-size: 0.85rem;
  color: var(--vp-c-text-3);
  margin: 0 0 0.75rem;
}

.paper-tldr {
  font-size: 0.9rem;
  color: var(--vp-c-text-2);
  margin: 0 0 1rem;
  line-height: 1.5;
}

.paper-footer {
  display: flex;
  justify-content: flex-end;
}

.read-more {
  font-size: 0.85rem;
  font-weight: 600;
  color: #7c3aed;
  opacity: 0;
  transform: translateX(-10px);
  transition: all 0.2s ease;
}

.paper-card:hover .read-more {
  opacity: 1;
  transform: translateX(0);
}

/* Empty State */
.empty-prompt {
  text-align: center;
  padding: 4rem 2rem;
  background: var(--vp-c-bg-soft);
  border-radius: 1rem;
  border: 1px dashed var(--vp-c-divider);
}

.empty-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
}

.empty-prompt p {
  color: var(--vp-c-text-2);
  font-size: 1.1rem;
}

/* Responsive */
@media (max-width: 768px) {
  .dashboard-title { font-size: 2rem; }
  .stats-bar { gap: 1.5rem; flex-wrap: wrap; }
  .paper-grid { grid-template-columns: 1fr; }
}
</style>
