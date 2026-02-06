<template>
  <div class="thoughts-dashboard">
    <!-- Hero with Gradient Mesh -->
    <div class="dashboard-hero">
      <div class="hero-bg">
        <div class="mesh-gradient"></div>
      </div>
      <div class="hero-content">
        <h1 class="dashboard-title">
          <span class="icon">💭</span>
          随想
        </h1>
        <p class="dashboard-desc">碎片化的灵感捕捉，未成体系的思考火花</p>
      </div>
    </div>

    <!-- Quote -->
    <div class="quote-card">
      <div class="quote-marks">"</div>
      <p class="quote-text">{{ currentQuote }}</p>
      <button class="refresh-btn" @click="refreshQuote">换一条</button>
    </div>

    <!-- Thoughts Masonry -->
    <div class="thoughts-masonry" v-if="thoughts.length > 0">
      <a 
        v-for="(thought, i) in thoughts" 
        :key="thought.link"
        :href="thought.link"
        class="thought-card"
        :class="'size-' + (i % 3)"
      >
        <span class="thought-emoji">{{ thought.emoji || '💡' }}</span>
        <h3 class="thought-title">{{ thought.title }}</h3>
        <p class="thought-snippet">{{ thought.snippet }}</p>
        <span class="thought-date">{{ thought.date }}</span>
      </a>
    </div>

    <!-- Empty State -->
    <div class="empty-prompt" v-else>
      <div class="empty-icon">🌌</div>
      <p>思绪如星辰，正在收集中...</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const quotes = [
  '写代码如写诗，追求优雅与简洁。',
  '在复杂系统中寻找秩序，是工程师的浪漫。',
  '每一次重构，都是对过去自己的超越。',
  '调参的尽头是玄学，玄学的尽头是信仰。',
  '当你凝视 Debug，Debug 也在凝视你。'
]

const currentQuote = ref(quotes[0])

const refreshQuote = () => {
  const current = currentQuote.value
  let newQuote = current
  while (newQuote === current) {
    newQuote = quotes[Math.floor(Math.random() * quotes.length)]
  }
  currentQuote.value = newQuote
}

const thoughts: Array<{
  title: string
  link: string
  snippet: string
  date: string
  emoji?: string
}> = []
</script>

<style scoped>
.thoughts-dashboard {
  max-width: 1000px;
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
  background: #1e1e2e;
}

.hero-bg {
  position: absolute;
  inset: 0;
  overflow: hidden;
}

.mesh-gradient {
  position: absolute;
  inset: -50%;
  background: 
    radial-gradient(circle at 20% 30%, rgba(236, 72, 153, 0.4) 0%, transparent 50%),
    radial-gradient(circle at 80% 70%, rgba(168, 85, 247, 0.4) 0%, transparent 50%),
    radial-gradient(circle at 50% 50%, rgba(59, 130, 246, 0.3) 0%, transparent 50%);
  animation: mesh-rotate 20s infinite linear;
}

@keyframes mesh-rotate {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
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
  color: rgba(255, 255, 255, 0.8);
  margin: 0;
}

/* Quote */
.quote-card {
  position: relative;
  padding: 2rem 3rem;
  background: var(--vp-c-bg-soft);
  border-radius: 1rem;
  margin-bottom: 2rem;
  border: 1px solid var(--vp-c-divider);
  text-align: center;
}

.quote-marks {
  position: absolute;
  top: -0.5rem;
  left: 1rem;
  font-size: 4rem;
  color: var(--vp-c-brand-1);
  opacity: 0.3;
  font-family: Georgia, serif;
  line-height: 1;
}

.quote-text {
  font-size: 1.1rem;
  font-style: italic;
  color: var(--vp-c-text-1);
  margin: 0 0 1rem;
  line-height: 1.6;
}

.refresh-btn {
  background: var(--vp-c-bg-mute);
  border: 1px solid var(--vp-c-divider);
  padding: 0.4rem 1rem;
  border-radius: 999px;
  font-size: 0.8rem;
  cursor: pointer;
  color: var(--vp-c-text-2);
  transition: all 0.2s ease;
}

.refresh-btn:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}

/* Masonry */
.thoughts-masonry {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
}

.thought-card {
  display: flex;
  flex-direction: column;
  padding: 1.5rem;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 1rem;
  text-decoration: none !important;
  transition: all 0.3s ease;
}

.thought-card.size-0 { min-height: 180px; }
.thought-card.size-1 { min-height: 220px; }
.thought-card.size-2 { min-height: 160px; }

.thought-card:hover {
  transform: translateY(-4px) rotate(-1deg);
  border-color: #a855f7;
  box-shadow: 0 16px 32px -8px rgba(168, 85, 247, 0.15);
}

.thought-emoji {
  font-size: 2rem;
  margin-bottom: 0.75rem;
}

.thought-title {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin: 0 0 0.5rem;
}

.thought-snippet {
  font-size: 0.9rem;
  color: var(--vp-c-text-2);
  margin: 0;
  flex: 1;
  line-height: 1.5;
}

.thought-date {
  font-size: 0.8rem;
  color: var(--vp-c-text-3);
  margin-top: 1rem;
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
  .thoughts-masonry { grid-template-columns: 1fr; }
}
</style>
