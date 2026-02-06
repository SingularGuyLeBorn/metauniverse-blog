<template>
  <div class="yearly-dashboard">
    <!-- Hero with Counter -->
    <div class="dashboard-hero">
      <div class="hero-bg">
        <div class="year-ring" v-for="i in 4" :key="i" :class="'ring-' + i"></div>
      </div>
      <div class="hero-content">
        <h1 class="dashboard-title">
          <span class="icon">📅</span>
          年度总结
        </h1>
        <p class="dashboard-desc">记录每一年的成长轨迹与感悟</p>
        <div class="year-counter">
          <div class="counter-box">
            <span class="counter-value">{{ Object.keys(yearlyReviews).length }}</span>
            <span class="counter-label">年</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Year Cards -->
    <div class="year-grid" v-if="Object.keys(yearlyReviews).length > 0">
      <a 
        v-for="year in sortedYears" 
        :key="year"
        :href="yearlyReviews[year].link"
        class="year-card"
      >
        <div class="year-number">{{ year }}</div>
        <h3 class="year-title">{{ yearlyReviews[year].title }}</h3>
        <p class="year-keywords">
          <span v-for="kw in yearlyReviews[year].keywords" :key="kw" class="keyword">{{ kw }}</span>
        </p>
        <div class="year-stats">
          <div class="stat">
            <span class="stat-num">{{ yearlyReviews[year].articles }}</span>
            <span class="stat-text">篇文章</span>
          </div>
          <div class="stat">
            <span class="stat-num">{{ yearlyReviews[year].commits }}</span>
            <span class="stat-text">次提交</span>
          </div>
        </div>
      </a>
    </div>

    <!-- Empty State -->
    <div class="empty-prompt" v-else>
      <div class="empty-icon">🗓️</div>
      <p>新的一年，新的篇章正在书写...</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface YearReview {
  title: string
  link: string
  keywords: string[]
  articles: number
  commits: number
}

const yearlyReviews: Record<string, YearReview> = {}

const sortedYears = computed(() => 
  Object.keys(yearlyReviews).sort((a, b) => Number(b) - Number(a))
)
</script>

<style scoped>
.yearly-dashboard {
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
  background: linear-gradient(135deg, #0f766e 0%, #14b8a6 50%, #5eead4 100%);
}

.hero-bg {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.year-ring {
  position: absolute;
  border: 2px solid rgba(255, 255, 255, 0.2);
  border-radius: 50%;
  animation: pulse-ring 4s infinite ease-out;
}

.ring-1 { width: 100px; height: 100px; animation-delay: 0s; }
.ring-2 { width: 200px; height: 200px; animation-delay: 1s; }
.ring-3 { width: 300px; height: 300px; animation-delay: 2s; }
.ring-4 { width: 400px; height: 400px; animation-delay: 3s; }

@keyframes pulse-ring {
  0% { transform: scale(0.8); opacity: 0.8; }
  100% { transform: scale(1.5); opacity: 0; }
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
  margin: 0 0 1.5rem;
}

.year-counter {
  display: flex;
  justify-content: center;
}

.counter-box {
  display: flex;
  align-items: baseline;
  gap: 0.25rem;
  background: rgba(255, 255, 255, 0.2);
  padding: 0.5rem 1.5rem;
  border-radius: 999px;
  backdrop-filter: blur(8px);
}

.counter-value {
  font-size: 2rem;
  font-weight: 800;
  color: white;
}

.counter-label {
  font-size: 1rem;
  color: rgba(255, 255, 255, 0.9);
}

/* Year Grid */
.year-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
}

.year-card {
  position: relative;
  padding: 2rem;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 1.25rem;
  text-decoration: none !important;
  transition: all 0.3s ease;
  overflow: hidden;
}

.year-card:hover {
  transform: translateY(-4px);
  border-color: #14b8a6;
  box-shadow: 0 20px 40px -15px rgba(20, 184, 166, 0.2);
}

.year-number {
  font-size: 4rem;
  font-weight: 900;
  color: var(--vp-c-divider);
  position: absolute;
  right: 1rem;
  top: 0.5rem;
  line-height: 1;
  opacity: 0.5;
  transition: all 0.3s ease;
}

.year-card:hover .year-number {
  color: #14b8a6;
  opacity: 0.3;
}

.year-title {
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--vp-c-text-1);
  margin: 0 0 1rem;
  position: relative;
  z-index: 1;
}

.year-keywords {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin: 0 0 1.5rem;
}

.keyword {
  background: var(--vp-c-bg-mute);
  padding: 0.25rem 0.75rem;
  border-radius: 999px;
  font-size: 0.8rem;
  color: var(--vp-c-text-2);
}

.year-stats {
  display: flex;
  gap: 2rem;
  border-top: 1px solid var(--vp-c-divider);
  padding-top: 1rem;
}

.stat {
  display: flex;
  flex-direction: column;
}

.stat-num {
  font-size: 1.25rem;
  font-weight: 700;
  color: #14b8a6;
}

.stat-text {
  font-size: 0.8rem;
  color: var(--vp-c-text-2);
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
  .year-grid { grid-template-columns: 1fr; }
  .year-number { font-size: 3rem; }
}
</style>
