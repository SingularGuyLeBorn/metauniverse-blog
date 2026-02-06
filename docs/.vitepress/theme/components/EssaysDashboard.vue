<template>
  <div class="essays-dashboard">
    <!-- Hero with Wave Effect -->
    <div class="dashboard-hero">
      <div class="hero-bg">
        <svg class="wave-svg" viewBox="0 0 1440 320" preserveAspectRatio="none">
          <path class="wave wave-1" d="M0,192L48,197.3C96,203,192,213,288,229.3C384,245,480,267,576,250.7C672,235,768,181,864,181.3C960,181,1056,235,1152,234.7C1248,235,1344,181,1392,154.7L1440,128L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"></path>
          <path class="wave wave-2" d="M0,256L48,261.3C96,267,192,277,288,261.3C384,245,480,203,576,197.3C672,192,768,224,864,234.7C960,245,1056,235,1152,213.3C1248,192,1344,160,1392,144L1440,128L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"></path>
        </svg>
      </div>
      <div class="hero-content">
        <h1 class="dashboard-title">
          <span class="icon">💬</span>
          杂谈
        </h1>
        <p class="dashboard-desc">技术之外的话题，关于生活、思考与人生</p>
      </div>
    </div>

    <!-- Stats -->
    <div class="stats-bar">
      <div class="stat-item">
        <span class="stat-value">{{ essays.length }}</span>
        <span class="stat-label">篇随笔</span>
      </div>
      <div class="stat-item">
        <span class="stat-value">☕ 慢慢写</span>
        <span class="stat-label">节奏</span>
      </div>
    </div>

    <!-- Essay Timeline -->
    <div class="essay-timeline" v-if="essays.length > 0">
      <a 
        v-for="essay in essays" 
        :key="essay.link"
        :href="essay.link"
        class="essay-item"
      >
        <div class="timeline-dot"></div>
        <div class="essay-content">
          <span class="essay-date">{{ essay.date }}</span>
          <h3 class="essay-title">{{ essay.title }}</h3>
          <p class="essay-excerpt">{{ essay.excerpt }}</p>
        </div>
      </a>
    </div>

    <!-- Empty State -->
    <div class="empty-prompt" v-else>
      <div class="empty-icon">☕</div>
      <p>泡一杯咖啡，静待灵感降临...</p>
    </div>
  </div>
</template>

<script setup lang="ts">
const essays: Array<{
  title: string
  link: string
  date: string
  excerpt: string
}> = []
</script>

<style scoped>
.essays-dashboard {
  max-width: 900px;
  margin: 0 auto;
  padding: 0 1.5rem 3rem;
}

/* Hero */
.dashboard-hero {
  position: relative;
  padding: 4rem 2rem 6rem;
  border-radius: 1.5rem;
  margin-bottom: 2rem;
  overflow: hidden;
  background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 50%, #fcd34d 100%);
}

.hero-bg {
  position: absolute;
  inset: 0;
  overflow: hidden;
}

.wave-svg {
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  height: 100px;
}

.wave {
  fill: var(--vp-c-bg);
}

.wave-1 {
  opacity: 0.5;
  animation: wave-move 8s infinite ease-in-out;
}

.wave-2 {
  opacity: 0.8;
  animation: wave-move 6s infinite ease-in-out reverse;
}

@keyframes wave-move {
  0%, 100% { transform: translateX(0); }
  50% { transform: translateX(-20px); }
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
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.dashboard-title .icon {
  font-size: 2.5rem;
}

.dashboard-desc {
  font-size: 1.1rem;
  color: rgba(255, 255, 255, 0.95);
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
  margin-bottom: 2.5rem;
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
  color: #f59e0b;
}

.stat-label {
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
}

/* Essay Timeline */
.essay-timeline {
  position: relative;
  padding-left: 2rem;
  border-left: 2px solid var(--vp-c-divider);
}

.essay-item {
  position: relative;
  display: block;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 1rem;
  text-decoration: none !important;
  transition: all 0.3s ease;
}

.essay-item:hover {
  transform: translateX(8px);
  border-color: #f59e0b;
  box-shadow: 0 8px 24px -8px rgba(245, 158, 11, 0.2);
}

.timeline-dot {
  position: absolute;
  left: -2.5rem;
  top: 1.75rem;
  width: 12px;
  height: 12px;
  background: #f59e0b;
  border-radius: 50%;
  border: 3px solid var(--vp-c-bg);
  transition: transform 0.2s ease;
}

.essay-item:hover .timeline-dot {
  transform: scale(1.3);
}

.essay-date {
  font-size: 0.85rem;
  color: #f59e0b;
  font-weight: 600;
}

.essay-title {
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin: 0.5rem 0;
}

.essay-excerpt {
  font-size: 0.9rem;
  color: var(--vp-c-text-2);
  margin: 0;
  line-height: 1.5;
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
}
</style>
