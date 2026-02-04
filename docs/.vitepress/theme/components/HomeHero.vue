<template>
  <div class="home-hero">
    <div class="hero-content">
      <h1 class="hero-title">
        <span class="gradient-text">MetaUniverse</span>
      </h1>
      <p class="hero-subtitle">
        <span class="typewriter">{{ typedText }}</span><span class="cursor">|</span>
      </p>
      
      <div class="hero-actions">
        <a href="/papers/" class="action-btn primary">阅读论文</a>
        <a href="/knowledge/" class="action-btn secondary">查阅知识库</a>
      </div>
    </div>

    <div class="dashboard-grid">
      <a href="/papers/" class="dashboard-card papers">
        <div class="card-icon">📄</div>
        <h3>论文阅读</h3>
        <p>前沿 AI 论文深度解析与笔记</p>
      </a>
      <a href="/knowledge/" class="dashboard-card knowledge">
        <div class="card-icon">📚</div>
        <h3>知识库</h3>
        <p>LLM、CS336、DeepSeek 系统化知识</p>
      </a>
      <a href="/essays/" class="dashboard-card essays">
        <div class="card-icon">✍️</div>
        <h3>杂谈</h3>
        <p>技术之外的观察与思考</p>
      </a>
      <a href="/thoughts/" class="dashboard-card thoughts">
        <div class="card-icon">💡</div>
        <h3>随想</h3>
        <p>灵感片段与碎碎念</p>
      </a>
    </div>

    <div class="hero-background">
      <div class="blob blob-1"></div>
      <div class="blob blob-2"></div>
      <div class="blob blob-3"></div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

const slogans = [
  "探索 AI 的无限可能",
  "构建你的第二大脑",
  "从 0 到 1 理解大模型",
  "深度学习，深度思考"
]

const typedText = ref('')
let sloganIndex = 0
let charIndex = 0
let isDeleting = false
let timer: any = null

const typeEffect = () => {
  const currentSlogan = slogans[sloganIndex]
  
  if (isDeleting) {
    typedText.value = currentSlogan.substring(0, charIndex - 1)
    charIndex--
  } else {
    typedText.value = currentSlogan.substring(0, charIndex + 1)
    charIndex++
  }

  let typeSpeed = isDeleting ? 50 : 100

  if (!isDeleting && charIndex === currentSlogan.length) {
    typeSpeed = 2000 // Pause at end
    isDeleting = true
  } else if (isDeleting && charIndex === 0) {
    isDeleting = false
    sloganIndex = (sloganIndex + 1) % slogans.length
    typeSpeed = 500 // Pause before typing new
  }

  timer = setTimeout(typeEffect, typeSpeed)
}

const handleMouseMove = (e: MouseEvent) => {
  const cards = document.querySelectorAll('.dashboard-card')
  cards.forEach((card) => {
    const rect = (card as HTMLElement).getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    ;(card as HTMLElement).style.setProperty('--x', `${x}px`)
    ;(card as HTMLElement).style.setProperty('--y', `${y}px`)
  })
}

onMounted(() => {
  typeEffect()
  window.addEventListener('mousemove', handleMouseMove)
})

onUnmounted(() => {
  if (timer) clearTimeout(timer)
  window.removeEventListener('mousemove', handleMouseMove)
})
</script>

<style scoped>
/* ... existing styles ... */

.dashboard-card {
  background: var(--vp-c-bg-soft);
  padding: 2rem;
  border-radius: 1rem;
  border: 1px solid var(--vp-c-divider);
  text-decoration: none;
  transition: transform 0.3s ease, border-color 0.3s ease; /* Removed background transition for spotlight */
  backdrop-filter: blur(10px);
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  position: relative;
  overflow: hidden;
}

.dashboard-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: radial-gradient(
    800px circle at var(--x) var(--y),
    rgba(255, 255, 255, 0.06),
    transparent 40%
  );
  z-index: 0;
  pointer-events: none;
}

.dashboard-card > * {
  position: relative;
  z-index: 1;
}

/* Dark mode adjustment for spotlight */
:root.dark .dashboard-card::before {
  background: radial-gradient(
    800px circle at var(--x) var(--y),
    rgba(255, 255, 255, 0.1),
    transparent 40%
  );
}

.dashboard-card:hover {
  transform: translateY(-5px);
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.1);
  /* background: var(--vp-c-bg-mute);  Removed to let spotlight show */
}

.card-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
}

.dashboard-card h3 {
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
  color: var(--vp-c-text-1);
}

.dashboard-card p {
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
}

/* Animated Background */
.hero-background {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 0;
  overflow: hidden;
  opacity: 0.5;
  pointer-events: none;
}

.blob {
  position: absolute;
  border-radius: 50%;
  filter: blur(80px);
  opacity: 0.6;
  animation: float 10s infinite ease-in-out;
}

.blob-1 {
  width: 400px;
  height: 400px;
  background: var(--vp-c-brand-3);
  top: -100px;
  left: -100px;
  animation-delay: 0s;
}

.blob-2 {
  width: 300px;
  height: 300px;
  background: #d946ef; /* accent color */
  bottom: 10%;
  right: -50px;
  animation-delay: -2s;
}

.blob-3 {
  width: 250px;
  height: 250px;
  background: #0ea5e9; /* primary blue */
  top: 40%;
  left: 30%;
  animation-delay: -4s;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}

@keyframes float {
  0% { transform: translate(0, 0) scale(1); }
  33% { transform: translate(30px, -50px) scale(1.1); }
  66% { transform: translate(-20px, 20px) scale(0.9); }
  100% { transform: translate(0, 0) scale(1); }
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 640px) {
  .hero-title { font-size: 2.5rem; }
  .hero-subtitle { font-size: 1rem; }
  .dashboard-grid { grid-template-columns: 1fr; }
}
</style>
