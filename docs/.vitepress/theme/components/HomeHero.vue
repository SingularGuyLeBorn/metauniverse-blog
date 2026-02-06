<template>
  <div class="home-hero">
    <div class="hero-background">
      <div class="blob blob-1"></div>
      <div class="blob blob-2"></div>
      <div class="blob blob-3"></div>
    </div>

    <div class="hero-content">
      <div class="title-container">
        <h1 class="hero-title">
          <span class="gradient-text">MetaUniverse</span>
        </h1>
        <div class="badge">AI & Tech Blog</div>
      </div>
      
      <p class="hero-subtitle">
        <span class="typewriter">{{ typedText }}</span><span class="cursor">|</span>
      </p>
      
      <div class="hero-actions">
        <a href="/papers/" class="action-btn primary">
          <span>阅读论文</span>
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>
        </a>
        <a href="/knowledge/" class="action-btn secondary">
          <span>查阅知识库</span>
        </a>
      </div>
    </div>

    <div class="dashboard-container">
      <div class="dashboard-grid">
        <a href="/posts/" class="dashboard-card posts">
          <div class="card-content">
            <div class="card-icon-wrapper">
              <span class="card-icon">✍️</span>
            </div>
            <div class="card-text">
              <h3>技术文章</h3>
              <p>Transformer、RLHF 等深度技术解析</p>
            </div>
            <div class="card-arrow">→</div>
          </div>
        </a>
        
        <a href="/papers/" class="dashboard-card papers">
          <div class="card-content">
            <div class="card-icon-wrapper">
              <span class="card-icon">📄</span>
            </div>
            <div class="card-text">
              <h3>论文阅读</h3>
              <p>前沿 AI 论文深度解析与笔记</p>
            </div>
            <div class="card-arrow">→</div>
          </div>
        </a>
        
        <a href="/knowledge/" class="dashboard-card knowledge">
          <div class="card-content">
            <div class="card-icon-wrapper">
              <span class="card-icon">📚</span>
            </div>
            <div class="card-text">
              <h3>知识库</h3>
              <p>LLM、CS336、DeepSeek 系统沉淀</p>
            </div>
            <div class="card-arrow">→</div>
          </div>
        </a>
        
        <a href="/posts/features-demo" class="dashboard-card features">
          <div class="card-content">
            <div class="card-icon-wrapper">
              <span class="card-icon">🧪</span>
            </div>
            <div class="card-text">
              <h3>实验特性</h3>
              <p>八大交互功能 Demo 演示</p>
            </div>
            <div class="card-arrow">→</div>
          </div>
        </a>

        <a href="/essays/" class="dashboard-card essays">
          <div class="card-content">
            <div class="card-icon-wrapper">
              <span class="card-icon">💬</span>
            </div>
            <div class="card-text">
              <h3>杂谈</h3>
              <p>技术之外的观察与思考</p>
            </div>
            <div class="card-arrow">→</div>
          </div>
        </a>
        
        <a href="/thoughts/" class="dashboard-card thoughts">
          <div class="card-content">
            <div class="card-icon-wrapper">
              <span class="card-icon">💡</span>
            </div>
            <div class="card-text">
              <h3>随想</h3>
              <p>灵感片段与碎碎念</p>
            </div>
            <div class="card-arrow">→</div>
          </div>
        </a>
      </div>
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
.home-hero {
  position: relative;
  min-height: calc(100vh - 64px); /* Subtract navbar height */
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 2rem;
  overflow: hidden;
  text-align: center;
}

/* Background */
.hero-background {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 0;
  overflow: hidden;
  pointer-events: none;
}

.blob {
  position: absolute;
  border-radius: 50%;
  filter: blur(80px);
  opacity: 0.4;
  animation: float 10s infinite ease-in-out;
}

.blob-1 {
  width: 500px;
  height: 500px;
  background: radial-gradient(circle, var(--vp-c-brand-3) 0%, transparent 70%);
  top: -20%;
  left: -10%;
  animation-delay: 0s;
}

.blob-2 {
  width: 400px;
  height: 400px;
  background: radial-gradient(circle, #d946ef 0%, transparent 70%);
  bottom: -10%;
  right: -5%;
  animation-delay: -2s;
}

.blob-3 {
  width: 300px;
  height: 300px;
  background: radial-gradient(circle, #0ea5e9 0%, transparent 70%);
  top: 40%;
  left: 30%;
  animation-delay: -4s;
}

/* Content Layout */
.hero-content {
  position: relative;
  z-index: 10;
  margin-bottom: 4rem;
  max-width: 800px;
  width: 100%;
}

.title-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.badge {
  background: var(--vp-c-bg-mute);
  padding: 0.25rem 0.75rem;
  border-radius: 999px;
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--vp-c-brand-1);
  border: 1px solid var(--vp-c-divider);
  letter-spacing: 0.05em;
  text-transform: uppercase;
  animation: fadeInDown 0.8s ease-out;
}

.hero-title {
  font-size: 5rem;
  line-height: 1.1;
  font-weight: 800;
  letter-spacing: -0.02em;
  margin: 0;
  padding: 0;
  animation: fadeUp 0.8s ease-out;
}

.gradient-text {
  background: linear-gradient(135deg, var(--vp-c-brand-1) 30%, #d946ef 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero-subtitle {
  font-size: 1.5rem;
  color: var(--vp-c-text-2);
  margin-bottom: 2.5rem;
  min-height: 1.5em;
  font-family: var(--vp-font-family-mono);
  animation: fadeUp 0.8s ease-out 0.2s backwards;
}

.cursor {
  display: inline-block;
  width: 2px;
  background-color: var(--vp-c-brand-1);
  animation: blink 1s infinite;
  margin-left: 4px;
}

/* Actions */
.hero-actions {
  display: flex;
  gap: 1rem;
  justify-content: center;
  animation: fadeUp 0.8s ease-out 0.4s backwards;
}

.action-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.875rem 2rem;
  border-radius: 999px;
  font-weight: 600;
  font-size: 1rem;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  text-decoration: none !important;
}

.action-btn.primary {
  background: linear-gradient(135deg, var(--vp-c-brand-1), var(--vp-c-brand-2));
  color: white;
  box-shadow: 0 4px 15px rgba(var(--vp-c-brand-1-rgb), 0.4);
}

.action-btn.primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(var(--vp-c-brand-1-rgb), 0.5);
}

.action-btn.secondary {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
  border: 1px solid var(--vp-c-divider);
}

.action-btn.secondary:hover {
  background: var(--vp-c-bg-mute);
  transform: translateY(-2px);
  border-color: var(--vp-c-text-2);
}

/* Dashboard Grid */
.dashboard-container {
  width: 100%;
  max-width: 1000px;
  position: relative;
  z-index: 10;
  animation: fadeUp 1s ease-out 0.6s backwards;
}

.dashboard-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1.5rem;
}

.dashboard-card {
  position: relative;
  background: var(--vp-c-bg-soft);
  border-radius: 1.5rem;
  border: 1px solid var(--vp-c-divider);
  text-decoration: none !important;
  overflow: hidden;
  transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
}

/* Spotlight Effect */
.dashboard-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: radial-gradient(
    600px circle at var(--x) var(--y),
    rgba(var(--vp-c-brand-1-rgb), 0.1),
    transparent 40%
  );
  z-index: 0;
  pointer-events: none;
}

.card-content {
  position: relative;
  z-index: 1;
  padding: 1.5rem;
  display: flex;
  align-items: center;
  gap: 1.5rem;
  height: 100%;
  backdrop-filter: blur(10px);
}

.card-icon-wrapper {
  width: 64px;
  height: 64px;
  border-radius: 1rem;
  background: var(--vp-c-bg-mute);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  transition: transform 0.3s ease;
  border: 1px solid var(--vp-c-divider);
}

.card-icon {
  font-size: 2rem;
}

.card-text {
  flex: 1;
  text-align: left;
}

.card-text h3 {
  font-size: 1.25rem;
  font-weight: 700;
  margin: 0 0 0.5rem 0;
  color: var(--vp-c-text-1);
}

.card-text p {
  font-size: 0.9rem;
  color: var(--vp-c-text-2);
  margin: 0;
  line-height: 1.4;
}

.card-arrow {
  opacity: 0;
  transform: translateX(-10px);
  transition: all 0.3s ease;
  color: var(--vp-c-brand-1);
  font-weight: 600;
  font-size: 1.25rem;
}

/* Hover Effects */
.dashboard-card:hover {
  transform: translateY(-4px);
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.1);
}

.dashboard-card:hover .card-icon-wrapper {
  transform: scale(1.1) rotate(5deg);
  background: white;
  border-color: transparent;
  box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

html.dark .dashboard-card:hover .card-icon-wrapper {
  background: var(--vp-c-bg-mute); 
}

.dashboard-card:hover .card-arrow {
  opacity: 1;
  transform: translateX(0);
}

/* Animations */
@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}

@keyframes float {
  0% { transform: translate(0, 0) scale(1); }
  33% { transform: translate(30px, -30px) scale(1.1); }
  66% { transform: translate(-20px, 20px) scale(0.9); }
  100% { transform: translate(0, 0) scale(1); }
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeInDown {
  from { opacity: 0; transform: translateY(-10px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Responsive */
@media (max-width: 1024px) {
  .dashboard-grid { grid-template-columns: repeat(2, 1fr); }
}

@media (max-width: 768px) {
  .hero-title { font-size: 3rem; }
  .dashboard-grid { grid-template-columns: 1fr; }
  .card-content { flex-direction: column; text-align: center; gap: 1rem; }
  .card-text { text-align: center; }
  .card-arrow { display: none; }
}
</style>
