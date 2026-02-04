<template>
  <div class="about-profile">
    <div class="profile-header">
      <div class="avatar-container">
        <div class="avatar-placeholder">ME</div>
      </div>
      <div class="profile-info">
        <h2>博主 (Author)</h2>
        <p class="tagline">全栈 AI 工程师 / 开源爱好者 / 终身学习者</p>
        <div class="social-links">
          <a href="https://github.com/yourusername" target="_blank" class="social-btn github">GitHub</a>
          <a href="mailto:your.email@example.com" class="social-btn email">Email</a>
        </div>
      </div>
    </div>

    <div class="content-grid">
      <!-- 技能树 -->
      <div class="section skills">
        <h3>🛠 技术栈 (Tech Stack)</h3>
        <div class="skill-bars">
          <div class="skill-item" v-for="skill in skills" :key="skill.name">
            <div class="skill-info">
              <span>{{ skill.name }}</span>
              <span>{{ skill.level }}%</span>
            </div>
            <div class="progress-bar">
              <div class="progress-fill" :style="{ width: skill.level + '%' }"></div>
            </div>
          </div>
        </div>
      </div>

      <!-- 时间轴 -->
      <div class="section timeline">
        <h3>📅 经历 (Experience)</h3>
        <div class="timeline-container">
          <div class="timeline-item" v-for="(item, index) in experience" :key="index">
            <div class="timeline-dot"></div>
            <div class="timeline-content">
              <span class="timeline-date">{{ item.date }}</span>
              <h4>{{ item.title }}</h4>
              <p>{{ item.desc }}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
const skills = [
  { name: 'Python / PyTorch', level: 90 },
  { name: 'LLM Engineering (RAG, Fine-tuning)', level: 85 },
  { name: 'Vue.js / VitePress', level: 80 },
  { name: 'TypeScript', level: 75 },
  { name: 'Docker / K8s', level: 70 }
]

const experience = [
  {
    date: '2024 - Present',
    title: '独立开发者 / AI 研究员',
    desc: '专注于大模型应用落地与开源社区贡献。'
  },
  {
    date: '2022 - 2024',
    title: '高级算法工程师',
    desc: '负责推荐系统与 NLP 模型优化，主导了多次模型迭代。'
  },
  {
    date: '2020 - 2022',
    title: '全栈开发工程师',
    desc: '从后端开发转向 AI 领域，积累了扎实的工程化经验。'
  }
]
</script>

<style scoped>
.about-profile {
  margin-top: 2rem;
}

.profile-header {
  display: flex;
  align-items: center;
  gap: 2rem;
  background: var(--vp-c-bg-soft);
  padding: 2rem;
  border-radius: 1rem;
  margin-bottom: 2rem;
  border: 1px solid var(--vp-c-divider);
}

.avatar-placeholder {
  width: 100px;
  height: 100px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--vp-c-brand-1), var(--vp-c-brand-2));
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 2rem;
  font-weight: bold;
}

.profile-info h2 {
  margin: 0 0 0.5rem 0;
  border: none;
}

.tagline {
  color: var(--vp-c-text-2);
  margin-bottom: 1rem;
}

.social-links {
  display: flex;
  gap: 1rem;
}

.social-btn {
  padding: 0.25rem 1rem;
  border-radius: 0.25rem;
  background: var(--vp-c-bg-mute);
  color: var(--vp-c-text-1);
  text-decoration: none;
  font-size: 0.875rem;
  transition: all 0.2s;
}

.social-btn:hover {
  background: var(--vp-c-brand-1);
  color: white;
}

.content-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
}

.section {
  background: var(--vp-c-bg-soft);
  padding: 1.5rem;
  border-radius: 1rem;
  border: 1px solid var(--vp-c-divider);
}

.section h3 {
  margin-top: 0;
  margin-bottom: 1.5rem;
  border-bottom: 1px solid var(--vp-c-divider);
  padding-bottom: 0.5rem;
}

/* Skills */
.skill-item {
  margin-bottom: 1rem;
}

.skill-info {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.25rem;
  font-size: 0.875rem;
}

.progress-bar {
  height: 8px;
  background: var(--vp-c-bg-mute);
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: var(--vp-c-brand-1);
  border-radius: 4px;
  transition: width 1s ease-out;
}

/* Timeline */
.timeline-container {
  position: relative;
  border-left: 2px solid var(--vp-c-divider);
  padding-left: 1.5rem;
  margin-left: 0.5rem;
}

.timeline-item {
  position: relative;
  margin-bottom: 2rem;
}

.timeline-item:last-child {
  margin-bottom: 0;
}

.timeline-dot {
  position: absolute;
  left: -2.1rem;
  top: 0.25rem;
  width: 12px;
  height: 12px;
  background: var(--vp-c-brand-1);
  border-radius: 50%;
  border: 2px solid var(--vp-c-bg);
}

.timeline-date {
  font-size: 0.875rem;
  color: var(--vp-c-brand-1);
  font-weight: 600;
}

.timeline-content h4 {
  margin: 0.25rem 0;
  font-size: 1rem;
}

.timeline-content p {
  margin: 0;
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
}

@media (max-width: 768px) {
  .profile-header {
    flex-direction: column;
    text-align: center;
  }
  .content-grid {
    grid-template-columns: 1fr;
  }
}
</style>
