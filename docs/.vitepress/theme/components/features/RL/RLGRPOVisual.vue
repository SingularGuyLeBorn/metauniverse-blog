<template>
  <RLVisualizer title="GRPO: 组内相对优势可视化 (Group Relative Policy Optimization)">
    <div class="grpo-visual">
      <div class="sampling-display">
        <div class="group-header">
          <span>组大小 (Group Size) G = {{ groupSize }}</span>
          <button @click="sampleNewGroup" class="btn-sm">重新采样 (Resample)</button>
        </div>
        
        <div class="samples-grid">
          <div v-for="(sample, index) in samples" :key="index" class="sample-card" :class="{ 'is-best': sample.advantage > 0.8 }">
            <div class="sample-id">样本 {{ index + 1 }}</div>
            <div class="reward-bar-container">
              <div class="reward-bar" :style="{ width: (sample.reward * 10) + '%', background: getRewardColor(sample.reward) }"></div>
            </div>
            <div class="sample-stats">
              <span>奖励: {{ sample.reward.toFixed(1) }}</span>
              <span class="adv" :class="sample.advantage >= 0 ? 'pos' : 'neg'">
                {{ sample.advantage >= 0 ? '+' : '' }}{{ sample.advantage.toFixed(2) }}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div class="calculation-logic glass-card">
        <h5>优势函数计算 (Advantage Calculation)</h5>
        <div class="calc-step">
          <label>1. 计算均值 (Mean Reward):</label>
          <span class="math">\mu = \frac{1}{G} \sum r_i = {{ meanReward.toFixed(2) }}</span>
        </div>
        <div class="calc-step">
          <label>2. 计算标准差 (Std Dev):</label>
          <span class="math">\sigma = \sqrt{\frac{1}{G} \sum (r_i - \mu)^2} = {{ stdReward.toFixed(2) }}</span>
        </div>
        <div class="calc-step">
          <label>3. 相对优势 (Relative Advantage):</label>
          <span class="math">A_i = \frac{r_i - \mu}{\sigma}</span>
        </div>
        <p class="insight">
          <i class="i-carbon-idea"></i>
          不需要 Critic 网络，仅靠组内对比即可获得基准。
        </p>
      </div>
    </div>

    <template #footer>
      <p>GRPO 是 DeepSeek-R1 的核心创新。它通过对同一 Prompt 采样多组输出，利用组内奖励的相对差异来替代传统的 Critic 价值网络，极大地节省了显存开销。</p>
    </template>
  </RLVisualizer>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import RLVisualizer from './RLVisualizer.vue'

const groupSize = 8
const samples = ref<any[]>([])

const sampleNewGroup = () => {
  const newSamples = []
  for (let i = 0; i < groupSize; i++) {
    // Random reward between 1 and 10
    newSamples.push({ reward: Math.random() * 9 + 1, advantage: 0 })
  }
  
  // Calculate stats
  const rewards = newSamples.map(s => s.reward)
  const mean = rewards.reduce((a, b) => a + b) / groupSize
  const variance = rewards.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / groupSize
  const std = Math.sqrt(variance) || 1

  newSamples.forEach(s => {
    s.advantage = (s.reward - mean) / std
  })

  samples.value = newSamples
}

const meanReward = computed(() => {
  if (samples.value.length === 0) return 0
  return samples.value.map(s => s.reward).reduce((a, b) => a + b) / groupSize
})

const stdReward = computed(() => {
  if (samples.value.length === 0) return 1
  const mean = meanReward.value
  const variance = samples.value.reduce((a, b) => a + Math.pow(b.reward - mean, 2), 0) / groupSize
  return Math.sqrt(variance) || 1
})

const getRewardColor = (reward: number) => {
  if (reward > 7) return '#10b981'
  if (reward > 4) return '#0ea5e9'
  return '#f59e0b'
}

onMounted(() => {
  sampleNewGroup()
})
</script>

<style scoped>
.grpo-visual {
  display: flex;
  flex-direction: column;
  gap: 2rem;
  width: 100%;
}

.group-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.btn-sm {
  font-size: 0.75rem;
  padding: 0.25rem 0.75rem;
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  border: 1px solid var(--vp-c-brand-1);
  border-radius: 4px;
}

.samples-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 1rem;
}

.sample-card {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 0.75rem;
  transition: transform 0.2s;
}

.sample-card.is-best {
  border-color: #10b981;
  box-shadow: 0 0 10px rgba(16, 185, 129, 0.2);
}

.sample-id {
  font-size: 0.7rem;
  color: var(--vp-c-text-3);
  margin-bottom: 0.5rem;
}

.reward-bar-container {
  height: 4px;
  background: rgba(148, 163, 184, 0.1);
  border-radius: 2px;
  margin-bottom: 0.5rem;
  overflow: hidden;
}

.reward-bar {
  height: 100%;
  transition: width 0.5s ease-out;
}

.sample-stats {
  display: flex;
  justify-content: space-between;
  font-size: 0.8rem;
}

.adv { font-weight: 700; }
.pos { color: #10b981; }
.neg { color: #ef4444; }

.calculation-logic {
  padding: 1.5rem;
}

.calculation-logic h5 {
  margin: 0 0 1rem 0;
  font-size: 1rem;
  color: var(--vp-c-text-1);
}

.calc-step {
  margin-bottom: 0.75rem;
  font-size: 0.9rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.calc-step label { color: var(--vp-c-text-2); }
.math { font-family: serif; font-style: italic; color: var(--vp-c-brand-1); }

.insight {
  margin-top: 1rem;
  font-size: 0.8rem;
  color: #10b981;
  background: rgba(16, 185, 129, 0.05);
  padding: 0.5rem;
  border-radius: 4px;
}

@media (min-width: 960px) {
  .grpo-visual { flex-direction: row; align-items: stretch; }
  .sampling-display { flex: 1; }
  .calculation-logic { width: 320px; }
}
</style>
