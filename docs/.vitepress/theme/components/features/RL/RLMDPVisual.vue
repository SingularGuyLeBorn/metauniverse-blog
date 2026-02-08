<template>
  <RLVisualizer title="MDP: 状态转移与奖励 (MDP: State Transitions & Reward)">
    <div class="mdp-container">
      <div class="grid-world">
        <div v-for="(cell, index) in grid" :key="index" 
             class="grid-cell" 
             :class="{ 
               'is-agent': agentPos === index, 
               'is-goal': cell === 'G', 
               'is-trap': cell === 'T' 
             }">
          <span v-if="cell === 'G'">🏁</span>
          <span v-if="cell === 'T'">⚠️</span>
          <div v-if="agentPos === index" class="agent-puck">🤖</div>
        </div>
      </div>

      <div class="mdp-info">
        <div class="info-item">
          <label>当前状态 (State):</label>
          <span>S_{{ agentPos }}</span>
        </div>
        <div class="info-item">
          <label>累计奖励 (Cumulative Reward):</label>
          <span :class="{ 'pos': totalReward > 0, 'neg': totalReward < 0 }">{{ totalReward }}</span>
        </div>
        <div class="info-item">
          <label>步数 (Step):</label>
          <span>{{ stepCount }}</span>
        </div>
        
        <div class="controls-panel">
          <button @click="reset" class="btn-secondary">重置 (Reset)</button>
          <div class="action-buttons">
            <button @click="move('up')" :disabled="isGameOver">⬆️</button>
            <div class="middle-row">
              <button @click="move('left')" :disabled="isGameOver">⬅️</button>
              <button @click="move('right')" :disabled="isGameOver">➡️</button>
            </div>
            <button @click="move('down')" :disabled="isGameOver">⬇️</button>
          </div>
        </div>
      </div>
    </div>

    <template #footer>
      <p>MDP (马尔可夫决策过程) 是强化学习的数学基石。在这里，智能体 (Agent) 通过在状态空间中采取动作，从环境获得反馈（奖励/惩罚），并转移到新状态。</p>
    </template>
  </RLVisualizer>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import RLVisualizer from './RLVisualizer.vue'

const size = 4
const grid = ref(['', '', '', 'G', '', 'T', '', '', '', '', '', '', '', '', '', ''])
const agentPos = ref(12) // Bottom left start
const totalReward = ref(0)
const stepCount = ref(0)
const isGameOver = ref(false)

const move = (direction: string) => {
  if (isGameOver.value) return

  let row = Math.floor(agentPos.value / size)
  let col = agentPos.value % size

  if (direction === 'up' && row > 0) row--
  else if (direction === 'down' && row < size - 1) row++
  else if (direction === 'left' && col > 0) col--
  else if (direction === 'right' && col < size - 1) col++
  else return // Boundary

  agentPos.value = row * size + col
  stepCount.value++

  // Update Reward
  if (grid.value[agentPos.value] === 'G') {
    totalReward.value += 100
    isGameOver.value = true
  } else if (grid.value[agentPos.value] === 'T') {
    totalReward.value -= 50
    isGameOver.value = true
  } else {
    totalReward.value -= 1 // Step penalty
  }
}

const reset = () => {
  agentPos.value = 12
  totalReward.value = 0
  stepCount.value = 0
  isGameOver.value = false
}
</script>

<style scoped>
.mdp-container {
  display: flex;
  gap: 2rem;
  width: 100%;
  align-items: flex-start;
}

.grid-world {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
  background: rgba(14, 165, 233, 0.1);
  padding: 8px;
  border-radius: 8px;
  border: 1px solid rgba(14, 165, 233, 0.2);
}

.grid-cell {
  width: 60px;
  height: 60px;
  background: var(--vp-c-bg-soft);
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.5rem;
  position: relative;
  border: 1px solid var(--vp-c-divider);
}

.is-goal { background: rgba(16, 185, 129, 0.2); border-color: #10b981; }
.is-trap { background: rgba(239, 68, 68, 0.2); border-color: #ef4444; }

.agent-puck {
  position: absolute;
  z-index: 2;
  transition: all 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

.mdp-info {
  flex-grow: 1;
}

.info-item {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.75rem;
  font-size: 0.9rem;
}

.info-item label { color: var(--vp-c-text-2); }
.info-item span { font-weight: 600; font-family: monospace; }
.pos { color: #10b981; }
.neg { color: #ef4444; }

.controls-panel {
  margin-top: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.action-buttons {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
}

.middle-row {
  display: flex;
  gap: 0.5rem;
}

button {
  padding: 0.5rem 1rem;
  background: var(--vp-c-bg-mute);
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

button:hover:not(:disabled) {
  background: var(--vp-c-brand-soft);
  border-color: var(--vp-c-brand-1);
}

button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-secondary {
  width: 100%;
  font-size: 0.8rem;
}

@media (max-width: 640px) {
  .mdp-container { flex-direction: column; align-items: center; }
}
</style>
