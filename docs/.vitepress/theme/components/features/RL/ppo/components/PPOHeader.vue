<template>
  <header class="scholar-header">
    <div class="header-row-top">
      <!-- Title & Logo -->
      <div class="header-left">
        <div class="logo-badge">PPO</div>
        <div class="title-group">
          <h1 class="main-title">Proximal Policy Optimization</h1>
          <div class="meta-info">
            <span class="act-badge">Act {{ act }}</span>
            <span class="divider">|</span>
            <span class="frame-counter">Frame {{ currentFrame }} / {{ totalFrames }}</span>
          </div>
        </div>
      </div>

      <!-- Controls (Right aligned on top row) -->
      <div class="header-controls">
        <div class="view-toggle">
          <button 
            class="toggle-btn" 
            :class="{ active: viewMode === 'visual' }"
            @click="$emit('toggle-view', 'visual')"
          >Visual</button>
          <button 
            class="toggle-btn" 
            :class="{ active: viewMode === 'code' }"
            @click="$emit('toggle-view', 'code')"
          >Code</button>
        </div>
        
        <div class="divider-v"></div>

        <div class="playback-controls">
           <div class="speed-control">
            <span class="label">Speed</span>
            <input 
              type="range" 
              :value="playSpeed" 
              @input="$emit('update:speed', parseFloat(($event.target as HTMLInputElement).value))"
              min="0.5" max="5" step="0.5" 
              class="speed-slider"
            >
            <span class="speed-val">{{ playSpeed }}x</span>
          </div>
          
          <div class="btn-group">
            <button @click="$emit('reset')" class="btn-icon" title="Reset">↺</button>
            <button @click="$emit('prev')" class="btn-icon" :disabled="currentFrame === 1">←</button>
            <button @click="$emit('toggle-play')" class="btn-play">
              {{ isPlaying ? '❚❚' : '▶' }}
            </button>
            <button @click="$emit('next')" class="btn-icon" :disabled="currentFrame === totalFrames">→</button>
          </div>
        </div>
      </div>
    </div>

    <div class="header-row-bottom">
      <!-- Timeline (Full width on bottom row) -->
       <div class="timeline-track">
        <div class="timeline-progress" :style="{ width: progress + '%' }"></div>
        <div class="milestone" 
             v-for="m in 4" 
             :key="m"
             :class="{ active: act >= m, current: act === m }">
          {{ ['Init', 'Interaction (Step 1)', 'Interaction (Step 2)', 'Training'][m-1] }}
        </div>
      </div>
       <div class="phase-display">{{ phase }}</div>
    </div>
  </header>
</template>

<script setup lang="ts">
defineProps<{
  currentFrame: number;
  totalFrames: number;
  act: number;
  phase: string;
  progress: number;
  isPlaying: boolean;
  playSpeed: number;
  viewMode?: 'visual' | 'code';
}>();

const emit = defineEmits<{
  (e: 'toggle-play'): void;
  (e: 'next'): void;
  (e: 'prev'): void;
  (e: 'reset'): void;
  (e: 'update:speed', val: number): void;
  (e: 'toggle-view', mode: 'visual' | 'code'): void;
}>();
</script>

<style scoped>
.scholar-header {
  background: #ffffff;
  border-bottom: 1px solid #e2e8f0;
  display: flex;
  flex-direction: column;
  padding: 16px 24px;
  gap: 16px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

/* Run 1: Top Row */
.header-row-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 16px;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.logo-badge {
  width: 40px;
  height: 40px;
  background: linear-gradient(135deg, #3b82f6, #2563eb);
  color: white;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  font-size: 0.9rem;
}

.main-title {
  font-size: 1.1rem;
  font-weight: 700;
  color: #0f172a;
  margin: 0;
  line-height: 1.2;
}

.meta-info {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.8rem;
  color: #64748b;
  margin-top: 4px;
}

.act-badge {
  background: #eff6ff;
  color: #3b82f6;
  padding: 2px 8px;
  border-radius: 4px;
  font-weight: 600;
}

.divider { color: #cbd5e1; }

.header-controls {
  display: flex;
  align-items: center;
  gap: 16px;
}

.view-toggle {
  display: flex;
  background: #f1f5f9;
  padding: 4px;
  border-radius: 8px;
  gap: 4px;
}

.toggle-btn {
  padding: 6px 16px;
  border: none;
  background: transparent;
  color: #64748b;
  font-size: 0.85rem;
  font-weight: 600;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

.toggle-btn.active {
  background: white;
  color: #3b82f6;
  box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

.divider-v {
  width: 1px;
  height: 32px;
  background: #e2e8f0;
}

.playback-controls {
  display: flex;
  align-items: center;
  gap: 16px;
}

.speed-control {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.8rem;
  color: #64748b;
  background: #f8fafc;
  padding: 4px 12px;
  border-radius: 20px;
}

.speed-slider {
  width: 80px;
  height: 4px;
  -webkit-appearance: none;
  background: #e2e8f0;
  border-radius: 2px;
  outline: none;
}

.speed-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 12px;
  height: 12px;
  background: #3b82f6;
  border-radius: 50%;
  cursor: pointer;
}

.btn-group {
  display: flex;
  gap: 8px;
}

.btn-icon, .btn-play {
  width: 36px;
  height: 36px;
  border: 1px solid #e2e8f0;
  background: white;
  border-radius: 8px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.1rem;
  color: #475569;
  transition: all 0.2s;
}

.btn-play {
  background: #3b82f6;
  border-color: #3b82f6;
  color: white;
}

.btn-icon:hover:not(:disabled) {
  background: #f8fafc;
  color: #1e293b;
}

.btn-play:hover {
  background: #2563eb;
}

/* Row 2: Bottom */
.header-row-bottom {
  display: flex;
  align-items: center;
  gap: 24px;
}

.timeline-track {
  flex: 1;
  height: 8px;
  background: #f1f5f9;
  border-radius: 4px;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 10%; /* Spacing for milestones */
}

.timeline-progress {
  position: absolute;
  left: 0;
  top: 0;
  height: 100%;
  background: linear-gradient(90deg, #60a5fa, #3b82f6);
  border-radius: 4px;
  transition: width 0.3s ease-out;
}

.milestone {
  position: relative;
  z-index: 2;
  font-size: 0.75rem;
  color: #94a3b8;
  padding: 4px 12px;
  background: white;
  border-radius: 12px;
  border: 1px solid #e2e8f0;
  font-weight: 500;
  transition: all 0.3s;
}

.milestone.active {
  color: #3b82f6;
  border-color: #bfdbfe;
  background: #eff6ff;
}

.milestone.current {
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
  font-weight: 700;
  transform: scale(1.1);
}

.phase-display {
  font-size: 0.9rem;
  font-weight: 600;
  color: #475569;
  background: #f8fafc;
  padding: 4px 12px;
  border-radius: 6px;
  border: 1px solid #e2e8f0;
  white-space: nowrap;
}
</style>
