<template>
  <div class="rl-visualizer glass-card">
    <div class="visualizer-header">
      <h4><i class="i-carbon-analytics"></i> {{ title }}</h4>
      <div class="visualizer-controls">
        <slot name="controls"></slot>
      </div>
    </div>
    
    <div class="visualizer-body" :class="{ 'is-loading': loading }">
      <slot></slot>
    </div>

    <div class="visualizer-footer" v-if="$slots.footer">
      <slot name="footer"></slot>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
  loading?: boolean
}>()
</script>

<style scoped>
.rl-visualizer {
  margin: 2rem 0;
  padding: 1.5rem;
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  overflow: hidden;
}

.dark .rl-visualizer {
  background: rgba(15, 23, 42, 0.3);
}

.visualizer-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
  border-bottom: 1px solid var(--vp-c-divider);
  padding-bottom: 0.75rem;
}

.visualizer-header h4 {
  margin: 0;
  font-size: 1.1rem;
  color: var(--vp-c-brand-1);
}

.visualizer-body {
  position: relative;
  min-height: 200px;
  display: flex;
  justify-content: center;
  align-items: center;
}

.is-loading {
  opacity: 0.5;
  pointer-events: none;
}

.visualizer-footer {
  margin-top: 1.5rem;
  font-size: 0.9rem;
  color: var(--vp-c-text-2);
  line-height: 1.6;
}
</style>
