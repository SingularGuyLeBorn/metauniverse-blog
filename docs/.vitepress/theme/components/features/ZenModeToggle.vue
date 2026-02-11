<script setup lang="ts">
import { useLayoutStore } from '../../stores/layout'
import { onMounted, onUnmounted } from 'vue'

const store = useLayoutStore()

const handleKeydown = (e: KeyboardEvent) => {
  if (e.key === 'Escape' && store.zenMode) {
    store.toggleZenMode()
  }
}

onMounted(() => {
  window.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeydown)
})
</script>

<template>
  <Transition name="fade">
    <button 
      v-if="store.zenMode" 
      class="zen-exit-btn" 
      @click="store.toggleZenMode()"
      title="Exit Zen Mode (Esc)"
    >
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 14h6v6"/><path d="M20 10h-6V4"/><path d="M14 10l7-7"/><path d="M10 14L3 21"/></svg>
      <span>Exit Zen Mode</span>
    </button>
  </Transition>
</template>

<style scoped>
.zen-exit-btn {
  position: fixed;
  bottom: 32px;
  right: 32px;
  z-index: 9999;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 20px;
  background: var(--vp-c-brand);
  color: white;
  border: none;
  border-radius: 30px;
  font-weight: 500;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(0,0,0,0.2);
  transition: all 0.3s ease;
  backdrop-filter: blur(10px);
}

.zen-exit-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(0,0,0,0.3);
  background: var(--vp-c-brand-dark);
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease, transform 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
  transform: translateY(20px);
}
</style>
