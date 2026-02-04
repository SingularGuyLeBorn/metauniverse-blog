<template>
  <div class="scroll-progress" :style="{ width: `${progress}%` }"></div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

const progress = ref(0)

const updateProgress = () => {
  const scrollTop = window.scrollY
  const docHeight = document.documentElement.scrollHeight
  const winHeight = window.innerHeight
  const scrollPercent = scrollTop / (docHeight - winHeight)
  progress.value = Math.min(100, Math.max(0, scrollPercent * 100))
}

onMounted(() => {
  window.addEventListener('scroll', updateProgress, { passive: true })
})

onUnmounted(() => {
  window.removeEventListener('scroll', updateProgress)
})
</script>

<style scoped>
.scroll-progress {
  position: fixed;
  top: 0;
  left: 0;
  height: 3px;
  background: linear-gradient(90deg, 
    var(--vp-c-brand-1), 
    #d946ef, 
    var(--vp-c-brand-2)
  );
  z-index: 200;
  transition: width 0.1s ease-out;
  box-shadow: 0 0 10px rgba(var(--vp-c-brand-1-rgb), 0.5);
}
</style>
