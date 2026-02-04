<template>
  <div class="mermaid-container" v-if="svg" v-html="svg"></div>
  <div class="mermaid-loading" v-else>Loading diagram...</div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'

const props = defineProps<{
  code: string // URI encoded string
}>()

const svg = ref('')

onMounted(async () => {
  renderChart()
})

watch(() => props.code, () => {
  renderChart()
})

async function renderChart() {
  if (typeof window === 'undefined') return

  try {
    const { default: mermaid } = await import('mermaid')
    
    mermaid.initialize({
      startOnLoad: false,
      theme: document.documentElement.classList.contains('dark') ? 'dark' : 'default',
      securityLevel: 'loose', // Allow interactive scripts if needed
    })

    const decoded = decodeURIComponent(props.code)
    const id = `mermaid-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    
    const { svg: svgCode } = await mermaid.render(id, decoded)
    svg.value = svgCode
  } catch (error) {
    console.error('Mermaid render error:', error)
    svg.value = `<div class="mermaid-error">Error rendering chart: ${error}</div>`
  }
}
</script>

<style scoped>
.mermaid-container {
  display: flex;
  justify-content: center;
  margin: 2rem 0;
  overflow-x: auto;
}
.mermaid-loading, .mermaid-error {
  padding: 1rem;
  text-align: center;
  color: var(--vp-c-text-2);
}
.mermaid-error {
  color: var(--vp-c-danger-1);
}
</style>
