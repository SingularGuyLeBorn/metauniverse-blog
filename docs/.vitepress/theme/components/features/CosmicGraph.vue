<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import { data } from '../../utils/knowledge.data'
import { useRouter } from 'vitepress'

const chartContainer = ref<HTMLElement | null>(null)
let visible = ref(true)

const router = useRouter()

onMounted(async () => {
  if (typeof window === 'undefined') return

  // Dynamic import to avoid SSR issues
  const ForceGraph3D = (await import('3d-force-graph')).default as any

  const Graph = ForceGraph3D()
    (chartContainer.value!)
    .graphData(data)
    .nodeLabel('name')
    .nodeAutoColorBy('group')
    .linkWidth(1)
    .linkOpacity(0.5)
    .nodeVal('val')
    .onNodeClick((node: any) => {
      if (node.link) {
        router.go(node.link)
      }
    })
    
  // Config
  Graph.backgroundColor('rgba(0,0,0,0)') // Transparent
  Graph.showNavInfo(false)
  
  // Custom Node Object (Optional: Star texture?)
  // For now simple dots are elegant enough
  
  // Responsive
  const resizeObserver = new ResizeObserver(() => {
    if (!chartContainer.value) return
    const { width, height } = chartContainer.value.getBoundingClientRect()
    Graph.width(width)
    Graph.height(height)
  })
  
  resizeObserver.observe(chartContainer.value!)
})
</script>

<template>
  <div class="cosmic-graph-container" ref="chartContainer">
    <!-- Graph rendered here -->
    <div class="graph-overlay">
      <h3>Cosmic Knowledge Graph</h3>
      <p>{{ data.nodes.length }} Nodes · {{ data.links.length }} Connections</p>
    </div>
  </div>
</template>

<style scoped>
.cosmic-graph-container {
  width: 100%;
  height: 600px; /* Default height */
  position: relative;
  background: radial-gradient(circle at center, var(--vp-c-bg-alt), var(--vp-c-bg));
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid var(--vp-c-divider);
}

.graph-overlay {
  position: absolute;
  top: 20px;
  left: 20px;
  pointer-events: none;
  z-index: 10;
}

.graph-overlay h3 {
  margin: 0;
  font-size: 1.2rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.graph-overlay p {
  margin: 4px 0 0;
  font-size: 0.9rem;
  color: var(--vp-c-text-2);
}
</style>
