<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'

interface GridCell {
  x: number
  y: number
  type: 'empty' | 'wall' | 'start' | 'goal' | 'trap'
  value?: number // V(s) or max Q(s,a)
  qValues?: Record<string, number> // { up: 0, down: 0, ... }
  reward?: number
}

interface AgentState {
  x: number
  y: number
  action?: string
  prevX?: number
  prevY?: number
}

interface StepData {
  step: number
  agent: AgentState
  gridUpdates?: { x: number, y: number, qValues: Record<string, number>, value?: number }[]
  description: string
  formula?: string
  variables?: Record<string, string | number>
  highlightLine?: number
}

interface VisualizationData {
  gridSize: [number, number] // [rows, cols]
  gridLayout: GridCell[]
  steps: StepData[]
  code?: string
}

const props = defineProps<{
  data: VisualizationData
  title?: string
}>()

const currentStepIndex = ref(0)
const isPlaying = ref(false)
const playbackSpeed = ref(1000) // ms per step
let timer: any = null

const currentStepData = computed(() => {
  return props.data.steps[currentStepIndex.value] || props.data.steps[0]
})

const gridState = computed(() => {
  // Reconstruct grid state up to current step
  // Base layout
  const grid = JSON.parse(JSON.stringify(props.data.gridLayout)) as GridCell[]
  const gridMap = new Map<string, GridCell>()
  grid.forEach(cell => gridMap.set(`${cell.x},${cell.y}`, cell))

  // Apply updates
  for (let i = 0; i <= currentStepIndex.value; i++) {
    const step = props.data.steps[i]
    if (step.gridUpdates) {
      step.gridUpdates.forEach(update => {
        const cell = gridMap.get(`${update.x},${update.y}`)
        if (cell) {
          if (update.qValues) cell.qValues = { ...cell.qValues, ...update.qValues }
          if (update.value !== undefined) cell.value = update.value
        }
      })
    }
  }
  return Array.from(gridMap.values())
})

const agentPosition = computed(() => currentStepData.value.agent)

// Playback Control
const togglePlay = () => {
  isPlaying.value = !isPlaying.value
  if (isPlaying.value) {
    play()
  } else {
    pause()
  }
}

const play = () => {
  if (currentStepIndex.value >= props.data.steps.length - 1) {
    currentStepIndex.value = 0
  }
  timer = setInterval(() => {
    if (currentStepIndex.value < props.data.steps.length - 1) {
      currentStepIndex.value++
    } else {
      pause()
      isPlaying.value = false
    }
  }, playbackSpeed.value)
}

const pause = () => {
  if (timer) clearInterval(timer)
}

const nextStep = () => {
  pause()
  isPlaying.value = false
  if (currentStepIndex.value < props.data.steps.length - 1) currentStepIndex.value++
}

const prevStep = () => {
  pause()
  isPlaying.value = false
  if (currentStepIndex.value > 0) currentStepIndex.value--
}

const reset = () => {
  pause()
  isPlaying.value = false
  currentStepIndex.value = 0
}

watch(playbackSpeed, () => {
  if (isPlaying.value) {
    pause()
    play()
  }
})

onUnmounted(() => {
  pause()
})

// Formatting
const formatNumber = (num: number) => num?.toFixed(2)

// Styles helper
const getCellColor = (cell: GridCell) => {
  switch (cell.type) {
    case 'wall': return 'bg-gray-800'
    case 'start': return 'bg-blue-900/30'
    case 'goal': return 'bg-green-900/30'
    case 'trap': return 'bg-red-900/30'
    default: return 'bg-gray-100 dark:bg-gray-800'
  }
}

const getHeatmapColor = (value: number) => {
  if (value === undefined) return 'transparent'
  // Simple heatmap: -1 (red) to 1 (green)
  const opacity = Math.min(Math.abs(value), 1) * 0.5
  return value > 0 
    ? `rgba(0, 255, 0, ${opacity})`
    : `rgba(255, 0, 0, ${opacity})`
}

</script>

<template>
  <div class="algorithm-visualizer border rounded-lg overflow-hidden bg-white dark:bg-[#1e1e1e] border-gray-200 dark:border-gray-700 my-8 shadow-lg">
    <!-- Header -->
    <div class="header p-4 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-[#252526] flex justify-between items-center">
      <h3 class="font-bold text-lg m-0">{{ title || 'Algorithm Visualization' }}</h3>
      <div class="controls flex items-center gap-2">
         <span class="text-xs text-gray-500">Step {{ currentStepIndex + 1 }} / {{ data.steps.length }}</span>
         <button @click="reset" class="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded" title="Reset">
           ⏮
         </button>
         <button @click="prevStep" class="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded" title="Previous">
           ◀
         </button>
         <button @click="togglePlay" class="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded w-8" :title="isPlaying ? 'Pause' : 'Play'">
           {{ isPlaying ? '⏸' : '▶' }}
         </button>
         <button @click="nextStep" class="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded" title="Next">
           ▶
         </button>
         <select v-model="playbackSpeed" class="text-xs bg-transparent border border-gray-300 dark:border-gray-600 rounded ml-2 p-1">
           <option :value="2000">Slow</option>
           <option :value="1000">Normal</option>
           <option :value="500">Fast</option>
           <option :value="100">Hyper</option>
         </select>
      </div>
    </div>

    <div class="content grid grid-cols-1 lg:grid-cols-2 gap-0">
      <!-- Left: Visual Grid -->
      <div class="visual-panel p-6 flex justify-center items-center bg-gray-50 dark:bg-[#1e1e1e] border-b lg:border-b-0 lg:border-r border-gray-200 dark:border-gray-700 min-h-[300px]">
        
        <div class="grid-container relative" :style="{ 
          display: 'grid', 
          gridTemplateColumns: `repeat(${data.gridSize[1]}, 60px)`,
          gridTemplateRows: `repeat(${data.gridSize[0]}, 60px)`,
          gap: '4px'
        }">
          <div v-for="(cell, idx) in gridState" :key="idx"
            class="cell relative w-[60px] h-[60px] border border-gray-200 dark:border-gray-600 rounded flex justify-center items-center text-xs"
            :class="getCellColor(cell)"
          >
            <!-- Heatmap overlay -->
            <div class="absolute inset-0 z-0" :style="{ backgroundColor: getHeatmapColor(cell.value || 0) }"></div>

            <!-- Content -->
            <div class="z-10 relative text-center">
              <div v-if="cell.type === 'start'">S</div>
              <div v-if="cell.type === 'goal'">G</div>
              <div v-if="cell.type === 'trap'">T</div>
              <div v-if="cell.value !== undefined" class="font-mono font-bold">{{ formatNumber(cell.value) }}</div>
            </div>

            <!-- Q-Values (arrows/triangles) -->
            <div v-if="cell.qValues" class="absolute inset-0 z-0 opacity-50">
               <!-- Simple visualization for Q-values could go here, e.g. triangles -->
            </div>

             <!-- Agent -->
            <div v-if="agentPosition.x === cell.x && agentPosition.y === cell.y" 
              class="absolute inset-0 z-20 flex justify-center items-center">
              <div class="w-8 h-8 bg-blue-500 rounded-full shadow-lg border-2 border-white flex justify-center items-center text-white text-lg transition-all duration-300 transform scale-110">
                🤖
              </div>
            </div>
          </div>
        </div>

      </div>

      <!-- Right: Info & Math -->
      <div class="info-panel flex flex-col h-full">
        <!-- Step Description -->
        <div class="p-4 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-[#252526]">
          <h4 class="font-bold mb-2 text-sm text-gray-500 uppercase tracking-wider">Current Action</h4>
          <p class="text-base">{{ currentStepData.description }}</p>
        </div>

        <!-- Math / Variables -->
        <div class="p-4 flex-1 bg-gray-50 dark:bg-[#1e1e1e] overflow-y-auto">
          <h4 class="font-bold mb-2 text-sm text-gray-500 uppercase tracking-wider">Calculation</h4>
          
          <div v-if="currentStepData.formula" class="math-block bg-white dark:bg-black p-4 rounded border border-gray-200 dark:border-gray-700 mb-4 font-mono text-sm shadow-sm">
             {{ currentStepData.formula }}
          </div>

          <div v-if="currentStepData.variables" class="variables grid grid-cols-2 gap-2 text-sm">
            <div v-for="(val, key) in currentStepData.variables" :key="key" class="flex justify-between items-center bg-white dark:bg-[#2d2d2d] p-2 rounded border border-gray-200 dark:border-gray-600">
               <span class="text-gray-500 font-mono">{{ key }}</span>
               <span class="font-bold font-mono text-blue-600 dark:text-blue-400">{{ val }}</span>
            </div>
          </div>
        </div>

      </div>
    </div>
  </div>
</template>

<style scoped>
.cell {
  transition: all 0.3s ease;
}
</style>
