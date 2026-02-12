<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'

const currentFrame = ref(1)
const totalFrames = 3000 // Extended for cinematic pacing (approx 50-60s at 60fps)
const playSpeed = ref(1)
const isPlaying = ref(false)
const viewMode = ref('visual')

// Scene-Based Phases (1-9)
const phases = [
  { id: 'SCENE1_Init', name: 'Scene 1: Environment Breathe', frames: 200 },
  { id: 'SCENE2_StateSpawn', name: 'Scene 2: State Extraction', frames: 150 },
  { id: 'SCENE3_Split', name: 'Scene 3: Dual Path Split', frames: 100 },
  { id: 'SCENE4_Critic1', name: 'Scene 4: Critic Value V(st)', frames: 250 },
  { id: 'SCENE5_Actor', name: 'Scene 5: Actor Policy', frames: 300 },
  { id: 'SCENE6_ActionForm', name: 'Scene 6: Action Assembly', frames: 150 }, // Distinct phase for clarity
  { id: 'SCENE7_Interact', name: 'Scene 7: Interaction & Reward', frames: 300 },
  { id: 'SCENE8_Critic2', name: 'Scene 8: Memory & V(st+1)', frames: 400 },
  { id: 'SCENE9_Math', name: 'Scene 9: GAE & Update', frames: 500 }
]

// --- Computed Helpers ---
const currentPhaseIndex = computed(() => {
  let count = 0
  for (let i = 0; i < phases.length; i++) {
    count += phases[i].frames
    if (currentFrame.value <= count) return i
  }
  return phases.length - 1
})
const currentPhase = computed(() => phases[currentPhaseIndex.value])
const phase = computed(() => currentPhase.value.id)
const act = computed(() => {
    // Mapping phases to simplified "Steps" for the UI Header
    const i = currentPhaseIndex.value
    if (i === 0) return 0 // Env
    if (i < 3) return 1 // Obs
    if (i < 6) return 2 // Decision
    if (i < 7) return 3 // Action
    if (i < 8) return 4 // Store
    return 5 // Train
})
const progress = computed(() => (currentFrame.value / totalFrames) * 100)

const phaseProgress = computed(() => {
  let framesBefore = 0
  for (let i = 0; i < currentPhaseIndex.value; i++) {
    framesBefore += phases[i].frames
  }
  const localFrame = currentFrame.value - framesBefore
  return Math.max(0, Math.min(1, localFrame / currentPhase.value.frames))
})

// Buffer Logic (Visual only)
const bufferTuples = ref([])
watch(phase, (newPhase) => {
  if (newPhase === 'SCENE8_Critic2' && bufferTuples.value.length === 0) {
    // Add "Ghost" tuple that fills in during the scene
    bufferTuples.value.push({ s: 'sₜ', a: 'aₜ', r: '...', s_next: '...', val: '...' })
  }
  if (newPhase === 'SCENE1_Init') bufferTuples.value = []
})

const animState = computed(() => ({
  phase: phase.value,
  progress: phaseProgress.value,
  tuples: bufferTuples.value
}))

// Info Mapping
const infoData = computed(() => {
  const map = {
    SCENE1_Init: { title: '环境初始化', formula: 'Env \\text{ Ready}', desc: '环境网格呼吸，等待智能体观测' },
    SCENE2_StateSpawn: { title: '状态提取', formula: 's_t \\in \\mathbb{R}^4', desc: '从环境格点提取状态向量 s_t' },
    SCENE3_Split: { title: '双流处理', formula: 's_t \\to \\{ \\pi, V \\}', desc: '状态向量复制，分别送往 Actor 和 Critic' },
    SCENE4_Critic1: { title: '价值评估 (V1)', formula: 'V(s_t) \\approx 12.5', desc: 'Critic 网络评估当前状态价值 V(s_t)' },
    SCENE5_Actor: { title: '策略计算', formula: '\\pi_\\theta(a|s_t)', desc: 'Actor 网络计算动作概率分布' },
    SCENE6_ActionForm: { title: '动作生成', formula: 'a_t \\sim \\pi(\\cdot)', desc: '采样动作并进行 One-hot 编码' },
    SCENE7_Interact: { title: '环境交互', formula: 's_{t+1}, r_t \\leftarrow \\text{Step}(a_t)', desc: '执行动作，获得奖励和新状态' },
    SCENE8_Critic2: { title: '价值评估 (V2)', formula: 'V(s_{t+1}) \\approx 15.2', desc: 'Critic 评估下一状态价值 (用于计算优势)' },
    SCENE9_Math: { title: 'PPO 更新', formula: '\\theta_{new} \\leftarrow \\theta + \\alpha \\nabla J', desc: '计算 GAE 优势，Clip Loss 并更新网络' }
  }
  return map[phase.value] || map.SCENE1_Init
})

// Loop
let animationId = null
const startAnimation = () => {
  const step = () => {
    if (currentFrame.value < totalFrames) {
      currentFrame.value += 1
      animationId = requestAnimationFrame(step) // Use RAF for smoother look
    } else {
      currentFrame.value = 1
      animationId = requestAnimationFrame(step)
    }
  }
  animationId = requestAnimationFrame(step)
}
const stopAnimation = () => { if (animationId) cancelAnimationFrame(animationId) }

watch(isPlaying, (val) => { if (val) startAnimation(); else stopAnimation() })
watch(playSpeed, () => { 
    // Speed not directly used in RAF loop logic above for simplicity, 
    // but could be: currentFrame += playSpeed.value
    // For now purely relying on RAF (60fps)
})
onUnmounted(() => stopAnimation())

// Controls
const togglePlay = () => isPlaying.value = !isPlaying.value
const reset = () => { currentFrame.value = 1; isPlaying.value = false; bufferTuples.value = [] }
const next = () => { 
    // Jump to next phase
    let count = 0
    for (let i = 0; i <= currentPhaseIndex.value; i++) count += phases[i].frames
    currentFrame.value = Math.min(count + 1, totalFrames)
} 
const prev = () => {
    let count = 0
    for (let i = 0; i < currentPhaseIndex.value; i++) count += phases[i].frames
    currentFrame.value = Math.max(count, 1)
}
</script>

# PPO Cinematic Experience

<PPOHeader
  :currentFrame="currentFrame"
  :totalFrames="totalFrames"
  :act="act"
  :phase="currentPhase.name"
  :progress="progress"
  :isPlaying="isPlaying"
  :playSpeed="playSpeed"
  :viewMode="viewMode"
  @toggle-play="togglePlay"
  @next="next"
  @prev="prev"
  @reset="reset"
  @update:speed="s => playSpeed = s"
  @toggle-view="m => viewMode = m"
/>

<div class="ppo-container">
  <div class="viz-area">
      <PPODependencyGraph 
        v-if="viewMode === 'visual'" 
        :animState="animState" 
      />
      <PPOCodeTrace 
        v-else 
        :phase="phase" 
      />
  </div>
  
  <div class="info-area">
    <PPOInfo 
      :title="infoData.title"
      :formula="infoData.formula"
      :desc="infoData.desc"
    />
  </div>
</div>

<style>
.ppo-container {
  margin-top: 20px;
  display: flex;
  flex-direction: column;
  gap: 24px;
  max-width: 100%;
  margin-left: auto;
  margin-right: auto;
}

.viz-area {
  background: #f8fafc; /* User requested F8FAFC */
  border-radius: 24px;
  padding: 0; /* Remove padding to let SVG fill */
  border: 1px solid #e2e8f0;
  box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.05); /* Premium shadow */
  aspect-ratio: 16/9; /* 1920x1080 prop */
  width: 100%;
  position: relative;
  overflow: hidden;
}

.info-area {
  z-index: 10;
}
</style>
