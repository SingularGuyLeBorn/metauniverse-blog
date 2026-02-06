<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'

// ============ Types ============
interface LayoutState {
  leftWidth: number
  rightWidth: number
  leftVisible: boolean
  rightVisible: boolean
  contentWidth: 'narrow' | 'half' | 'full'
}

// ============ State ============
const STORAGE_KEY = 'mu-layout-state'
const DEFAULT_LEFT_WIDTH = 280
const DEFAULT_RIGHT_WIDTH = 240
const MIN_WIDTH = 180
const MAX_LEFT_WIDTH = 400
const MAX_RIGHT_WIDTH = 320

const layoutState = ref<LayoutState>({
  leftWidth: DEFAULT_LEFT_WIDTH,
  rightWidth: DEFAULT_RIGHT_WIDTH,
  leftVisible: true,
  rightVisible: true,
  contentWidth: 'half'
})

// ============ Drag State ============
const isDraggingLeft = ref(false)
const isDraggingRight = ref(false)
const startX = ref(0)
const startWidth = ref(0)

// ============ Persistence ============
const loadState = () => {
  if (typeof window === 'undefined') return
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved) {
      const parsed = JSON.parse(saved)
      Object.assign(layoutState.value, parsed)
    }
  } catch (e) {
    console.warn('Failed to load layout state:', e)
  }
}

const saveState = () => {
  if (typeof window === 'undefined') return
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(layoutState.value))
  } catch (e) {
    console.warn('Failed to save layout state:', e)
  }
}

// Watch for changes and save
watch(layoutState, saveState, { deep: true })

// ============ Drag Handlers ============
const startDragLeft = (e: MouseEvent) => {
  isDraggingLeft.value = true
  startX.value = e.clientX
  startWidth.value = layoutState.value.leftWidth
  document.body.style.cursor = 'col-resize'
  document.body.style.userSelect = 'none'
}

const startDragRight = (e: MouseEvent) => {
  isDraggingRight.value = true
  startX.value = e.clientX
  startWidth.value = layoutState.value.rightWidth
  document.body.style.cursor = 'col-resize'
  document.body.style.userSelect = 'none'
}

const onMouseMove = (e: MouseEvent) => {
  if (isDraggingLeft.value) {
    const delta = e.clientX - startX.value
    const newWidth = Math.min(MAX_LEFT_WIDTH, Math.max(MIN_WIDTH, startWidth.value + delta))
    layoutState.value.leftWidth = newWidth
  } else if (isDraggingRight.value) {
    const delta = startX.value - e.clientX
    const newWidth = Math.min(MAX_RIGHT_WIDTH, Math.max(MIN_WIDTH, startWidth.value + delta))
    layoutState.value.rightWidth = newWidth
  }
}

const onMouseUp = () => {
  isDraggingLeft.value = false
  isDraggingRight.value = false
  document.body.style.cursor = ''
  document.body.style.userSelect = ''
}

// ============ Toggle Functions ============
const toggleLeft = () => {
  layoutState.value.leftVisible = !layoutState.value.leftVisible
}

const toggleRight = () => {
  layoutState.value.rightVisible = !layoutState.value.rightVisible
}

const setContentWidth = (width: 'narrow' | 'half' | 'full') => {
  layoutState.value.contentWidth = width
}

// ============ Computed Styles ============
const leftStyle = computed(() => ({
  width: layoutState.value.leftVisible ? `${layoutState.value.leftWidth}px` : '0px',
  minWidth: layoutState.value.leftVisible ? `${MIN_WIDTH}px` : '0px',
  opacity: layoutState.value.leftVisible ? 1 : 0,
  overflow: layoutState.value.leftVisible ? 'auto' : 'hidden'
}))

const rightStyle = computed(() => ({
  width: layoutState.value.rightVisible ? `${layoutState.value.rightWidth}px` : '0px',
  minWidth: layoutState.value.rightVisible ? `${MIN_WIDTH}px` : '0px',
  opacity: layoutState.value.rightVisible ? 1 : 0,
  overflow: layoutState.value.rightVisible ? 'auto' : 'hidden'
}))

const contentMaxWidth = computed(() => {
  switch (layoutState.value.contentWidth) {
    case 'narrow': return '680px'
    case 'half': return '900px'
    case 'full': return '100%'
    default: return '900px'
  }
})

// ============ Lifecycle ============
onMounted(() => {
  loadState()
  document.addEventListener('mousemove', onMouseMove)
  document.addEventListener('mouseup', onMouseUp)
})

onUnmounted(() => {
  document.removeEventListener('mousemove', onMouseMove)
  document.removeEventListener('mouseup', onMouseUp)
})

// Expose for template
defineExpose({
  layoutState,
  toggleLeft,
  toggleRight,
  setContentWidth,
  startDragLeft,
  startDragRight,
  leftStyle,
  rightStyle,
  contentMaxWidth
})
</script>

<template>
  <div class="resizable-layout" :class="{ 'dragging': isDraggingLeft || isDraggingRight }">
    <!-- Toolbar -->
    <div class="layout-toolbar">
      <!-- Left toggle -->
      <button 
        class="toolbar-btn" 
        @click="toggleLeft" 
        :class="{ active: layoutState.leftVisible }"
        title="切换侧边栏"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="3" y="3" width="18" height="18" rx="2"/>
          <line x1="9" y1="3" x2="9" y2="21"/>
        </svg>
      </button>
      
      <!-- Content width selector -->
      <div class="width-selector">
        <button 
          class="width-btn" 
          :class="{ active: layoutState.contentWidth === 'narrow' }"
          @click="setContentWidth('narrow')"
          title="窄"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="8" y="4" width="8" height="16" rx="1"/>
          </svg>
        </button>
        <button 
          class="width-btn" 
          :class="{ active: layoutState.contentWidth === 'half' }"
          @click="setContentWidth('half')"
          title="半宽"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="5" y="4" width="14" height="16" rx="1"/>
          </svg>
        </button>
        <button 
          class="width-btn" 
          :class="{ active: layoutState.contentWidth === 'full' }"
          @click="setContentWidth('full')"
          title="全宽"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="2" y="4" width="20" height="16" rx="1"/>
          </svg>
        </button>
      </div>
      
      <!-- Right toggle -->
      <button 
        class="toolbar-btn" 
        @click="toggleRight" 
        :class="{ active: layoutState.rightVisible }"
        title="切换大纲"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="3" y="3" width="18" height="18" rx="2"/>
          <line x1="15" y1="3" x2="15" y2="21"/>
        </svg>
      </button>
    </div>

    <!-- Main Layout -->
    <div class="layout-container">
      <!-- Left Sidebar -->
      <aside class="layout-left" :style="leftStyle">
        <slot name="left">
          <!-- Default: VitePress sidebar will be here -->
        </slot>
      </aside>
      
      <!-- Left Resizer -->
      <div 
        v-if="layoutState.leftVisible"
        class="resizer resizer-left" 
        @mousedown="startDragLeft"
      >
        <div class="resizer-handle"></div>
      </div>
      
      <!-- Content -->
      <main class="layout-content" :style="{ maxWidth: contentMaxWidth }">
        <slot name="content">
          <!-- Default content -->
        </slot>
      </main>
      
      <!-- Right Resizer -->
      <div 
        v-if="layoutState.rightVisible"
        class="resizer resizer-right" 
        @mousedown="startDragRight"
      >
        <div class="resizer-handle"></div>
      </div>
      
      <!-- Right Sidebar (Outline) -->
      <aside class="layout-right" :style="rightStyle">
        <slot name="right">
          <!-- Default: VitePress outline will be here -->
        </slot>
      </aside>
    </div>
  </div>
</template>

<style scoped>
.resizable-layout {
  display: flex;
  flex-direction: column;
  height: 100%;
  width: 100%;
}

.resizable-layout.dragging {
  user-select: none;
}

/* Toolbar */
.layout-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 16px;
  background: var(--vp-c-bg);
  border-bottom: 1px solid var(--vp-c-divider);
  position: sticky;
  top: 0;
  z-index: 100;
}

.toolbar-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border: none;
  background: transparent;
  border-radius: 6px;
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition: all 0.2s;
}

.toolbar-btn:hover {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
}

.toolbar-btn.active {
  color: var(--vp-c-brand-1);
}

.width-selector {
  display: flex;
  gap: 4px;
  padding: 4px;
  background: var(--vp-c-bg-soft);
  border-radius: 8px;
}

.width-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 24px;
  border: none;
  background: transparent;
  border-radius: 4px;
  color: var(--vp-c-text-3);
  cursor: pointer;
  transition: all 0.15s;
}

.width-btn:hover {
  color: var(--vp-c-text-2);
}

.width-btn.active {
  background: var(--vp-c-bg);
  color: var(--vp-c-brand-1);
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

/* Layout Container */
.layout-container {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.layout-left,
.layout-right {
  flex-shrink: 0;
  transition: width 0.2s ease, opacity 0.2s ease;
  background: var(--vp-c-bg);
  border-color: var(--vp-c-divider);
}

.layout-left {
  border-right: 1px solid var(--vp-c-divider);
}

.layout-right {
  border-left: 1px solid var(--vp-c-divider);
}

.layout-content {
  flex: 1;
  min-width: 0;
  overflow: auto;
  margin: 0 auto;
  padding: 24px;
  transition: max-width 0.2s ease;
}

/* Resizer */
.resizer {
  width: 8px;
  cursor: col-resize;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  background: transparent;
  transition: background 0.15s;
}

.resizer:hover,
.resizer:active {
  background: var(--vp-c-brand-soft);
}

.resizer-handle {
  width: 3px;
  height: 40px;
  background: var(--vp-c-divider);
  border-radius: 2px;
  transition: all 0.15s;
}

.resizer:hover .resizer-handle,
.resizer:active .resizer-handle {
  background: var(--vp-c-brand-1);
  height: 60px;
}

/* Responsive */
@media (max-width: 768px) {
  .layout-toolbar {
    padding: 6px 12px;
  }
  
  .layout-left,
  .layout-right {
    position: absolute;
    z-index: 50;
    height: 100%;
  }
  
  .layout-left {
    left: 0;
  }
  
  .layout-right {
    right: 0;
  }
  
  .resizer {
    display: none;
  }
}
</style>
