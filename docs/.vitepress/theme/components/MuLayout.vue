<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, provide, h } from 'vue'
import { useData } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import ScrollProgress from './features/ScrollProgress.vue'
import ModeSwitcher from './features/ModeSwitcher.vue'
import SemanticHeatmap from './features/SemanticHeatmap.vue'
import RelatedReferences from './features/RelatedReferences.vue'
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

// Provide layout state to child components
provide('layoutState', layoutState)

const { frontmatter, page } = useData()

// ============ Drag State ============
const isDraggingLeft = ref(false)
const isDraggingRight = ref(false)
const startX = ref(0)
const startWidth = ref(0)

// ============ Computed ============
const isDocPage = computed(() => {
  return page.value.relativePath && 
         !page.value.relativePath.endsWith('index.md') &&
         frontmatter.value.layout !== 'home' &&
         frontmatter.value.layout !== 'page'
})

const showToolbar = computed(() => isDocPage.value)

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
</script>

<template>
  <div 
    class="mu-layout-wrapper" 
    :class="{ 
      'dragging': isDraggingLeft || isDraggingRight,
      'has-toolbar': showToolbar,
      'left-hidden': !layoutState.leftVisible,
      'right-hidden': !layoutState.rightVisible
    }"
  >
    <!-- Layout Toolbar -->
    <Transition name="toolbar-fade">
      <div v-if="showToolbar" class="mu-layout-toolbar">
        <!-- Left toggle -->
        <button 
          class="mu-toolbar-btn" 
          @click="toggleLeft" 
          :class="{ active: layoutState.leftVisible }"
          title="切换侧边栏 (Ctrl+[)"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="3" width="18" height="18" rx="2"/>
            <line x1="9" y1="3" x2="9" y2="21"/>
          </svg>
        </button>
        
        <!-- Content width selector -->
        <div class="mu-width-selector">
          <button 
            class="mu-width-btn" 
            :class="{ active: layoutState.contentWidth === 'narrow' }"
            @click="setContentWidth('narrow')"
            title="窄"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="8" y="4" width="8" height="16" rx="1"/>
            </svg>
          </button>
          <button 
            class="mu-width-btn" 
            :class="{ active: layoutState.contentWidth === 'half' }"
            @click="setContentWidth('half')"
            title="半宽"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="5" y="4" width="14" height="16" rx="1"/>
            </svg>
          </button>
          <button 
            class="mu-width-btn" 
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
          class="mu-toolbar-btn" 
          @click="toggleRight" 
          :class="{ active: layoutState.rightVisible }"
          title="切换大纲 (Ctrl+])"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="3" width="18" height="18" rx="2"/>
            <line x1="15" y1="3" x2="15" y2="21"/>
          </svg>
        </button>
      </div>
    </Transition>

    <!-- Left Resizer (only when visible) -->
    <div 
      v-if="showToolbar && layoutState.leftVisible"
      class="mu-resizer mu-resizer-left" 
      @mousedown="startDragLeft"
      :style="{ left: `${layoutState.leftWidth}px` }"
    >
      <div class="mu-resizer-handle"></div>
    </div>
    
    <!-- Right Resizer (only when visible) -->
    <div 
      v-if="showToolbar && layoutState.rightVisible"
      class="mu-resizer mu-resizer-right" 
      @mousedown="startDragRight"
      :style="{ right: `${layoutState.rightWidth}px` }"
    >
      <div class="mu-resizer-handle"></div>
    </div>

    <!-- Scroll Progress -->
    <ScrollProgress />

    <!-- Default VitePress Layout -->
    <DefaultTheme.Layout>
      <!-- layout-bottom slot for mode switcher and heatmap -->
      <template #layout-bottom>
        <div id="mu-teleport-container">
          <ModeSwitcher />
          <SemanticHeatmap />
        </div>
      </template>
      
      <!-- doc-after slot for related references -->
      <template #doc-after>
        <RelatedReferences v-if="frontmatter.graph !== false" />
      </template>
    </DefaultTheme.Layout>
  </div>
</template>

<style>
/* ============ Layout Wrapper ============ */
.mu-layout-wrapper {
  position: relative;
}

.mu-layout-wrapper.has-toolbar {
  --mu-toolbar-height: 40px;
}

.mu-layout-wrapper.dragging {
  user-select: none;
}

/* ============ Toolbar ============ */
.mu-layout-toolbar {
  position: fixed;
  top: var(--vp-nav-height, 64px);
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
  z-index: 100;
  backdrop-filter: blur(8px);
}

.toolbar-fade-enter-active,
.toolbar-fade-leave-active {
  transition: all 0.2s ease;
}

.toolbar-fade-enter-from,
.toolbar-fade-leave-to {
  opacity: 0;
  transform: translateX(-50%) translateY(-10px);
}

.mu-toolbar-btn {
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

.mu-toolbar-btn:hover {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
}

.mu-toolbar-btn.active {
  color: var(--vp-c-brand-1);
}

.mu-width-selector {
  display: flex;
  gap: 2px;
  padding: 3px;
  background: var(--vp-c-bg-soft);
  border-radius: 8px;
}

.mu-width-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 24px;
  border: none;
  background: transparent;
  border-radius: 5px;
  color: var(--vp-c-text-3);
  cursor: pointer;
  transition: all 0.15s;
}

.mu-width-btn:hover {
  color: var(--vp-c-text-2);
}

.mu-width-btn.active {
  background: var(--vp-c-bg);
  color: var(--vp-c-brand-1);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

/* ============ Resizers ============ */
.mu-resizer {
  position: fixed;
  top: calc(var(--vp-nav-height, 64px) + 20px);
  bottom: 0;
  width: 8px;
  cursor: col-resize;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 50;
  background: transparent;
  transition: background 0.15s;
}

.mu-resizer:hover,
.mu-resizer:active {
  background: var(--vp-c-brand-soft);
}

.mu-resizer-handle {
  width: 3px;
  height: 40px;
  background: var(--vp-c-divider);
  border-radius: 2px;
  transition: all 0.15s;
}

.mu-resizer:hover .mu-resizer-handle,
.mu-resizer:active .mu-resizer-handle {
  background: var(--vp-c-brand-1);
  height: 60px;
}

/* ============ Dynamic Sidebar Widths ============ */
.mu-layout-wrapper.has-toolbar .VPSidebar {
  width: v-bind('`${layoutState.leftWidth}px`') !important;
  transition: width 0.2s ease, opacity 0.2s ease;
}

.mu-layout-wrapper.left-hidden .VPSidebar {
  width: 0 !important;
  opacity: 0;
  overflow: hidden;
}

/* Right sidebar (outline) */
.mu-layout-wrapper.has-toolbar .VPDocAsideOutline,
.mu-layout-wrapper.has-toolbar .VPDoc .aside {
  width: v-bind('`${layoutState.rightWidth}px`') !important;
  transition: width 0.2s ease, opacity 0.2s ease;
}

.mu-layout-wrapper.right-hidden .VPDocAsideOutline,
.mu-layout-wrapper.right-hidden .VPDoc .aside {
  width: 0 !important;
  opacity: 0;
  overflow: hidden;
}

/* Content max-width */
.mu-layout-wrapper.has-toolbar .VPDoc .content-container {
  max-width: v-bind('contentMaxWidth') !important;
  transition: max-width 0.2s ease;
}

/* Adjust main content margin when sidebars change */
.mu-layout-wrapper.has-toolbar .VPDoc.has-aside .content {
  padding-right: v-bind('layoutState.rightVisible ? `calc(${layoutState.rightWidth}px + 32px)` : "32px"') !important;
  transition: padding-right 0.2s ease;
}

.mu-layout-wrapper.has-toolbar .VPDoc .content {
  padding-left: v-bind('layoutState.leftVisible ? `calc(${layoutState.leftWidth}px + 32px)` : "32px"') !important;
  transition: padding-left 0.2s ease;
}

/* ============ Responsive ============ */
@media (max-width: 960px) {
  .mu-layout-toolbar {
    display: none;
  }
  
  .mu-resizer {
    display: none;
  }
}
</style>
