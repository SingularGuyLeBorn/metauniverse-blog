<script setup lang="ts">
/**
 * Layout Toolbar - 完全重写
 * 
 * 核心逻辑：
 * 1. 顶部栏绝对不受影响 (z-index: 100)
 * 2. 左侧栏左边缘 = 浏览器左边缘
 * 3. 右侧栏右边缘 = 浏览器右边缘
 * 4. 分割线只调整侧栏宽度，不移动侧栏位置
 */
import { ref, computed, onMounted, watch, onUnmounted, nextTick } from 'vue'
import { useData } from 'vitepress'
import FindInPage from './FindInPage.vue'

// Constants
const STORAGE_KEY = 'mu-layout-state-v2'
const MIN_SIDEBAR_WIDTH = 200
const MAX_SIDEBAR_WIDTH = 500
const MIN_ASIDE_WIDTH = 180
const MAX_ASIDE_WIDTH = 400

// State
interface LayoutState {
  leftVisible: boolean
  rightVisible: boolean
  leftWidth: number
  rightWidth: number
  contentWidth: 'narrow' | 'half' | 'full'
}

const state = ref<LayoutState>({
  leftVisible: true,
  rightVisible: true,
  leftWidth: 280,
  rightWidth: 224,
  contentWidth: 'half'
})

const isResizing = ref(false)
const resizingSide = ref<'left' | 'right' | null>(null)
const showFind = ref(false)

const { page, frontmatter } = useData()

const isDocPage = computed(() => {
  return page.value.relativePath &&
         !page.value.relativePath.endsWith('index.md') &&
         frontmatter.value.layout !== 'home'
})

// 应用布局到 CSS 变量
const applyLayout = (): void => {
  if (typeof document === 'undefined') return
  
  const root = document.documentElement
  
  // 设置宽度变量
  root.style.setProperty('--mu-sidebar-width', state.value.leftVisible ? `${state.value.leftWidth}px` : '0px')
  root.style.setProperty('--mu-aside-width', state.value.rightVisible ? `${state.value.rightWidth}px` : '0px')
  
  // 设置内容宽度属性
  root.setAttribute('data-content-width', state.value.contentWidth)
  
  // 设置状态类
  root.classList.toggle('mu-left-hidden', !state.value.leftVisible)
  root.classList.toggle('mu-right-hidden', !state.value.rightVisible)
  root.classList.toggle('is-resizing', isResizing.value)
}

// 切换侧边栏
const toggleLeft = (): void => {
  state.value.leftVisible = !state.value.leftVisible
  applyLayout()
}

const toggleRight = (): void => {
  state.value.rightVisible = !state.value.rightVisible
  applyLayout()
}

const setContentWidth = (width: 'narrow' | 'half' | 'full'): void => {
  state.value.contentWidth = width
  applyLayout()
}

// ============ 拖拽逻辑 (完全重写) ============

const startResize = (side: 'left' | 'right', e: MouseEvent): void => {
  e.preventDefault()
  isResizing.value = true
  resizingSide.value = side
  applyLayout()
  
  document.addEventListener('mousemove', handleResize)
  document.addEventListener('mouseup', stopResize)
}

const handleResize = (e: MouseEvent): void => {
  if (!resizingSide.value) return
  
  if (resizingSide.value === 'left') {
    // 左侧栏宽度 = 鼠标X坐标 (因为左边缘在0)
    const newWidth = Math.max(MIN_SIDEBAR_WIDTH, Math.min(e.clientX, MAX_SIDEBAR_WIDTH))
    state.value.leftWidth = newWidth
  } else {
    // 右侧栏宽度 = 屏幕宽度 - 鼠标X坐标 (因为右边缘在屏幕最右)
    const screenWidth = window.innerWidth
    const newWidth = Math.max(MIN_ASIDE_WIDTH, Math.min(screenWidth - e.clientX, MAX_ASIDE_WIDTH))
    state.value.rightWidth = newWidth
  }
  
  applyLayout()
}

const stopResize = (): void => {
  isResizing.value = false
  resizingSide.value = null
  applyLayout()
  
  document.removeEventListener('mousemove', handleResize)
  document.removeEventListener('mouseup', stopResize)
}

// 持久化
watch(state, (val) => {
  if (typeof window !== 'undefined') {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(val))
  }
}, { deep: true })

onMounted(() => {
  const saved = localStorage.getItem(STORAGE_KEY)
  if (saved) {
    try {
      Object.assign(state.value, JSON.parse(saved))
    } catch (e) { /* ignore */ }
  }
  nextTick(() => applyLayout())
})

onUnmounted(() => {
  document.removeEventListener('mousemove', handleResize)
  document.removeEventListener('mouseup', stopResize)
})
</script>

<template>
  <div v-if="isDocPage" class="layout-controller">
    <!-- 顶部工具栏 -->
    <div class="layout-toolbar">
      <button class="btn" @click="toggleLeft" :class="{ active: state.leftVisible }" title="切换侧边栏">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <rect x="3" y="3" width="18" height="18" rx="2"/>
          <line x1="9" y1="3" x2="9" y2="21"/>
        </svg>
      </button>

      <div class="width-group">
        <button class="width-btn" :class="{ active: state.contentWidth === 'narrow' }" @click="setContentWidth('narrow')">窄</button>
        <button class="width-btn" :class="{ active: state.contentWidth === 'half' }" @click="setContentWidth('half')">中</button>
        <button class="width-btn" :class="{ active: state.contentWidth === 'full' }" @click="setContentWidth('full')">宽</button>
      </div>

      <button class="btn" @click="toggleRight" :class="{ active: state.rightVisible }" title="切换大纲">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <rect x="3" y="3" width="18" height="18" rx="2"/>
          <line x1="15" y1="3" x2="15" y2="21"/>
        </svg>
      </button>

      <button class="btn" @click="showFind = !showFind" :class="{ active: showFind }" title="查找">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <circle cx="11" cy="11" r="8"/>
          <line x1="21" y1="21" x2="16.65" y2="16.65"/>
        </svg>
      </button>
    </div>

    <Teleport to="body">
      <FindInPage v-if="showFind" @close="showFind = false" />

      <!-- 左侧分割线：位置 = 左侧栏宽度 -->
      <div
        v-if="state.leftVisible"
        class="resizer resizer-left"
        :style="{ left: state.leftWidth + 'px' }"
        @mousedown="startResize('left', $event)"
      >
        <div class="resizer-handle"></div>
      </div>

      <!-- 右侧分割线：位置 = 右侧边距 -->
      <div
        v-if="state.rightVisible"
        class="resizer resizer-right"
        :style="{ right: state.rightWidth + 'px' }"
        @mousedown="startResize('right', $event)"
      >
        <div class="resizer-handle"></div>
      </div>
    </Teleport>
  </div>
</template>

<style scoped>
.layout-toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-left: 16px;
}

.btn {
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
}
.btn:hover { background: var(--vp-c-bg-soft); }
.btn.active { color: var(--vp-c-brand-1); background: var(--vp-c-bg-soft); }

.width-group {
  display: flex;
  gap: 2px;
  padding: 2px;
  background: var(--vp-c-bg-soft);
  border-radius: 6px;
}

.width-btn {
  padding: 4px 8px;
  border: none;
  background: transparent;
  border-radius: 4px;
  font-size: 12px;
  color: var(--vp-c-text-3);
  cursor: pointer;
}
.width-btn:hover { color: var(--vp-c-text-2); }
.width-btn.active { background: var(--vp-c-bg); color: var(--vp-c-brand-1); }

/* 分割线 - 关键样式 */
.resizer {
  position: fixed;
  top: 64px; /* 从导航栏下方开始，绝不影响顶部 */
  bottom: 0;
  width: 8px;
  cursor: col-resize;
  z-index: 50;
  display: flex;
  justify-content: center;
}

.resizer-left {
  transform: translateX(-50%);
}

.resizer-right {
  transform: translateX(50%);
}

.resizer-handle {
  width: 2px;
  height: 100%;
  background: transparent;
  transition: background 0.2s;
}

.resizer:hover .resizer-handle {
  background: var(--vp-c-brand-1);
  opacity: 0.5;
}
</style>
