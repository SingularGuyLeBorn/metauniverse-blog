<script setup lang="ts">
/**
 * Global Layout Control
 * 
 * 核心职责：
 * 1. 监听全局鼠标事件，处理侧边栏拖拽
 * 2. 渲染 Teleport 到 body 的分割线
 * 3. 确保分割线和拖拽逻辑完全独立于 Navbar
 * 
 * 关键：分割线的 top 必须是 64px (顶部栏高度)，绝不能影响顶部栏
 */
import { onMounted, onUnmounted, computed } from 'vue'
import { useLayoutStore } from '../../stores/layout'

const layoutStore = useLayoutStore()

// 响应式位置计算
const leftResizerLeft = computed(() => `${layoutStore.leftWidth}px`)
const rightResizerRight = computed(() => `${layoutStore.rightWidth}px`)

// 拖拽逻辑
const handleMouseMove = (e: MouseEvent) => {
  if (!layoutStore.isResizing || !layoutStore.resizingSide) return
  
  e.preventDefault()
  
  if (layoutStore.resizingSide === 'left') {
    // 左侧：宽度 = 鼠标 X 坐标
    layoutStore.setLeftWidth(e.clientX)
  } else {
    // 右侧：宽度 = 窗口宽度 - 鼠标 X 坐标
    const windowWidth = document.documentElement.clientWidth
    layoutStore.setRightWidth(windowWidth - e.clientX)
  }
}

const handleMouseUp = () => {
  if (layoutStore.isResizing) {
    layoutStore.stopResize()
  }
}

// 启动拖拽
const startResizeLeft = (e: MouseEvent) => {
  e.preventDefault()
  layoutStore.startResize('left')
}

const startResizeRight = (e: MouseEvent) => {
  e.preventDefault()
  layoutStore.startResize('right')
}

// 生命周期
onMounted(() => {
  document.addEventListener('mousemove', handleMouseMove)
  document.addEventListener('mouseup', handleMouseUp)
  // 确保初始布局正确应用
  layoutStore.applyLayout()
})

onUnmounted(() => {
  document.removeEventListener('mousemove', handleMouseMove)
  document.removeEventListener('mouseup', handleMouseUp)
})
</script>

<template>
  <Teleport to="body">
    <!-- 左侧分割线 (严格位于顶部栏下方) -->
    <div
      v-show="layoutStore.leftVisible"
      class="mu-resizer mu-resizer-left"
      :style="{ left: leftResizerLeft }"
      @mousedown="startResizeLeft"
    >
      <div class="mu-resizer-handle"></div>
    </div>

    <!-- 右侧分割线 (严格位于顶部栏下方) -->
    <div
      v-show="layoutStore.rightVisible"
      class="mu-resizer mu-resizer-right"
      :style="{ right: rightResizerRight }"
      @mousedown="startResizeRight"
    >
      <div class="mu-resizer-handle"></div>
    </div>
    
    <!-- 拖拽时的遮罩层 (防止 iframe 等干扰) -->
    <div v-if="layoutStore.isResizing" class="mu-resize-overlay"></div>
  </Teleport>
</template>

<style scoped>
/* 分割线基础样式 */
.mu-resizer {
  position: fixed;
  /* 关键：严格从顶部栏下方开始，64px = --mu-header-height */
  top: 64px;
  bottom: 0;
  width: 12px; /* 增加感应区域 */
  z-index: 90; /* 低于 Navbar (100) 但高于内容 */
  cursor: col-resize;
  display: flex;
  justify-content: center;
  align-items: center;
  user-select: none;
}

/* 左侧分割线定位 */
.mu-resizer-left {
  transform: translateX(-50%); /* 居中对齐边界 */
}

/* 右侧分割线定位 */
.mu-resizer-right {
  transform: translateX(50%); /* 居中对齐边界 */
}

/* 分割线手柄 */
.mu-resizer-handle {
  width: 2px;
  height: 100%;
  background-color: transparent;
  transition: background-color 0.2s, opacity 0.2s;
}

/* 悬停和激活状态 */
.mu-resizer:hover .mu-resizer-handle,
.mu-resizer:active .mu-resizer-handle {
  background-color: var(--vp-c-brand);
  opacity: 0.8;
}

/* 增加可视反馈：当 hover 时显示一条细线 */
.mu-resizer:hover::after {
  content: '';
  position: absolute;
  top: 0;
  bottom: 0;
  width: 1px;
  background: var(--vp-c-divider);
  opacity: 0.5;
}

/* 拖拽时的遮罩层 */
.mu-resize-overlay {
  position: fixed;
  /* 关键：遮罩从顶部栏下方开始，不覆盖顶部栏 */
  top: 64px;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 9999;
  cursor: col-resize;
  background: transparent;
}
</style>
