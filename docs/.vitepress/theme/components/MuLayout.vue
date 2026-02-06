<script setup lang="ts">
/**
 * MuLayout - 博客主布局组件
 * 
 * 职责：
 * 1. 提供基础布局结构
 * 2. 状态管理完全由 stores/layout.ts 负责
 * 3. 分割线由 GlobalLayoutControl.vue 渲染 (Teleport to body)
 * 
 * 不再包含：
 * - 独立的 layoutState (已迁移到 Pinia)
 * - 拖拽逻辑 (由 GlobalLayoutControl 处理)
 * - 分割线渲染 (由 GlobalLayoutControl 处理)
 */
import { computed, onMounted, watch } from 'vue'
import { useData } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import { useLayoutStore } from '../stores/layout'

// 功能组件
import ScrollProgress from './features/ScrollProgress.vue'
import ModeSwitcher from './features/ModeSwitcher.vue'
import SemanticHeatmap from './features/SemanticHeatmap.vue'
import RelatedReferences from './features/RelatedReferences.vue'
import GlobalLayoutControl from './features/GlobalLayoutControl.vue'

const layoutStore = useLayoutStore()
const { frontmatter, page } = useData()

// ============ Computed ============
const isDocPage = computed(() => {
  return page.value.relativePath && 
         !page.value.relativePath.endsWith('index.md') &&
         frontmatter.value.layout !== 'home' &&
         frontmatter.value.layout !== 'page'
})

const showToolbar = computed(() => isDocPage.value)

// 内容最大宽度
const contentMaxWidth = computed(() => {
  switch (layoutStore.contentWidth) {
    case 'narrow': return '680px'
    case 'half': return '900px'
    case 'full': return '100%'
    default: return '900px'
  }
})

// ============ 响应式 CSS 变量绑定 ============
const leftWidthPx = computed(() => `${layoutStore.leftWidth}px`)
const rightWidthPx = computed(() => `${layoutStore.rightWidth}px`)
const leftPadding = computed(() => layoutStore.leftVisible ? `calc(${layoutStore.leftWidth}px + 32px)` : '32px')
const rightPadding = computed(() => layoutStore.rightVisible ? `calc(${layoutStore.rightWidth}px + 32px)` : '32px')

// ============ Lifecycle ============
onMounted(() => {
  // 确保初始布局正确应用
  layoutStore.applyLayout()
})
</script>

<template>
  <div 
    class="mu-layout-wrapper" 
    :class="{ 
      'dragging': layoutStore.isResizing,
      'has-toolbar': showToolbar,
      'left-hidden': !layoutStore.leftVisible,
      'right-hidden': !layoutStore.rightVisible
    }"
  >
    <!-- Layout Toolbar (顶部居中的布局控制) -->
    <Transition name="toolbar-fade">
      <div v-if="showToolbar" class="mu-layout-toolbar">
        <!-- Left toggle -->
        <button 
          class="mu-toolbar-btn" 
          @click="layoutStore.toggleLeft()" 
          :class="{ active: layoutStore.leftVisible }"
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
            :class="{ active: layoutStore.contentWidth === 'narrow' }"
            @click="layoutStore.setContentWidth('narrow')"
            title="窄"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="8" y="4" width="8" height="16" rx="1"/>
            </svg>
          </button>
          <button 
            class="mu-width-btn" 
            :class="{ active: layoutStore.contentWidth === 'half' }"
            @click="layoutStore.setContentWidth('half')"
            title="半宽"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="5" y="4" width="14" height="16" rx="1"/>
            </svg>
          </button>
          <button 
            class="mu-width-btn" 
            :class="{ active: layoutStore.contentWidth === 'full' }"
            @click="layoutStore.setContentWidth('full')"
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
          @click="layoutStore.toggleRight()" 
          :class="{ active: layoutStore.rightVisible }"
          title="切换大纲 (Ctrl+])"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="3" width="18" height="18" rx="2"/>
            <line x1="15" y1="3" x2="15" y2="21"/>
          </svg>
        </button>
      </div>
    </Transition>

    <!-- Global Layout Control (处理分割线 + 拖拽) -->
    <GlobalLayoutControl />

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
  z-index: 99; /* 低于 Navbar (100)，但高于内容 */
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

/* ============ Dynamic Sidebar Widths (使用 Pinia Store 的值) ============ */
.mu-layout-wrapper.has-toolbar .VPSidebar {
  width: v-bind('leftWidthPx') !important;
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
  width: v-bind('rightWidthPx') !important;
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
  padding-right: v-bind('rightPadding') !important;
  transition: padding-right 0.2s ease;
}

.mu-layout-wrapper.has-toolbar .VPDoc .content {
  padding-left: v-bind('leftPadding') !important;
  transition: padding-left 0.2s ease;
}

/* ============ Responsive ============ */
@media (max-width: 960px) {
  .mu-layout-toolbar {
    display: none;
  }
}
</style>
