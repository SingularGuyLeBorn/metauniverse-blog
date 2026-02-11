<script setup lang="ts">
/**
 * Layout Toolbar
 * 
 * 职责：
 * 1. 提供布局控制按钮 (切换侧边栏、内容宽度)
 * 2. 状态完全由 stores/layout.ts 管理
 * 3. 不包含任何拖拽/Resize 逻辑
 */
import { ref, computed } from 'vue'
import { useData } from 'vitepress'
import { useLayoutStore } from '../../stores/layout'
import { useAnnotationStore } from '../../stores/annotation'
import FindInPage from './FindInPage.vue'

const layoutStore = useLayoutStore()
const annotationStore = useAnnotationStore()
const showFind = ref(false)

const { page, frontmatter } = useData()

// 仅在非首页且非索引页显示
const isDocPage = computed(() => {
  return page.value.relativePath &&
         !page.value.relativePath.endsWith('index.md') &&
         frontmatter.value.layout !== 'home'
})
</script>

<template>
  <div v-if="isDocPage" class="layout-controller">
    <!-- 顶部工具栏 -->
    <div class="layout-toolbar">
      <!-- 左侧边栏切换 -->
      <button 
        class="btn" 
        @click="layoutStore.toggleLeft()" 
        :class="{ active: layoutStore.leftVisible }" 
        title="切换侧边栏 (Left Sidebar)"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <rect x="3" y="3" width="18" height="18" rx="2"/>
          <line x1="9" y1="3" x2="9" y2="21"/>
        </svg>
      </button>

      <!-- 内容宽度切换 -->
      <div class="width-group">
        <button 
          class="width-btn" 
          :class="{ active: layoutStore.contentWidth === 'narrow' }" 
          @click="layoutStore.setContentWidth('narrow')"
          title="窄模式"
        >窄</button>
        <button 
          class="width-btn" 
          :class="{ active: layoutStore.contentWidth === 'half' }" 
          @click="layoutStore.setContentWidth('half')"
          title="标准模式"
        >中</button>
        <button 
          class="width-btn" 
          :class="{ active: layoutStore.contentWidth === 'full' }" 
          @click="layoutStore.setContentWidth('full')"
          title="宽模式"
        >宽</button>
      </div>

      <!-- 右侧边栏切换 -->
      <button 
        class="btn" 
        @click="layoutStore.toggleRight()" 
        :class="{ active: layoutStore.rightVisible }" 
        title="切换大纲 (Right Sidebar)"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <rect x="3" y="3" width="18" height="18" rx="2"/>
          <line x1="15" y1="3" x2="15" y2="21"/>
        </svg>
      </button>

      <!-- Zen Mode Switch -->
      <button 
        class="btn" 
        @click="layoutStore.toggleZenMode()" 
        :class="{ active: layoutStore.zenMode }" 
        title="禅模式 (Zen Mode)"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/></svg>
      </button>

      <!-- 查找工具 -->
      <button 
        class="btn" 
        @click="showFind = !showFind" 
        :class="{ active: showFind }" 
        title="页内查找"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <circle cx="11" cy="11" r="8"/>
          <line x1="21" y1="21" x2="16.65" y2="16.65"/>
        </svg>
      </button>

      <!-- Markup Mode (Annotation) -->
      <button 
        class="btn markup-btn" 
        @click="annotationStore.toggleMarkupMode()" 
        :class="{ active: annotationStore.isMarkupMode }" 
        title="批注模式 (Markup Mode)"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M12 19l7-7 3 3-7 7-3-3z"/>
          <path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/>
          <path d="M2 2l1.5 1.5"/>
        </svg>
      </button>
    </div>

    <Teleport to="body">
      <FindInPage v-if="showFind" @close="showFind = false" />
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
  transition: all 0.2s;
}
.btn:hover { background: var(--vp-c-bg-soft); color: var(--vp-c-text-1); }
.btn.active { color: var(--vp-c-brand-1); background: var(--vp-c-bg-soft); }
.markup-btn.active { color: var(--vp-c-brand-1); background: rgba(var(--vp-c-brand-1-rgb, 107, 114, 255), 0.1); }

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
  transition: all 0.2s;
}
.width-btn:hover { color: var(--vp-c-text-2); }
.width-btn.active { background: var(--vp-c-bg); color: var(--vp-c-brand-1); font-weight: 500; }
</style>

