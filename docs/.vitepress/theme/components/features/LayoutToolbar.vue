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
import FindInPage from './FindInPage.vue'

const layoutStore = useLayoutStore()
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

