<script setup lang="ts">
/**
 * Sidebar Toolbar (完整重写)
 * 
 * 提供侧边栏的全局控制按钮：
 * - 全部展开
 * - 全部折叠  
 * - 定位当前文档
 */
import { onMounted, watch, nextTick } from 'vue'
import { useRoute } from 'vitepress'

const route = useRoute()

/**
 * 展开所有折叠的侧边栏分组
 */
const expandAll = (): void => {
  const groups = document.querySelectorAll('.VPSidebarItem.collapsible')
  groups.forEach(group => {
    // VitePress 使用 'collapsed' 类来标记折叠状态
    if (group.classList.contains('collapsed')) {
      const button = group.querySelector('.item') as HTMLElement
      if (button) {
        button.click()
      }
    }
  })
}

/**
 * 折叠所有展开的侧边栏分组
 */
const collapseAll = (): void => {
  const groups = document.querySelectorAll('.VPSidebarItem.collapsible')
  groups.forEach(group => {
    // 没有 'collapsed' 类表示当前是展开状态
    if (!group.classList.contains('collapsed')) {
      const button = group.querySelector('.item') as HTMLElement
      if (button) {
        button.click()
      }
    }
  })
}

/**
 * 定位并展开到当前激活的文档
 */
const locateCurrent = async (): Promise<void> => {
  await nextTick()
  
  // 等待 DOM 更新
  setTimeout(() => {
    const activeItem = document.querySelector('.VPSidebarItem.is-active')
    if (!activeItem) return
    
    // 1. 先展开所有父级分组
    let parent = activeItem.parentElement
    while (parent) {
      if (parent.classList.contains('VPSidebar')) break
      
      // 找到包含此节点的可折叠分组
      const collapsibleParent = parent.closest('.VPSidebarItem.collapsible')
      if (collapsibleParent && collapsibleParent.classList.contains('collapsed')) {
        const button = collapsibleParent.querySelector('.item') as HTMLElement
        if (button) button.click()
      }
      
      parent = parent.parentElement
    }
    
    // 2. 滚动到激活项
    setTimeout(() => {
      activeItem.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }, 100)
  }, 200)
}

// 页面加载和路由变化时自动定位
onMounted(() => {
  locateCurrent()
})

watch(
  () => route.path,
  () => {
    locateCurrent()
  }
)
</script>

<template>
  <div class="sidebar-toolbar">
    <div class="toolbar-actions">
      <button 
        class="toolbar-btn" 
        @click="expandAll" 
        title="全部展开"
        aria-label="全部展开"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="15 3 21 3 21 9"/>
          <polyline points="9 21 3 21 3 15"/>
          <line x1="21" y1="3" x2="14" y2="10"/>
          <line x1="3" y1="21" x2="10" y2="14"/>
        </svg>
      </button>
      
      <button 
        class="toolbar-btn" 
        @click="collapseAll" 
        title="全部折叠"
        aria-label="全部折叠"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="4 14 10 14 10 20"/>
          <polyline points="20 10 14 10 14 4"/>
          <line x1="14" y1="10" x2="21" y2="3"/>
          <line x1="3" y1="21" x2="10" y2="14"/>
        </svg>
      </button>
      
      <button 
        class="toolbar-btn" 
        @click="locateCurrent" 
        title="定位当前文档"
        aria-label="定位当前文档"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="10"/>
          <circle cx="12" cy="12" r="3"/>
        </svg>
      </button>
    </div>
  </div>
</template>

<style scoped>
.sidebar-toolbar {
  display: flex;
  justify-content: flex-end;
  padding: 12px 16px;
  background-color: var(--vp-c-bg);
  border-bottom: 1px solid var(--vp-c-divider);
  position: sticky;
  top: 0;
  z-index: 20;
}

.toolbar-actions {
  display: flex;
  gap: 8px;
}

.toolbar-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  padding: 0;
  border: 1px solid transparent;
  background-color: transparent;
  color: var(--vp-c-text-2);
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.15s ease;
}

.toolbar-btn:hover {
  background-color: var(--vp-c-bg-soft);
  color: var(--vp-c-brand-1);
  border-color: var(--vp-c-divider);
}

.toolbar-btn:active {
  transform: scale(0.95);
}

.toolbar-btn svg {
  width: 16px;
  height: 16px;
}
</style>
