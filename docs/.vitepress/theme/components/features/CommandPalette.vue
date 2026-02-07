<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRouter, useData } from 'vitepress'
import { useLayoutStore } from '../../stores/layout'
import Fuse from 'fuse.js'

const visible = ref(false)
const searchQuery = ref('')
const selectedIndex = ref(0)
const inputRef = ref<HTMLInputElement | null>(null)

const router = useRouter()
const { theme, site } = useData()
const layoutStore = useLayoutStore()

// --- Data Indexing ---
interface CommandItem {
  id: string
  title: string
  description?: string
  link?: string
  action?: () => void
  category: 'Navigation' | 'Command' | 'Theme'
}

const commands = ref<CommandItem[]>([])
let fuse: Fuse<CommandItem> | null = null

const initCommands = () => {
  const list: CommandItem[] = []

  // 1. System Commands
  list.push(
    {
      id: 'cmd-theme-toggle',
      title: '切换外观模式 (Toggle Theme)',
      description: 'Switch between Dark and Light mode',
      category: 'Theme',
      action: () => {
        // VitePress default theme toggle logic relies on .VPSwitchAppearance
        // We can simulate click or manually toggle class
        const switchBtn = document.querySelector('.VPSwitchAppearance') as HTMLElement
        if (switchBtn) switchBtn.click()
      }
    },
    {
      id: 'cmd-zen-mode',
      title: '切换深空沉浸模式 (Toggle Zen Mode)',
      description: 'Hide sidebars for immersive reading',
      category: 'Command',
      action: () => {
        layoutStore.toggleZenMode()
      }
    },
    {
      id: 'cmd-expand-all',
      title: '展开所有侧边栏 (Expand Sidebar)',
      description: 'Expand all sidebar groups',
      category: 'Command',
      action: () => {
        const btn = document.querySelector('.mu-sidebar-btn[title*="展开"]') as HTMLElement
        if (btn) btn.click()
      }
    }
  )

  // 2. Navigation (Flatten sidebar)
  // This is a simplified traversal. Actual sidebar structure can be complex.
  // We'll try to walk through theme.sidebar if available.
  const sidebar = theme.value.sidebar
  if (sidebar) {
    const walk = (items: any[], prefix = '') => {
      for (const item of items) {
        if (item.link) {
          list.push({
            id: `nav-${item.link}`,
            title: item.text,
            description: prefix ? `${prefix} > ${item.text}` : item.text,
            link: item.link,
            category: 'Navigation'
          })
        }
        if (item.items) {
          walk(item.items, prefix ? `${prefix} > ${item.text}` : item.text)
        }
      }
    }
    
    // Sidebar can be an object (path -> items) or array
    if (Array.isArray(sidebar)) {
      walk(sidebar)
    } else if (typeof sidebar === 'object') {
      for (const path in sidebar) {
        walk(sidebar[path])
      }
    }
  }

  commands.value = list
  
  fuse = new Fuse(list, {
    keys: ['title', 'description'],
    threshold: 0.3
  })
}

// --- Search Logic ---
const filteredResults = computed(() => {
  if (!searchQuery.value) return commands.value.slice(0, 10) // Show top 10 by default
  return fuse?.search(searchQuery.value).map(r => r.item).slice(0, 10) || []
})

// --- Interaction ---
const onKeydown = (e: KeyboardEvent) => {
  if (e.key === 'k' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault()
    visible.value = !visible.value
  }
  
  if (!visible.value) return

  if (e.key === 'Escape') {
    visible.value = false
  } else if (e.key === 'ArrowDown') {
    e.preventDefault()
    selectedIndex.value = (selectedIndex.value + 1) % filteredResults.value.length
  } else if (e.key === 'ArrowUp') {
    e.preventDefault()
    selectedIndex.value = (selectedIndex.value - 1 + filteredResults.value.length) % filteredResults.value.length
  } else if (e.key === 'Enter') {
    e.preventDefault()
    execute(filteredResults.value[selectedIndex.value])
  }
}

const execute = (item: CommandItem) => {
  if (!item) return
  
  if (item.action) {
    item.action()
  } else if (item.link) {
    router.go(item.link)
  }
  
  visible.value = false
}

// --- Lifecycle ---
watch(visible, (val) => {
  if (val) {
    initCommands() // Refresh commands when opening
    searchQuery.value = ''
    selectedIndex.value = 0
    setTimeout(() => inputRef.value?.focus(), 50)
    // Lock scroll
    document.body.style.overflow = 'hidden'
  } else {
    document.body.style.overflow = ''
  }
})

onMounted(() => {
  window.addEventListener('keydown', onKeydown)
})

onUnmounted(() => {
  window.removeEventListener('keydown', onKeydown)
})
</script>

<template>
  <Transition name="fade">
    <div v-if="visible" class="command-palette-mask" @click="visible = false">
      <div class="command-palette-modal" @click.stop>
        <div class="cp-search">
          <span class="cp-icon">🔍</span>
          <input 
            ref="inputRef"
            v-model="searchQuery" 
            placeholder="Type a command or search..."
            type="text"
          >
          <span class="cp-kdb">ESC</span>
        </div>
        
        <div class="cp-list" v-if="filteredResults.length">
          <div 
            v-for="(item, index) in filteredResults" 
            :key="item.id"
            class="cp-item"
            :class="{ selected: index === selectedIndex }"
            @click="execute(item)"
            @mouseenter="selectedIndex = index"
          >
            <div class="cp-item-icon">
              <span v-if="item.category === 'Navigation'">📄</span>
              <span v-else-if="item.category === 'Theme'">🎨</span>
              <span v-else>⚙️</span>
            </div>
            <div class="cp-item-content">
              <div class="cp-item-title">{{ item.title }}</div>
              <div class="cp-item-desc" v-if="item.description">{{ item.description }}</div>
            </div>
            <div class="cp-item-action" v-if="item.category === 'Navigation'">Jump</div>
            <div class="cp-item-action" v-else>Run</div>
          </div>
        </div>
        
        <div class="cp-empty" v-else>
          No matching commands.
        </div>
        
        <div class="cp-footer">
          MetaUniverse Star Compass
        </div>
      </div>
    </div>
  </Transition>
</template>

<style scoped>
.command-palette-mask {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0, 0, 0, 0.4);
  backdrop-filter: blur(4px);
  z-index: 2000;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding-top: 100px;
}

.command-palette-modal {
  width: 600px;
  max-width: 90vw;
  background: var(--vp-c-bg);
  border-radius: 12px;
  box-shadow: 0 16px 64px rgba(0, 0, 0, 0.2);
  border: 1px solid var(--vp-c-divider);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.cp-search {
  display: flex;
  align-items: center;
  padding: 16px;
  border-bottom: 1px solid var(--vp-c-divider);
  gap: 12px;
}

.cp-icon {
  font-size: 18px;
  opacity: 0.5;
}

.cp-search input {
  flex: 1;
  background: transparent;
  border: none;
  font-size: 16px;
  color: var(--vp-c-text-1);
  outline: none;
}

.cp-kdb {
  font-size: 12px;
  padding: 2px 6px;
  border-radius: 4px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
  border: 1px solid var(--vp-c-divider);
}

.cp-list {
  max-height: 400px;
  overflow-y: auto;
  padding: 8px;
}

.cp-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.1s;
}

.cp-item.selected {
  background: var(--vp-c-brand-soft);
}

.cp-item:hover {
  background: var(--vp-c-bg-soft);
}

.cp-item.selected .cp-item-title {
  color: var(--vp-c-brand-1);
}

.cp-item-content {
  flex: 1;
  overflow: hidden;
}

.cp-item-title {
  font-weight: 500;
  color: var(--vp-c-text-1);
}

.cp-item-desc {
  font-size: 12px;
  color: var(--vp-c-text-2);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  opacity: 0.7;
}

.cp-item-action {
  font-size: 12px;
  opacity: 0.5;
}

.cp-empty {
  padding: 32px;
  text-align: center;
  color: var(--vp-c-text-2);
}

.cp-footer {
  padding: 8px 16px;
  font-size: 10px;
  color: var(--vp-c-text-3);
  text-align: right;
  border-top: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-alt);
}

/* Scrollbar */
.cp-list::-webkit-scrollbar {
  width: 6px;
}
.cp-list::-webkit-scrollbar-thumb {
  background: var(--vp-c-divider);
  border-radius: 3px;
}

/* Animations */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

.fade-enter-active .command-palette-modal {
  animation: modal-in 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}
.fade-leave-active .command-palette-modal {
  animation: modal-in 0.25s cubic-bezier(0.16, 1, 0.3, 1) reverse;
}

@keyframes modal-in {
  from {
    opacity: 0;
    transform: scale(0.96) translateY(10px);
  }
  to {
    opacity: 1;
    transform: scale(1) translateY(0);
  }
}
</style>
