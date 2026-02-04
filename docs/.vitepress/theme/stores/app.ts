import { defineStore } from 'pinia'
import { ref, computed, watch } from 'vue'
import { usePreferredDark, useLocalStorage, useMediaQuery } from '@vueuse/core'

export type LayoutMode = 'default' | 'focus' | 'wide'
export type Theme = 'light' | 'dark' | 'auto'
export type ReadingMode = 'skimming' | 'experimental'

export const useAppStore = defineStore('app', () => {
  // ============ 状态 ============
  const theme = useLocalStorage<Theme>('mu-theme', 'auto')
  const layoutMode = useLocalStorage<LayoutMode>('mu-layout', 'default')
  const readingMode = useLocalStorage<ReadingMode>('mu-reading-mode', 'skimming')
  const sidebarCollapsed = useLocalStorage('mu-sidebar-collapsed', false)
  const fontSize = useLocalStorage('mu-font-size', 16)
  const lineHeight = useLocalStorage('mu-line-height', 1.75)
  const isLoading = ref(false)
  
  // ============ 计算属性 ============
  const prefersDark = usePreferredDark()
  
  const isDark = computed(() => {
    if (theme.value === 'auto') {
      return prefersDark.value
    }
    return theme.value === 'dark'
  })
  
  const isSkimmingMode = computed(() => readingMode.value === 'skimming')
  const isExperimentalMode = computed(() => readingMode.value === 'experimental')
  const isFocusMode = computed(() => layoutMode.value === 'focus')
  const isWideMode = computed(() => layoutMode.value === 'wide')
  const isMobile = useMediaQuery('(max-width: 768px)')
  
  // ============ 方法 ============
  const setTheme = (t: Theme) => {
    theme.value = t
  }
  
  const toggleTheme = () => {
    const themes: Theme[] = ['light', 'dark', 'auto']
    const currentIndex = themes.indexOf(theme.value)
    theme.value = themes[(currentIndex + 1) % themes.length]
  }
  
  const setLayoutMode = (mode: LayoutMode) => {
    layoutMode.value = mode
  }
  
  const toggleLayoutMode = () => {
    const modes: LayoutMode[] = ['default', 'focus', 'wide']
    const currentIndex = modes.indexOf(layoutMode.value)
    layoutMode.value = modes[(currentIndex + 1) % modes.length]
  }
  
  const setReadingMode = (mode: ReadingMode) => {
    readingMode.value = mode
    if (typeof document !== 'undefined') {
      document.documentElement.setAttribute('data-reading-mode', mode)
    }
  }
  
  const toggleReadingMode = () => {
    setReadingMode(readingMode.value === 'skimming' ? 'experimental' : 'skimming')
  }
  
  const toggleSidebar = () => {
    sidebarCollapsed.value = !sidebarCollapsed.value
  }
  
  const setLoading = (loading: boolean) => {
    isLoading.value = loading
  }
  
  // ============ 副作用 ============
  watch(isDark, (dark) => {
    if (typeof document !== 'undefined') {
      document.documentElement.classList.toggle('dark', dark)
    }
  }, { immediate: true })
  
  watch(layoutMode, (mode) => {
    if (typeof document !== 'undefined') {
      document.documentElement.setAttribute('data-layout', mode)
    }
  }, { immediate: true })
  
  watch(readingMode, (mode) => {
    if (typeof document !== 'undefined') {
      document.documentElement.setAttribute('data-reading-mode', mode)
    }
  }, { immediate: true })
  
  return {
    // 状态
    theme,
    layoutMode,
    readingMode,
    sidebarCollapsed,
    fontSize,
    lineHeight,
    isLoading,
    
    // 计算属性
    isDark,
    isSkimmingMode,
    isExperimentalMode,
    isFocusMode,
    isWideMode,
    isMobile,
    
    // 方法
    setTheme,
    toggleTheme,
    setLayoutMode,
    toggleLayoutMode,
    setReadingMode,
    toggleReadingMode,
    toggleSidebar,
    setLoading
  }
})
