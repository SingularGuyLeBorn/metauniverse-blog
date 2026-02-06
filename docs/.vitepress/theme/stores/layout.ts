import { defineStore } from 'pinia'
import { ref, watch } from 'vue'
import { useLocalStorage } from '@vueuse/core'

/**
 * 布局状态管理
 * 负责侧边栏宽度、显示状态的持久化和响应式更新
 */
export const useLayoutStore = defineStore('layout', () => {
    // ============ 常量 ============
    const MIN_SIDEBAR_WIDTH = 220
    const MAX_SIDEBAR_WIDTH = 500
    const MIN_ASIDE_WIDTH = 200
    const MAX_ASIDE_WIDTH = 400
    
    // ============ 状态 ============
    // 使用 useLocalStorage 自动持久化
    const leftWidth = useLocalStorage('mu-layout-left-width', 280)
    const rightWidth = useLocalStorage('mu-layout-right-width', 224)
    const leftVisible = useLocalStorage('mu-layout-left-visible', true)
    const rightVisible = useLocalStorage('mu-layout-right-visible', true)
    const contentWidth = useLocalStorage<'narrow' | 'half' | 'full'>('mu-layout-content-width', 'half')
    
    // 运行时状态 (不持久化)
    const isResizing = ref(false)
    const resizingSide = ref<'left' | 'right' | null>(null)

    // ============ Actions ============
    
    const setLeftWidth = (width: number) => {
        leftWidth.value = Math.max(MIN_SIDEBAR_WIDTH, Math.min(width, MAX_SIDEBAR_WIDTH))
    }
    
    const setRightWidth = (width: number) => {
        rightWidth.value = Math.max(MIN_ASIDE_WIDTH, Math.min(width, MAX_ASIDE_WIDTH))
    }
    
    const toggleLeft = () => {
        leftVisible.value = !leftVisible.value
    }
    
    const toggleRight = () => {
        rightVisible.value = !rightVisible.value
    }
    
    const setContentWidth = (width: 'narrow' | 'half' | 'full') => {
        contentWidth.value = width
    }
    
    const startResize = (side: 'left' | 'right') => {
        isResizing.value = true
        resizingSide.value = side
        document.body.style.cursor = 'col-resize'
        document.body.style.userSelect = 'none'
    }
    
    const stopResize = () => {
        isResizing.value = false
        resizingSide.value = null
        document.body.style.cursor = ''
        document.body.style.userSelect = ''
    }

    // ============ CSS 变量同步 ============
    // 监听状态变化并更新 CSS 变量，确保响应式
    const applyLayout = () => {
        if (typeof document === 'undefined') return
        
        const root = document.documentElement
        
        // 设置宽度变量
        root.style.setProperty('--mu-sidebar-width', leftVisible.value ? `${leftWidth.value}px` : '0px')
        root.style.setProperty('--mu-aside-width', rightVisible.value ? `${rightWidth.value}px` : '0px')
        
        // 设置内容宽度属性
        root.setAttribute('data-content-width', contentWidth.value)
        
        // 设置状态类 (辅助 CSS 选择器)
        root.classList.toggle('mu-left-hidden', !leftVisible.value)
        root.classList.toggle('mu-right-hidden', !rightVisible.value)
        root.classList.toggle('is-resizing', isResizing.value)
    }

    // 初始化和监听
    watch([leftWidth, rightWidth, leftVisible, rightVisible, contentWidth, isResizing], applyLayout, { immediate: true })

    return {
        // State
        leftWidth,
        rightWidth,
        leftVisible,
        rightVisible,
        contentWidth,
        isResizing,
        resizingSide,
        
        // Actions
        setLeftWidth,
        setRightWidth,
        toggleLeft,
        toggleRight,
        setContentWidth,
        startResize,
        stopResize,
        applyLayout,
        
        // Constants (exposed for drag logic)
        MIN_SIDEBAR_WIDTH,
        MAX_SIDEBAR_WIDTH,
        MIN_ASIDE_WIDTH,
        MAX_ASIDE_WIDTH
    }
})
