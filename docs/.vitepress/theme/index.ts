import type { Theme } from 'vitepress'
import { h, watch } from 'vue'
import { createPinia } from 'pinia'
import DefaultTheme from 'vitepress/theme'
import { useData } from 'vitepress'

// 样式
import './style.css'
import './styles/reading-mode.css'

// 状态管理
import { useAppStore } from './stores/app'

// 组件
import ModeSwitcher from './components/features/ModeSwitcher.vue'
import ThemeToggle from './components/widgets/ThemeToggle.vue'
import KnowledgeGraph from './components/features/KnowledgeGraph.vue'

export default {
  extends: DefaultTheme,
  
  Layout: () => {
    const { frontmatter } = useData()
    
    return h(DefaultTheme.Layout, null, {
      // 布局底部插槽 - 模式切换器
      'layout-bottom': () => h('div', { id: 'mu-teleport-container' }, [
        h(ModeSwitcher)
      ]),
      
      // 侧边栏底部 - 知识图谱
      'aside-bottom': () => {
        if (frontmatter.value.graph !== false) {
          return h(KnowledgeGraph)
        }
        return null
      }
    })
  },
  
  enhanceApp({ app, router, siteData }) {
    // 创建Pinia实例
    const pinia = createPinia()
    app.use(pinia)
    
    // 注册全局组件
    app.component('ThemeToggle', ThemeToggle)
    app.component('KnowledgeGraph', KnowledgeGraph)
    
    // 客户端初始化
    if (typeof window !== 'undefined') {
      // 初始化应用状态
      const appStore = useAppStore()
      
      // 路由导航钩子
      router.onBeforeRouteChange = (to) => {
        appStore.setLoading(true)
      }
      
      router.onAfterRouteChanged = (to) => {
        appStore.setLoading(false)
      }
    }
  },
  
  setup() {
    // 客户端初始化逻辑
    if (typeof window !== 'undefined') {
      const appStore = useAppStore()
      
      // 应用主题到DOM
      watch(() => appStore.isDark, (isDark) => {
        document.documentElement.classList.toggle('dark', isDark)
      }, { immediate: true })
      
      // 应用布局模式到DOM
      watch(() => appStore.layoutMode, (mode) => {
        document.documentElement.setAttribute('data-layout', mode)
      }, { immediate: true })
    }
  }
} as Theme
