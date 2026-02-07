<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vitepress'

/**
 * Quantum Popover
 * "量子纠缠" 悬浮预览卡片
 */

const props = defineProps<{
  // 没有任何 prop，直接操作全局 DOM，因为链接是 markdown 渲染生成的
}>()

const visible = ref(false)
const loading = ref(false)
const position = ref({ x: 0, y: 0 })
const content = ref({
  title: '',
  excerpt: '',
  link: ''
})

// 防抖计时器
let hoverTimer: any = null
let currentTarget: HTMLElement | null = null

// 缓存已获取的内容: url -> { title, excerpt }
const contentCache = new Map<string, { title: string, excerpt: string }>()

const handleMouseOver = (e: MouseEvent) => {
  const target = (e.target as HTMLElement).closest('a[href]') as HTMLAnchorElement
  if (!target) return

  // 检查是否是内部 wiki 链接 (拥有 .wiki-link 类名 或者 是内部相对链接)
  const isWikiLink = target.classList.contains('wiki-link') 
  // 或者简单的内部链接判断
  const isInternal = target.getAttribute('href')?.startsWith('/') || target.getAttribute('href')?.startsWith('.')

  if (!isWikiLink && !isInternal) return

  // 此时确定是我们要处理的链接
  if (hoverTimer) clearTimeout(hoverTimer)
  
  hoverTimer = setTimeout(() => {
    showPopover(target)
  }, 300) // 300ms 悬停才触发，避免误触
}

const handleMouseOut = (e: MouseEvent) => {
  if (hoverTimer) clearTimeout(hoverTimer)
  
  // 稍微延迟关闭，允许用户把鼠标移到 popover 上 (虽然目前逻辑没做移动到 popover 的保持)
  // 简单起见，鼠标离开链接即关闭
  visible.value = false
  currentTarget = null
}

const showPopover = async (element: HTMLAnchorElement) => {
  currentTarget = element
  const href = element.getAttribute('href')
  if (!href) return

  // 计算位置
  const rect = element.getBoundingClientRect()
  // 默认显示在下方，居中对齐
  position.value = {
    x: rect.left + rect.width / 2,
    y: rect.bottom + 10
  }

  // 先显示 loading 状态 (或者只显示卡片框架)
  content.value = { title: 'Loading...', excerpt: '', link: href }
  loading.value = true
  visible.value = true

  try {
    const data = await fetchContent(href)
    // 只有当鼠标还在同一个元素上时才更新内容
    if (currentTarget === element) {
      content.value = { ...data, link: href }
      loading.value = false
    }
  } catch (e) {
    if (currentTarget === element) {
      content.value.title = 'Error'
      content.value.excerpt = '无法加载预览内容'
      loading.value = false
    }
  }
}

const fetchContent = async (url: string) => {
  // 1. 检查缓存
  if (contentCache.has(url)) {
    return contentCache.get(url)!
  }

  // 2. Fetch HTML
  const res = await fetch(url)
  const html = await res.text()
  
  // 3. Parse
  const parser = new DOMParser()
  const doc = parser.parseFromString(html, 'text/html')
  
  // 提取标题 (h1)
  const title = doc.querySelector('h1')?.textContent || 
                doc.title.split('|')[0].trim() || 
                'Unknown Doc'
  
  // 提取摘要: 查找 .vp-doc 下的第一个非空 p 标签
  // 排除 h1, 排除空行
  const pTags = doc.querySelectorAll('.vp-doc p')
  let excerpt = ''
  for (const p of Array.from(pTags)) {
    const text = p.textContent?.trim()
    if (text && text.length > 10) {
      excerpt = text.slice(0, 120) + (text.length > 120 ? '...' : '')
      break
    }
  }
  
  if (!excerpt) excerpt = '暂无摘要'

  const data = { title, excerpt }
  contentCache.set(url, data)
  return data
}

onMounted(() => {
  // 全局事件代理
  // 使用 capture 阶段或者在 bubbling 阶段 body 上监听
  document.body.addEventListener('mouseover', handleMouseOver)
  document.body.addEventListener('mouseout', handleMouseOut)
})

onUnmounted(() => {
  document.body.removeEventListener('mouseover', handleMouseOver)
  document.body.removeEventListener('mouseout', handleMouseOut)
})
</script>

<template>
  <Transition name="popover-spring">
    <div 
      v-if="visible"
      class="quantum-popover"
      :style="{ 
        left: `${position.x}px`, 
        top: `${position.y}px` 
      }"
    >
      <div class="popover-content" :class="{ loading }">
        <div class="popover-header">
          <span class="icon">🪐</span>
          <span class="title">{{ content.title }}</span>
        </div>
        
        <div class="popover-body">
          {{ content.excerpt }}
        </div>
        
        <div class="popover-footer" v-if="!loading">
          <span class="read-more">点击链接阅读全文 →</span>
        </div>
      </div>
    </div>
  </Transition>
</template>

<style scoped>
.quantum-popover {
  position: fixed;
  z-index: 1000;
  width: 320px;
  max-width: 90vw;
  transform: translateX(-50%); /* 居中定位 */
  pointer-events: none; /* 让鼠标事件透过，防止触发 mouseout 导致闪烁? */
  /* 如果我们想让用户能把鼠标移到 popover 上，需要更复杂的 mouseout 逻辑。
     现在为了简单，设为 pointer-events: none，只作为视觉展示 */
}

/* Glassmorphism Card */
.popover-content {
  background: rgba(255, 255, 255, 0.7);
  backdrop-filter: blur(16px) saturate(180%);
  -webkit-backdrop-filter: blur(16px) saturate(180%);
  border: 1px solid rgba(255, 255, 255, 0.3);
  border-radius: 16px;
  padding: 16px;
  box-shadow: 
    0 4px 6px -1px rgba(0, 0, 0, 0.1), 
    0 2px 4px -1px rgba(0, 0, 0, 0.06),
    0 12px 32px rgba(0, 0, 0, 0.15); /* Deep shadow */
  color: var(--vp-c-text-1);
}

.dark .popover-content {
  background: rgba(30, 30, 30, 0.7);
  border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Loading State Shimmer */
.popover-content.loading .popover-body {
  opacity: 0.5;
  filter: blur(2px);
}

.popover-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  border-bottom: 1px solid var(--vp-c-divider);
  padding-bottom: 8px;
}

.popover-header .title {
  font-weight: 600;
  font-size: 14px;
  line-height: 1.4;
}

.popover-body {
  font-size: 13px;
  line-height: 1.5;
  color: var(--vp-c-text-2);
  display: -webkit-box;
  -webkit-line-clamp: 4;
  line-clamp: 4;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.popover-footer {
  margin-top: 8px;
  font-size: 12px;
  color: var(--vp-c-brand);
  text-align: right;
  opacity: 0.8;
}

/* Spring Animation */
.popover-spring-enter-active {
  transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); /* Springy */
  transform-origin: top center;
}

.popover-spring-leave-active {
  transition: all 0.2s ease-in;
  transform-origin: top center;
}

.popover-spring-enter-from,
.popover-spring-leave-to {
  opacity: 0;
  transform: translateX(-50%) translateY(10px) scale(0.9);
}

.popover-spring-enter-to {
  opacity: 1;
  transform: translateX(-50%) translateY(0) scale(1);
}
</style>
