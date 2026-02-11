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
  link: '',
  headings: [] as string[]
})

// 防抖计时器
let hoverTimer: any = null
let hideTimer: any = null
let currentTarget: HTMLElement | null = null

// 缓存已获取的内容: url -> { title, excerpt, headings }
const contentCache = new Map<string, { title: string, excerpt: string, headings: string[] }>()

const handleMouseOver = (e: MouseEvent) => {
  const target = (e.target as HTMLElement).closest('a[href]') as HTMLAnchorElement
  if (!target) {
    // 如果鼠标移入的是 popover 本身，清除隐藏计时器
    if ((e.target as HTMLElement).closest('.quantum-popover')) {
      if (hideTimer) clearTimeout(hideTimer)
    }
    return
  }

  // 检查是否是内部链接
  const href = target.getAttribute('href')
  if (!href) return
  
  const isInternal = href.startsWith('/') || href.startsWith('.') || href.includes(window.location.host)
  const isHash = href.startsWith('#')
  const isAsset = /\.(png|jpe?g|gif|svg|webp|pdf|zip)$/i.test(href)

  if (!isInternal || isHash || isAsset) return

  // 清除计时器
  if (hoverTimer) clearTimeout(hoverTimer)
  if (hideTimer) clearTimeout(hideTimer)
  
  currentTarget = target
  hoverTimer = setTimeout(() => {
    showPopover(target)
  }, 400) // 略微增加延迟，减少干扰
}

const handleMouseOut = (e: MouseEvent) => {
  if (hoverTimer) clearTimeout(hoverTimer)
  
  // 延迟隐藏，给用户机会把鼠标移动到 popover 上
  hideTimer = setTimeout(() => {
    visible.value = false
    currentTarget = null
  }, 300)
}

const handlePopoverMouseOver = () => {
  if (hideTimer) clearTimeout(hideTimer)
}

const handlePopoverMouseOut = () => {
  hideTimer = setTimeout(() => {
    visible.value = false
    currentTarget = null
  }, 200)
}

const showPopover = async (element: HTMLAnchorElement) => {
  const href = element.getAttribute('href')
  if (!href) return

  // 计算位置
  const rect = element.getBoundingClientRect()
  position.value = {
    x: rect.left + rect.width / 2,
    y: rect.bottom + 8
  }

  content.value = { title: '正在加载...', excerpt: '', link: href, headings: [] }
  loading.value = true
  visible.value = true

  try {
    const data = await fetchContent(href)
    if (currentTarget === element || visible.value) {
      content.value = { ...data, link: href }
      loading.value = false
    }
  } catch (e) {
    if (visible.value) {
      content.value.title = 'Preview Not Available'
      content.value.excerpt = '无法加载该内容的预览。'
      loading.value = false
    }
  }
}

const fetchContent = async (url: string) => {
  if (contentCache.has(url)) {
    return contentCache.get(url)!
  }

  const res = await fetch(url)
  const html = await res.text()
  
  const parser = new DOMParser()
  const doc = parser.parseFromString(html, 'text/html')
  
  const title = doc.querySelector('h1')?.textContent || 
                doc.querySelector('meta[property="og:title"]')?.getAttribute('content') ||
                doc.title.split('|')[0].trim() || 
                '文档'
  
  // 优化选择器，涵盖更多内容区域
  const pTags = doc.querySelectorAll('.vp-doc p, .vp-doc li, .main p')
  let excerpt = ''
  for (const p of Array.from(pTags)) {
    const text = p.textContent?.trim()
    // 排除非常短的代码片段或导航词
    if (text && text.length > 30) {
      excerpt = text.slice(0, 160) + (text.length > 160 ? '...' : '')
      break
    }
  }

  // 如果段落没抓到，尝试 meta description
  if (!excerpt) {
    excerpt = doc.querySelector('meta[name="description"]')?.getAttribute('content') || ''
  }

  // 提取标题结构，尝试更多层级
  const headings = Array.from(doc.querySelectorAll('.vp-doc h2, .vp-doc h3'))
    .slice(0, 4)
    .map(h => h.textContent?.trim().replace(/^#\s*/, '') || '')
    .filter(Boolean)
  
  if (!excerpt) excerpt = '暂无摘要，点击进入阅读全文。'

  const data = { title, excerpt, headings }
  contentCache.set(url, data)
  return data
}

onMounted(() => {
  document.body.addEventListener('mouseover', handleMouseOver)
  document.body.addEventListener('mouseout', handleMouseOut)
})

onUnmounted(() => {
  document.body.removeEventListener('mouseover', handleMouseOver)
  document.body.removeEventListener('mouseout', handleMouseOut)
  if (hoverTimer) clearTimeout(hoverTimer)
  if (hideTimer) clearTimeout(hideTimer)
})
</script>

<template>
  <Teleport to="body">
    <Transition name="popover-spring">
      <div 
        v-if="visible"
        class="quantum-popover"
        :style="{ 
          left: `${position.x}px`, 
          top: `${position.y}px` 
        }"
        @mouseover="handlePopoverMouseOver"
        @mouseout="handlePopoverMouseOut"
      >
        <div class="popover-content" :class="{ loading }">
          <div class="popover-header">
            <div class="popover-title-row">
              <span class="icon">📄</span>
              <span class="title">{{ content.title }}</span>
            </div>
          </div>
          
          <div class="popover-body">
            <p class="excerpt">{{ content.excerpt }}</p>
            
            <div v-if="content.headings.length > 0" class="structure">
              <div class="structure-label">文章大纲</div>
              <ul class="heading-list">
                <li v-for="h in content.headings" :key="h">{{ h }}</li>
              </ul>
            </div>
          </div>
          
          <div class="popover-footer" v-if="!loading">
            <span class="read-more">点击跳转阅读全文 →</span>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.quantum-popover {
  position: fixed;
  z-index: 1000;
  width: 320px;
  max-width: 90vw;
  transform: translateX(-50%);
  pointer-events: auto; /* 允许鼠标进入卡片 */
}

/* Glassmorphism Card */
.popover-content {
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(20px) saturate(180%);
  -webkit-backdrop-filter: blur(20px) saturate(180%);
  border: 1px solid rgba(255, 255, 255, 0.4);
  border-radius: 12px;
  padding: 14px;
  box-shadow: 
    0 10px 25px rgba(0, 0, 0, 0.1),
    0 4px 10px rgba(0, 0, 0, 0.05);
  transition: all 0.3s ease;
  overflow: hidden;
}

.dark .popover-content {
  background: rgba(26, 26, 29, 0.85);
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
}

.popover-header {
  margin-bottom: 10px;
}

.popover-title-row {
  display: flex;
  align-items: flex-start;
  gap: 8px;
}

.popover-header .icon {
  font-size: 16px;
  flex-shrink: 0;
  margin-top: 1px;
}

.popover-header .title {
  font-weight: 600;
  font-size: 15px;
  line-height: 1.3;
  color: var(--vp-c-text-1);
}

.popover-body {
  font-size: 13px;
  line-height: 1.6;
}

.excerpt {
  color: var(--vp-c-text-2);
  margin-bottom: 12px;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.structure {
  background: var(--vp-c-bg-soft);
  padding: 8px 10px;
  border-radius: 6px;
  margin-bottom: 10px;
}

.structure-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--vp-c-text-3);
  margin-bottom: 4px;
  font-weight: 600;
}

.heading-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.heading-list li {
  font-size: 12px;
  color: var(--vp-c-text-2);
  position: relative;
  padding-left: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.heading-list li::before {
  content: '·';
  position: absolute;
  left: 4px;
  font-weight: bold;
}

.popover-footer {
  font-size: 12px;
  color: var(--vp-c-brand);
  font-weight: 500;
  text-align: left;
}

/* Spring Animation */
.popover-spring-enter-active {
  transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.15);
  transform-origin: top center;
}

.popover-spring-leave-active {
  transition: all 0.2s cubic-bezier(0.4, 0, 1, 1);
  transform-origin: top center;
}

.popover-spring-enter-from,
.popover-spring-leave-to {
  opacity: 0;
  transform: translateX(-50%) translateY(5px) scale(0.95);
}

.popover-spring-enter-to {
  opacity: 1;
  transform: translateX(-50%) translateY(0) scale(1);
}
</style>
