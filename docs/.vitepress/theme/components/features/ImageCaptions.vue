<script setup lang="ts">
import { onMounted, onUnmounted, watch } from 'vue'
import { useRoute } from 'vitepress'

const route = useRoute()

const updateCaptions = () => {
  if (typeof window === 'undefined') return

  // 这里的延时是为了等待 VitePress 渲染内容完毕
  setTimeout(() => {
    const images = document.querySelectorAll('.vp-doc img')
    
    images.forEach((img) => {
      const imageElement = img as HTMLImageElement
      const alt = imageElement.alt
      
      // 如果没有 alt 或者已经是 figure 的一部分，则跳过
      if (!alt || imageElement.parentElement?.tagName === 'FIGURE') return
      
      // 检查是否已经添加过 caption
      if (imageElement.nextElementSibling?.classList.contains('image-caption')) return

      // 创建 caption 元素
      const caption = document.createElement('div')
      caption.className = 'image-caption'
      caption.textContent = alt
      
      // 插入到图片后面
      imageElement.parentNode?.insertBefore(caption, imageElement.nextSibling)
    })
  }, 300)
}

onMounted(() => {
  updateCaptions()
})

watch(
  () => route.path,
  () => {
    updateCaptions()
  }
)
</script>

<template>
  <slot />
</template>

<style>
/* 全局样式，因为是插入到 DOM 中的 */
.image-caption {
  text-align: center;
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
  margin-top: 0.5rem;
  margin-bottom: 2rem;
  font-style: italic;
  opacity: 0.8;
  font-family: var(--vp-font-family-mono);
}
</style>
