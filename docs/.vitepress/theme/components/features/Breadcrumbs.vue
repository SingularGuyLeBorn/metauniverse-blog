<script setup lang="ts">
import { useData, useRoute } from 'vitepress'
import { computed } from 'vue'

const { theme } = useData()
const route = useRoute()

const breadcrumbs = computed(() => {
  const path = route.path
  const parts = path.split('/').filter(p => p && !p.endsWith('.html'))
  
  const crumbs = [
    { text: '首页', link: '/' }
  ]

  let currentPath = '/'
  
  for (const part of parts) {
    currentPath += part + '/'
    const decodedPart = decodeURIComponent(part)
    
    // 尝试从 nav 配置中获取名称
    const navItem = theme.value.nav?.find((n: any) => n.link === currentPath)
    // 移除文件名后缀 .html
    const cleanName = decodedPart.replace(/\.html$/, '')
    const name = navItem ? navItem.text : (cleanName.charAt(0).toUpperCase() + cleanName.slice(1))
    
    crumbs.push({
      text: name,
      link: currentPath
    })
  }

  return crumbs
})
</script>

<template>
  <nav class="breadcrumbs" aria-label="breadcrumbs">
    <ol>
      <li v-for="(crumb, index) in breadcrumbs" :key="crumb.link">
        <a v-if="index < breadcrumbs.length - 1" :href="crumb.link">{{ crumb.text }}</a>
        <span v-else class="current" aria-current="page">{{ crumb.text }}</span>
        <span v-if="index < breadcrumbs.length - 1" class="separator">/</span>
      </li>
    </ol>
  </nav>
</template>

<style scoped>
.breadcrumbs {
  margin: 1rem 0 2rem 0;
  padding: 0.5rem 1rem;
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg-soft);
  border-radius: 12px;
  width: fit-content;
  border: 1px solid var(--vp-c-divider);
  backdrop-filter: blur(8px);
  transition: all 0.3s ease;
}

.breadcrumbs:hover {
  border-color: var(--vp-c-brand);
  background: var(--vp-c-bg-mute);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}

.breadcrumbs ol {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
}

.breadcrumbs li {
  display: flex;
  align-items: center;
}

.breadcrumbs a {
  color: var(--vp-c-text-2);
  text-decoration: none;
  transition: color 0.2s;
  position: relative;
  font-weight: 500;
}

.breadcrumbs a:hover {
  color: var(--vp-c-brand);
}

.breadcrumbs .current {
  color: var(--vp-c-text-1);
  font-weight: 600;
  background: var(--vp-c-brand-dimm);
  padding: 2px 8px;
  border-radius: 6px;
  color: var(--vp-c-brand-dark);
}

.separator {
  margin: 0;
  color: var(--vp-c-text-3);
  font-size: 0.8em;
}
</style>
