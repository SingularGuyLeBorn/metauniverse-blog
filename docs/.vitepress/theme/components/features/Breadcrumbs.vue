<script setup lang="ts">
import { useData, useRoute } from 'vitepress'
import { computed } from 'vue'

const { theme } = useData()
const route = useRoute()

const breadcrumbs = computed(() => {
  const path = route.path
  const parts = path.split('/').filter(p => p && !p.endsWith('.html'))
  
  // Home is implicit in the design now, or we can keep it as an icon
  const crumbs = [
    { text: 'Home', link: '/', icon: '🏠' }
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
        <a v-if="index < breadcrumbs.length - 1" :href="crumb.link">
           <span v-if="crumb.icon" class="icon">{{ crumb.icon }}</span>
           <span v-else>{{ crumb.text }}</span>
        </a>
        <span v-else class="current" aria-current="page">
          {{ crumb.text }}
        </span>
        <span v-if="index < breadcrumbs.length - 1" class="separator">
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>
        </span>
      </li>
    </ol>
  </nav>
</template>

<style scoped>
.breadcrumbs {
  margin: 1.5rem 0;
  padding: 0.6rem 1.2rem;
  font-size: 0.9rem;
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg-soft);
  border-radius: 16px;
  width: fit-content;
  border: 1px solid var(--vp-c-divider);
  backdrop-filter: blur(12px);
  transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
}

.breadcrumbs:hover {
  border-color: var(--vp-c-brand);
  transform: translateY(-1px);
  box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
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
  display: flex;
  align-items: center;
  gap: 4px;
  font-weight: 500;
  padding: 2px 6px;
  border-radius: 6px;
}

.breadcrumbs a:hover {
  color: var(--vp-c-brand);
  background-color: var(--vp-c-bg-mute);
}

.breadcrumbs .current {
  color: var(--vp-c-brand);
  font-weight: 600;
  padding: 2px 8px;
  background: var(--vp-c-brand-dimm);
  border-radius: 6px;
  border: 1px solid var(--vp-c-brand-soft);
}

.separator {
  margin: 0 4px;
  color: var(--vp-c-text-3);
  display: flex;
  align-items: center;
  opacity: 0.5;
}

.icon {
  font-size: 1.1em;
  margin-right: 2px;
}
</style>
