<script setup lang="ts">
import { useData } from 'vitepress'
import { computed } from 'vue'

const { frontmatter, page } = useData()

const author = computed(() => frontmatter.value.author || 'MetaUniverse')
const date = computed(() => {
  const d = new Date(frontmatter.value.date || page.value.lastUpdated || Date.now())
  return d.toLocaleDateString('zh-CN', { year: 'numeric', month: 'long', day: 'numeric' })
})

const wordCount = computed(() => frontmatter.value.wordCount || 0)
const readingTime = computed(() => frontmatter.value.readingTime || 1)
</script>

<template>
  <div class="article-metadata">
    <div class="meta-item author">
      <span class="icon">👤</span>
      <span>{{ author }}</span>
    </div>
    <div class="meta-item date">
      <span class="icon">📅</span>
      <span>{{ date }}</span>
    </div>
    <div class="meta-item words">
      <span class="icon">📝</span>
      <span>{{ wordCount }} 字</span>
    </div>
    <div class="meta-item time">
      <span class="icon">⏱️</span>
      <span>{{ readingTime }} 分钟</span>
    </div>
  </div>
</template>

<style scoped>
.article-metadata {
  display: flex;
  flex-wrap: wrap;
  gap: 1.5rem;
  margin-bottom: 2rem;
  padding: 1rem 0;
  border-bottom: 1px solid var(--vp-c-divider);
  color: var(--vp-c-text-2);
  font-size: 0.9rem;
  font-family: var(--vp-font-family-mono);
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.icon {
  opacity: 0.8;
}

@media (max-width: 640px) {
  .article-metadata {
    gap: 1rem;
    font-size: 0.8rem;
  }
}
</style>
