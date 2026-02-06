<template>
  <div class="article-index">
    <div v-for="section in sections" :key="section.title" class="section-group">
      <h2 v-if="section.title" class="section-title">{{ section.title }}</h2>
      <div class="card-grid">
        <a 
          v-for="article in section.items" 
          :key="article.link" 
          :href="article.link"
          class="article-card"
        >
          <div class="card-content">
            <h3 class="card-title">{{ article.text }}</h3>
            <p v-if="article.description" class="card-desc">{{ article.description }}</p>
            <div class="card-meta">
              <span class="read-more">Read more →</span>
            </div>
          </div>
        </a>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  sections: {
    title?: string
    items: {
      text: string
      link: string
      description?: string
    }[]
  }[]
}>()
</script>

<style scoped>
.article-index {
  padding: 2rem 0;
}

.section-title {
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 1.5rem;
  border-bottom: 1px solid var(--vp-c-divider);
  padding-bottom: 0.5rem;
}

.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 1.5rem;
  margin-bottom: 3rem;
}

.article-card {
  display: block;
  background-color: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-bg-soft);
  border-radius: 12px;
  text-decoration: none !important;
  transition: all 0.3s ease;
  overflow: hidden;
}

.article-card:hover {
  transform: translateY(-4px);
  border-color: var(--vp-c-brand);
  background-color: var(--vp-c-bg-mute);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}

.card-content {
  padding: 1.5rem;
}

.card-title {
  margin: 0 0 0.5rem 0 !important;
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.card-desc {
  font-size: 0.9rem;
  color: var(--vp-c-text-2);
  margin: 0 0 1rem 0;
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.read-more {
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--vp-c-brand);
}
</style>
