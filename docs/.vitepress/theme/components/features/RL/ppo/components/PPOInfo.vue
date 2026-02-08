<template>
  <div class="info-card">
    <div v-if="title" class="relation-title">{{ title }}</div>
    <div v-if="formula" class="relation-formula" v-html="renderMath(formula)"></div>
    <div v-if="desc" class="relation-desc">{{ desc }}</div>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue';

const props = defineProps<{
  title?: string;
  formula?: string;
  desc?: string;
}>();

const renderMath = (latex?: string) => {
  if (!latex) return '';
  // @ts-ignore
  if (window.katex) {
    try {
      // @ts-ignore
      return window.katex.renderToString(latex, { throwOnError: false, displayMode: false });
    } catch (e) { return latex; }
  }
  return latex;
};
</script>

<style scoped>
.info-card {
  background: white;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 16px;
  min-height: 100px;
}

.relation-title {
  font-size: 0.9rem;
  font-weight: 700;
  color: #1e293b;
  margin-bottom: 8px;
}

.relation-formula {
  font-size: 1.1rem;
  color: #3b82f6;
  margin-bottom: 8px;
  padding: 8px;
  background: #eff6ff;
  border-radius: 6px;
  text-align: center;
}

.relation-desc {
  font-size: 0.85rem;
  color: #64748b;
  line-height: 1.5;
}
</style>
