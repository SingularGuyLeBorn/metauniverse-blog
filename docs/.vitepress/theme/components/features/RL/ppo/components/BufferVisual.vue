<template>
  <g class="buffer-visual" :transform="`translate(${x}, ${y})`">
    <!-- Container -->
    <rect 
      :x="-width/2" 
      :y="-height/2"
      :width="width" 
      :height="height"
      rx="8"
      fill="#fffbeb"
      stroke="#f59e0b"
      stroke-width="2"
    />
    
    <!-- Title -->
    <text :x="-width/2 + 12" :y="-height/2 + 18" class="buffer-title">
      EXPERIENCE BUFFER
    </text>
    
    <!-- Tuple rows -->
    <g v-for="(tuple, ti) in displayTuples" :key="ti"
       :transform="`translate(0, ${getTupleY(ti)})`"
       class="tuple-row"
       :class="{ newest: ti === 0 }">
      
      <!-- 5 cells for tuple -->
      <g v-for="(cell, ci) in tuple" :key="ci">
        <rect 
          :x="getCellX(ci)"
          :y="-10"
          :width="cellWidth - 4"
          :height="20"
          rx="3"
          :fill="getCellColor(ci)"
          stroke="#e2e8f0"
        />
        <text 
          :x="getCellX(ci) + cellWidth/2 - 2"
          y="4"
          text-anchor="middle"
          class="cell-text">{{ cell }}</text>
      </g>
    </g>
    
    <!-- Count indicator -->
    <text :x="width/2 - 12" :y="-height/2 + 18" text-anchor="end" class="count-text">
      {{ tuples.length }} tuples
    </text>
    
    <!-- Empty state -->
    <text v-if="tuples.length === 0" y="10" text-anchor="middle" class="empty-text">
      Empty
    </text>
  </g>
</template>

<script setup lang="ts">
import { computed } from 'vue';

interface Tuple {
  s: string;
  a: string;
  r: string;
  s_next: string;
  logp: string;
}

const props = defineProps<{
  x: number;
  y: number;
  width?: number;
  height?: number;
  tuples: Tuple[];
}>();

const width = computed(() => props.width || 350);
const height = computed(() => props.height || 100);
const maxDisplay = 3;
const cellWidth = computed(() => (width.value - 40) / 5);

const displayTuples = computed(() => {
  // Show newest tuples first, max 3
  return props.tuples.slice(-maxDisplay).reverse().map(t => [t.s, t.a, t.r, t.s_next, t.logp]);
});

function getTupleY(index: number): number {
  return -height.value / 2 + 45 + index * 25;
}

function getCellX(index: number): number {
  return -width.value / 2 + 20 + index * cellWidth.value;
}

const cellColors = ['#dbeafe', '#d1fae5', '#fef3c7', '#e0e7ff', '#fce7f3'];

function getCellColor(index: number): string {
  return cellColors[index];
}
</script>

<style scoped>
.buffer-title {
  font-size: 11px;
  font-weight: 700;
  fill: #d97706;
  letter-spacing: 0.5px;
}

.count-text {
  font-size: 10px;
  fill: #92400e;
}

.cell-text {
  font-size: 9px;
  fill: #475569;
  font-weight: 600;
}

.empty-text {
  font-size: 12px;
  fill: #94a3b8;
  font-style: italic;
}

.tuple-row {
  opacity: 0.8;
  transition: opacity 0.3s;
}

.tuple-row.newest {
  opacity: 1;
}
</style>
