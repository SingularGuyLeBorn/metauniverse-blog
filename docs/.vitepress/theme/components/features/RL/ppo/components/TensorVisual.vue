<template>
  <g class="tensor-visual" :transform="`translate(${x}, ${y})`">
    <!-- Container -->
    <rect 
      :x="-totalWidth/2" 
      :y="-cellHeight/2 - 8"
      :width="totalWidth" 
      :height="cellHeight + 16"
      rx="6"
      :fill="bgColor"
      :stroke="strokeColor"
      stroke-width="2"
    />
    
    <!-- Label above -->
    <text 
      :y="-cellHeight/2 - 18" 
      text-anchor="middle" 
      class="tensor-label"
      :fill="strokeColor">{{ label }}</text>
    
    <!-- Cells -->
    <g v-for="(cell, i) in cells" :key="i" 
       :transform="`translate(${getCellX(i)}, 0)`">
      <rect 
        :x="-cellWidth/2 + 2" 
        :y="-cellHeight/2 + 2"
        :width="cellWidth - 4" 
        :height="cellHeight - 4"
        rx="3"
        :fill="getCellColor(i)"
      />
      <text 
        y="4" 
        text-anchor="middle" 
        class="cell-label"
        :fill="textColor">{{ cell }}</text>
    </g>
  </g>
</template>

<script setup lang="ts">
import { computed } from 'vue';

const props = defineProps<{
  x: number;
  y: number;
  label: string; // e.g., "sₜ" or "aₜ"
  cells: string[]; // e.g., ["s₁", "s₂", "s₃", "s₄"]
  color?: 'blue' | 'green' | 'yellow' | 'purple';
  cellWidth?: number;
  cellHeight?: number;
}>();

const cellWidth = computed(() => props.cellWidth || 32);
const cellHeight = computed(() => props.cellHeight || 28);
const totalWidth = computed(() => props.cells.length * cellWidth.value + 12);

const colorSchemes = {
  blue: { bg: '#dbeafe', stroke: '#3b82f6', cells: ['#93c5fd', '#60a5fa', '#3b82f6', '#2563eb'], text: '#1e40af' },
  green: { bg: '#d1fae5', stroke: '#10b981', cells: ['#6ee7b7', '#34d399', '#10b981', '#059669'], text: '#065f46' },
  yellow: { bg: '#fef3c7', stroke: '#f59e0b', cells: ['#fcd34d', '#fbbf24', '#f59e0b', '#d97706'], text: '#92400e' },
  purple: { bg: '#ede9fe', stroke: '#8b5cf6', cells: ['#c4b5fd', '#a78bfa', '#8b5cf6', '#7c3aed'], text: '#5b21b6' }
};

const scheme = computed(() => colorSchemes[props.color || 'blue']);
const bgColor = computed(() => scheme.value.bg);
const strokeColor = computed(() => scheme.value.stroke);
const textColor = computed(() => scheme.value.text);

function getCellX(index: number): number {
  const startX = -totalWidth.value / 2 + cellWidth.value / 2 + 6;
  return startX + index * cellWidth.value;
}

function getCellColor(index: number): string {
  return scheme.value.cells[index % scheme.value.cells.length];
}
</script>

<style scoped>
.tensor-label {
  font-size: 12px;
  font-weight: 700;
}

.cell-label {
  font-size: 10px;
  font-weight: 600;
}
</style>
