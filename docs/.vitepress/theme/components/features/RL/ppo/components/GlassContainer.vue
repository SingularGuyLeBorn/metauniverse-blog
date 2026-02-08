<template>
  <g class="glass-container" :opacity="opacity">
    <!-- Main Box -->
    <rect 
      :x="-w/2" :y="-h/2" 
      :width="w" :height="h" 
      :rx="12" 
      fill="rgba(255,255,255,0.9)" 
      :stroke="color" 
      stroke-width="1"
      filter="url(#soft-shadow)"
    />
    
    <!-- Header Label -->
    <text 
      :y="-h/2 - 10" 
      text-anchor="middle" 
      :fill="color" 
      font-weight="bold" 
      font-size="14"
    >{{ label }}</text>
    
    <!-- Cells -->
    <g transform="translate(0, 5)"> 
       <!-- 4 Cells centered -->
       <g v-for="(val, i) in values" :key="i" :transform="`translate(${(i - 1.5) * (cellW + 4)}, 0)`">
           <rect 
             :x="-cellW/2" :y="-cellH/2" 
             :width="cellW" :height="cellH" 
             rx="6" 
             :fill="getCellBg(i)" 
           />
           <text 
             y="4" 
             text-anchor="middle" 
             font-size="12" 
             :fill="color"
             font-family="monospace"
           >{{ val }}</text>
       </g>
    </g>
  </g>
</template>

<script setup lang="ts">
import { computed } from 'vue';

const props = defineProps<{
  label: string;
  values: (number | string)[];
  color: string;
  w: number;
  h: number;
  type?: string; // 'action' or 'state'
  opacity?: number;
}>();

const cellW = computed(() => (props.w - 40) / 4);
const cellH = computed(() => props.h - 30);

function getCellBg(i: number) {
    if (props.type === 'action' && i === 1) return props.color; // Highlight Action 1 (index 1)
    if (props.type === 'action') return '#F1F5F9';
    return adjustColorOpacity(props.color, 0.1);
}

function adjustColorOpacity(hex: string, alpha: number) {
    // Simple mock for hex to rgba, assume strict hex input
    // Just return a simplified version or the hex itself for now if alpha not easy
    // Better: use explicit colors in parent or valid implementation.
    // For now: 
    return hex + '1A'; // 10%
}
</script>

<style scoped>
.glass-container {
  transition: opacity 0.3s;
}
</style>
