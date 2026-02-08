<template>
  <g class="neural-network-visual" :transform="`translate(${x}, ${y}) scale(${scale})`" :opacity="opacity">
    <!-- Title -->
    <text v-if="title" x="0" y="-80" text-anchor="middle" font-weight="bold" :fill="color" font-size="14">{{ title }}</text>
    
    <!-- Layers -->
    <g v-for="(layerSize, lIndex) in layers" :key="lIndex" :transform="`translate(${(lIndex - (layers.length-1)/2) * layerGap}, 0)`">
       <!-- Nodes -->
       <g v-for="nIndex in layerSize" :key="nIndex" :transform="`translate(0, ${(nIndex - (layerSize+1)/2) * nodeGap})`">
          <!-- Connection Lines to Next Layer -->
          <g v-if="lIndex < layers.length - 1">
             <line 
               v-for="nextIndex in layers[lIndex+1]" :key="nextIndex"
               :x1="0" :y1="0"
               :x2="layerGap" :y2="(nextIndex - (layers[lIndex+1]+1)/2) * nodeGap - (nIndex - (layerSize+1)/2) * nodeGap"
               :stroke="getLineColor(lIndex)"
               :stroke-width="getLineWidth(lIndex)"
               stroke-linecap="round"
               :opacity="0.2"
             />
          </g>
          
          <!-- Node Circle -->
          <circle 
            r="8" 
            :fill="getNodeColor(lIndex)" 
            :stroke="color"
            stroke-width="1.5"
            class="node"
            :class="{ pulse: activeLayer === lIndex || activeLayer === 99 }"
          />
       </g>
    </g>
  </g>
</template>

<script setup lang="ts">
import { computed } from 'vue';

const props = defineProps<{
  x: number;
  y: number;
  title?: string;
  layers: number[]; // e.g. [4, 5, 5, 1]
  color: string; // Hex color
  activeLayer?: number; // 0, 1, 2... or 99 for all
  opacity?: number;
  scale?: number;
}>();

const layerGap = 60;
const nodeGap = 30;

function getNodeColor(lIndex: number) {
   if (props.activeLayer === 99 || props.activeLayer === lIndex) return props.color;
   return '#FFFFFF';
}

function getLineColor(lIndex: number) {
   // Active flow logic
   if (props.activeLayer === lIndex) return props.color;
   return '#CBD5E1'; // Slate-300
}

function getLineWidth(lIndex: number) {
   if (props.activeLayer === lIndex) return 2;
   return 1;
}
</script>

<style scoped>
.neural-network-visual {
  transition: opacity 0.5s ease, transform 0.5s ease;
}

.node {
  transition: fill 0.3s ease;
}

.node.pulse {
  animation: pulse 1s infinite alternate;
}

@keyframes pulse {
  from { stroke-width: 1.5; r: 8; }
  to { stroke-width: 3; r: 9; }
}
</style>
