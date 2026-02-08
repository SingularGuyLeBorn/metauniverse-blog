<template>
  <g class="state-grid-visual" :transform="`translate(${x}, ${y})`">
    <!-- Grid Container -->
    <rect 
      :x="-width/2" 
      :y="-height/2"
      :width="width" 
      :height="height"
      rx="12"
      fill="#ECFDF5"
      stroke="#A7F3D0"
      stroke-width="1"
    />
    
    <!-- Grid Cells -->
    <g v-for="r in rows" :key="'row-'+r">
        <g v-for="c in cols" :key="'col-'+c" :transform="`translate(${getCellX(c)}, ${getCellY(r)})`">
            <!-- Cell Rect -->
            <rect 
                :x="-cellCallback(r,c).w/2" 
                :y="-cellCallback(r,c).h/2" 
                :width="cellCallback(r,c).w" 
                :height="cellCallback(r,c).h" 
                rx="4"
                :fill="getCellColor(r, c)"
                :stroke="getCellStroke(r, c)"
                stroke-width="1.5"
                class="grid-cell"
                :class="{ 
                   active: isCurrent(r, c), 
                   next: isNext(r, c),
                   breathing: !active && !isCurrent(r,c) && !isNext(r,c) // Breathing idle
                }"
            />
            <!-- Label for Active/Next -->
            <text v-if="isCurrent(r,c)" y="5" font-size="14" text-anchor="middle" fill="#059669" font-weight="bold" class="fade-in">sₜ</text>
            <text v-if="isNext(r,c)" y="5" font-size="14" text-anchor="middle" fill="#059669" font-weight="bold" class="fade-in">sₜ₊₁</text>
        </g>
    </g>
    
    <!-- Transition Arrow (Curve from s_t to s_t+1) -->
    <path v-if="showTransition"
          :d="getTransitionPath()"
          fill="none" 
          stroke="#34D399" 
          stroke-width="3" 
          marker-end="url(#arrow-sm)"
          class="transition-arrow"
          filter="url(#glass)"
    />
  </g>
</template>

<script setup lang="ts">
import { computed } from 'vue';

const props = defineProps<{
  x: number;
  y: number;
  width?: number;
  height?: number;
  rows?: number;
  cols?: number;
  currentState?: { r: number; c: number }; // e.g. {r: 1, c: 1}
  nextState?: { r: number; c: number };    // e.g. {r: 1, c: 2}
  active?: boolean;
}>();

const rows = computed(() => props.rows || 4);
const cols = computed(() => props.cols || 4);
const width = computed(() => props.width || 320);
const height = computed(() => props.height || 320);

// Helper for cell dimensions
const cellCallback = (r: number, c: number) => {
    const w = (width.value - (cols.value + 1) * 10) / cols.value;
    const h = (width.value - (rows.value + 1) * 10) / rows.value; // Square cells based on width
    return { w, h };
};

function getCellX(c: number): number {
    const step = width.value / cols.value;
    return -width.value / 2 + step * c - step / 2 + 5; // Adjust centering
}

function getCellY(r: number): number {
    const step = height.value / rows.value;
    return -height.value / 2 + step * r - step / 2 + 5;
}

function isCurrent(r: number, c: number): boolean {
    return props.currentState?.r === r && props.currentState?.c === c;
}

function isNext(r: number, c: number): boolean {
    return props.nextState?.r === r && props.nextState?.c === c;
}

function getCellColor(r: number, c: number): string {
    if (isCurrent(r, c)) return '#ECFDF5'; // Active bg
    if (isNext(r, c)) return '#ECFDF5';    
    return '#FFFFFF';
}

function getCellStroke(r: number, c: number): string {
    if (isCurrent(r, c)) return '#34D399'; // Emerald 400
    if (isNext(r, c)) return '#34D399';
    return '#D1FAE5'; // Emerald 100
}

const showTransition = computed(() => props.currentState && props.nextState && props.active);

function getTransitionPath(): string {
    if (!props.currentState || !props.nextState) return '';
    const startX = getCellX(props.currentState.c);
    const startY = getCellY(props.currentState.r);
    const endX = getCellX(props.nextState.c);
    const endY = getCellY(props.nextState.r);
    
    // Simple curve
    const midX = (startX + endX) / 2;
    const midY = (startY + endY) / 2 - 40; // Higher arc
    
    return `M ${startX} ${startY} Q ${midX} ${midY} ${endX} ${endY}`;
}
</script>

<style scoped>
.grid-cell {
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}
.grid-cell.active {
    filter: drop-shadow(0 0 8px rgba(52, 211, 153, 0.4));
    fill: #D1FAE5; /* Emerald 100 */
}
.grid-cell.breathing {
    animation: breathe 3s ease-in-out infinite;
    transform-origin: center;
}

@keyframes breathe {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.02); }
}

.transition-arrow {
    stroke-dasharray: 200;
    stroke-dashoffset: 200;
    animation: drawArrow 1s ease-out forwards;
}

@keyframes drawArrow {
    to { stroke-dashoffset: 0; }
}

.fade-in {
    animation: fadeIn 0.5s ease-out forwards;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(5px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
