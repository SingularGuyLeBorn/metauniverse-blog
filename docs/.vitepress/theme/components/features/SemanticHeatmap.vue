<template>
  <div class="semantic-heatmap" v-if="heatmapData.length > 0">
    <div class="heatmap-header">
      <span class="heatmap-icon">🔥</span>
      <span class="heatmap-title">阅读热力</span>
    </div>
    <div class="heatmap-bar">
      <div
        v-for="segment in heatmapData"
        :key="segment.id"
        class="heatmap-segment"
        :style="{
          height: segment.height + 'px',
          backgroundColor: getHeatColor(segment.intensity),
        }"
        :title="`阅读时间: ${Math.round(segment.time / 1000)}秒`"
      ></div>
    </div>
    <div class="heatmap-legend">
      <span class="legend-cold">冷</span>
      <div class="legend-gradient"></div>
      <span class="legend-hot">热</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch, computed } from "vue";
import { useData } from "vitepress";
import { useReadingTracker } from "../../composables/useReadingTracker";

interface HeatmapSegment {
  id: string;
  intensity: number;
  time: number;
  views: number;
  height: number;
}

const { page } = useData();
const heatmapData = ref<HeatmapSegment[]>([]);

let tracker: ReturnType<typeof useReadingTracker> | null = null;

const getHeatColor = (intensity: number): string => {
  const colors = [
    { stop: 0, color: [59, 130, 246] }, // blue
    { stop: 0.3, color: [34, 197, 94] }, // green
    { stop: 0.6, color: [234, 179, 8] }, // yellow
    { stop: 1, color: [239, 68, 68] }, // red
  ];

  for (let i = 0; i < colors.length - 1; i++) {
    const curr = colors[i];
    const next = colors[i + 1];

    if (intensity >= curr.stop && intensity <= next.stop) {
      const t = (intensity - curr.stop) / (next.stop - curr.stop);
      const r = Math.round(curr.color[0] + (next.color[0] - curr.color[0]) * t);
      const g = Math.round(curr.color[1] + (next.color[1] - curr.color[1]) * t);
      const b = Math.round(curr.color[2] + (next.color[2] - curr.color[2]) * t);
      return `rgba(${r}, ${g}, ${b}, ${0.4 + intensity * 0.6})`;
    }
  }

  return "rgba(239, 68, 68, 1)";
};

function updateHeatmap() {
  if (!tracker) return;

  const data = tracker.getHeatmapData();
  const totalHeight = 200;
  const segmentHeight = totalHeight / Math.max(data.length, 1);

  heatmapData.value = data.map((d) => ({
    ...d,
    height: segmentHeight,
  }));
}

onMounted(() => {
  const articleId = page.value.relativePath;
  tracker = useReadingTracker(articleId);

  // 每5秒更新热力图
  const interval = setInterval(updateHeatmap, 5000);

  // 初始更新
  setTimeout(updateHeatmap, 2000);

  return () => clearInterval(interval);
});
</script>

<style scoped>
.semantic-heatmap {
  position: fixed;
  right: 1rem;
  top: 50%;
  transform: translateY(-50%);
  width: 30px;
  background: var(--vp-c-bg-soft);
  border-radius: 0.5rem;
  padding: 0.5rem;
  z-index: 50;
  opacity: 0.8;
  transition: opacity 0.2s;
}

.semantic-heatmap:hover {
  opacity: 1;
}

.heatmap-header {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-bottom: 0.5rem;
}

.heatmap-icon {
  font-size: 0.875rem;
}

.heatmap-title {
  font-size: 0.5rem;
  color: var(--vp-c-text-2);
  writing-mode: vertical-rl;
  text-orientation: mixed;
}

.heatmap-bar {
  display: flex;
  flex-direction: column;
  width: 12px;
  margin: 0 auto;
  border-radius: 3px;
  overflow: hidden;
}

.heatmap-segment {
  width: 100%;
  transition: background-color 0.3s ease;
  cursor: pointer;
}

.heatmap-segment:hover {
  opacity: 0.8;
}

.heatmap-legend {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-top: 0.5rem;
  font-size: 0.5rem;
  color: var(--vp-c-text-3);
}

.legend-gradient {
  width: 8px;
  height: 30px;
  margin: 0.25rem 0;
  background: linear-gradient(to bottom, #ef4444, #eab308, #22c55e, #3b82f6);
  border-radius: 2px;
}

@media (max-width: 1200px) {
  .semantic-heatmap {
    display: none;
  }
}
</style>
