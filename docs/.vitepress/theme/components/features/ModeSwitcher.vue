<template>
  <div
    ref="switcherRef"
    class="mode-switcher"
    :class="{ 
      'is-experimental': isExperimentalMode,
      'is-minimized': isMinimized,
      'is-dragging': isDragging
    }"
    :style="positionStyle"
  >
    <!-- 最小化状态：只显示小图标 -->
    <button
      v-if="isMinimized"
      class="minimize-toggle"
      @click="isMinimized = false"
      title="展开模式切换器"
    >
      <span class="toggle-icon">⚙️</span>
    </button>

    <!-- 展开状态：完整切换界面 -->
    <template v-else>
      <!-- 拖动手柄 -->
      <div 
        class="drag-handle"
        @mousedown="startDrag"
        @touchstart="startDrag"
        title="拖动移动位置"
      >
        ⋮⋮
      </div>

      <button
        class="mode-btn mode-skimming"
        :class="{ active: isSkimmingMode }"
        @click="setReadingMode('skimming')"
        title="扫读模式 (Cmd/Ctrl+E)"
      >
        <span class="mode-icon">📖</span>
        <span class="mode-label">扫读</span>
      </button>

      <div class="mode-toggle" @click="toggleReadingMode">
        <div class="toggle-track">
          <div
            class="toggle-thumb"
            :class="{ 'is-right': isExperimentalMode }"
          ></div>
        </div>
      </div>

      <button
        class="mode-btn mode-experimental"
        :class="{ active: isExperimentalMode }"
        @click="setReadingMode('experimental')"
        title="实验模式 (Cmd/Ctrl+E)"
      >
        <span class="mode-icon">🔬</span>
        <span class="mode-label">实验</span>
      </button>

      <!-- 最小化按钮 -->
      <button
        class="minimize-btn"
        @click="isMinimized = true"
        title="最小化"
      >
        ✕
      </button>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from "vue";
import { useAppStore } from "../../stores/app";
import { storeToRefs } from "pinia";
import { useLocalStorage } from "@vueuse/core";

const appStore = useAppStore();
const { isSkimmingMode, isExperimentalMode } = storeToRefs(appStore);
const { setReadingMode, toggleReadingMode } = appStore;

// 最小化状态
const isMinimized = useLocalStorage("mu-mode-switcher-minimized", true);

// 拖动相关状态
const switcherRef = ref<HTMLElement | null>(null);
const isDragging = ref(false);
const position = useLocalStorage("mu-mode-switcher-position", { x: -1, y: -1 });

// 计算位置样式
const positionStyle = computed(() => {
  if (position.value.x === -1 || position.value.y === -1) {
    // 默认位置：右下角
    return {
      bottom: "2rem",
      right: "2rem",
    };
  }
  return {
    left: `${position.value.x}px`,
    top: `${position.value.y}px`,
    bottom: "auto",
    right: "auto",
  };
});

let startX = 0;
let startY = 0;
let initialX = 0;
let initialY = 0;

function startDrag(e: MouseEvent | TouchEvent) {
  if (!switcherRef.value) return;
  
  isDragging.value = true;
  
  const clientX = "touches" in e ? e.touches[0].clientX : e.clientX;
  const clientY = "touches" in e ? e.touches[0].clientY : e.clientY;
  
  startX = clientX;
  startY = clientY;
  
  const rect = switcherRef.value.getBoundingClientRect();
  initialX = rect.left;
  initialY = rect.top;
  
  document.addEventListener("mousemove", onDrag);
  document.addEventListener("mouseup", stopDrag);
  document.addEventListener("touchmove", onDrag);
  document.addEventListener("touchend", stopDrag);
  
  e.preventDefault();
}

function onDrag(e: MouseEvent | TouchEvent) {
  if (!isDragging.value) return;
  
  const clientX = "touches" in e ? e.touches[0].clientX : e.clientX;
  const clientY = "touches" in e ? e.touches[0].clientY : e.clientY;
  
  const deltaX = clientX - startX;
  const deltaY = clientY - startY;
  
  let newX = initialX + deltaX;
  let newY = initialY + deltaY;
  
  // 边界检查
  const maxX = window.innerWidth - 100;
  const maxY = window.innerHeight - 50;
  
  newX = Math.max(0, Math.min(newX, maxX));
  newY = Math.max(0, Math.min(newY, maxY));
  
  position.value = { x: newX, y: newY };
}

function stopDrag() {
  isDragging.value = false;
  document.removeEventListener("mousemove", onDrag);
  document.removeEventListener("mouseup", stopDrag);
  document.removeEventListener("touchmove", onDrag);
  document.removeEventListener("touchend", stopDrag);
}

onUnmounted(() => {
  stopDrag();
});
</script>

<style scoped>
.mode-switcher {
  position: fixed;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.5rem 1rem;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 2rem;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  z-index: 1000;
  transition: all 0.3s ease;
  user-select: none;
}

.mode-switcher.is-minimized {
  padding: 0.25rem;
  border-radius: 50%;
  width: 36px;
  height: 36px;
  justify-content: center;
}

.mode-switcher.is-dragging {
  cursor: grabbing;
  opacity: 0.9;
}

.mode-switcher.is-experimental {
  background: var(--vp-c-bg-soft);
  border-color: var(--vp-c-brand-1);
}

.drag-handle {
  cursor: grab;
  padding: 0.25rem;
  color: var(--vp-c-text-3);
  font-size: 0.75rem;
  letter-spacing: -2px;
  user-select: none;
}

.drag-handle:active {
  cursor: grabbing;
}

.minimize-toggle {
  background: transparent;
  border: none;
  cursor: pointer;
  font-size: 1rem;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}

.minimize-btn {
  background: transparent;
  border: none;
  cursor: pointer;
  color: var(--vp-c-text-3);
  font-size: 0.875rem;
  padding: 0.25rem;
  margin-left: 0.25rem;
  opacity: 0.6;
  transition: opacity 0.2s;
}

.minimize-btn:hover {
  opacity: 1;
}

.mode-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.75rem;
  background: transparent;
  border: none;
  border-radius: 1rem;
  cursor: pointer;
  transition: all 0.2s ease;
  color: var(--vp-c-text-2);
  font-family: inherit;
}

.mode-btn:hover {
  background: var(--vp-c-bg-soft);
}

.mode-btn.active {
  color: var(--vp-c-brand-1);
  font-weight: 600;
}

.mode-icon {
  font-size: 1.125rem;
}

.mode-label {
  font-size: 0.875rem;
}

.mode-toggle {
  width: 48px;
  height: 24px;
  cursor: pointer;
}

.toggle-track {
  width: 100%;
  height: 100%;
  background: var(--vp-c-bg-soft);
  border-radius: 12px;
  position: relative;
  transition: background 0.3s ease;
}

.toggle-thumb {
  position: absolute;
  top: 2px;
  left: 2px;
  width: 20px;
  height: 20px;
  background: var(--vp-c-text-1);
  border-radius: 50%;
  transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.toggle-thumb.is-right {
  transform: translateX(24px);
  background: var(--vp-c-brand-1);
}

@media (max-width: 768px) {
  .mode-switcher {
    padding: 0.5rem;
    gap: 0.5rem;
  }

  .mode-label {
    display: none;
  }

  .mode-btn {
    padding: 0.5rem;
  }
  
  .drag-handle {
    display: none;
  }
}
</style>
