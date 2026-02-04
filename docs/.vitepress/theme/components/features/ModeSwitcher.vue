<template>
  <div class="mode-switcher" :class="{ 'is-experimental': isExperimentalMode }">
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
  </div>
</template>

<script setup lang="ts">
import { useAppStore } from "../../stores/app";
import { storeToRefs } from "pinia";

const appStore = useAppStore();
const { isSkimmingMode, isExperimentalMode } = storeToRefs(appStore);
const { setReadingMode, toggleReadingMode } = appStore;
</script>

<style scoped>
.mode-switcher {
  position: fixed;
  bottom: 2rem;
  right: 2rem;
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
}

.mode-switcher.is-experimental {
  background: var(--vp-c-bg-soft);
  border-color: var(--vp-c-brand-1);
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
    bottom: 1rem;
    right: 1rem;
    padding: 0.5rem;
    gap: 0.5rem;
  }

  .mode-label {
    display: none;
  }

  .mode-btn {
    padding: 0.5rem;
  }
}
</style>
