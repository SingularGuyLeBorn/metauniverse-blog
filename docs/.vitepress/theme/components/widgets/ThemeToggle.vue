<template>
  <button
    class="theme-toggle"
    :class="{ 'is-dark': isDark }"
    :title="title"
    @click="toggleTheme"
  >
    <Transition name="theme-icon" mode="out-in">
      <span v-if="isDark" key="sun" class="theme-toggle__icon">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <circle cx="12" cy="12" r="5" />
          <line x1="12" y1="1" x2="12" y2="3" />
          <line x1="12" y1="21" x2="12" y2="23" />
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
          <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
          <line x1="1" y1="12" x2="3" y2="12" />
          <line x1="21" y1="12" x2="23" y2="12" />
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
          <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
        </svg>
      </span>
      <span v-else key="moon" class="theme-toggle__icon">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
      </span>
    </Transition>
    <span class="theme-toggle__label">{{ label }}</span>
  </button>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { useAppStore } from "../../stores/app";
import { storeToRefs } from "pinia";

const appStore = useAppStore();
const { isDark, theme } = storeToRefs(appStore);
const { toggleTheme } = appStore;

const title = computed(() => {
  switch (theme.value) {
    case "light":
      return "当前：浅色模式";
    case "dark":
      return "当前：暗色模式";
    case "auto":
      return "当前：跟随系统";
    default:
      return "切换主题";
  }
});

const label = computed(() => {
  switch (theme.value) {
    case "light":
      return "浅色";
    case "dark":
      return "暗色";
    case "auto":
      return "自动";
    default:
      return "";
  }
});
</script>

<style scoped>
.theme-toggle {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.75rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 0.5rem;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  font-size: 0.875rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.theme-toggle:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}

.theme-toggle__icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
}

.theme-toggle__label {
  font-weight: 500;
}

.theme-icon-enter-active,
.theme-icon-leave-active {
  transition: all 0.2s ease;
}

.theme-icon-enter-from {
  opacity: 0;
  transform: rotate(-90deg) scale(0.5);
}

.theme-icon-leave-to {
  opacity: 0;
  transform: rotate(90deg) scale(0.5);
}

@media (max-width: 640px) {
  .theme-toggle__label {
    display: none;
  }
  .theme-toggle {
    padding: 0.5rem;
  }
}
</style>
