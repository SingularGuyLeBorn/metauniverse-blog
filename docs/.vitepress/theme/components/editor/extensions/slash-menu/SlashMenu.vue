<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted } from 'vue'

const props = defineProps({
  items: {
    type: Array,
    required: true,
  },
  command: {
    type: Function,
    required: true,
  },
})

const selectedIndex = ref(0)

const onKeyDown = ({ event }: { event: KeyboardEvent }) => {
  if (event.key === 'ArrowUp') {
    upHandler()
    return true
  }

  if (event.key === 'ArrowDown') {
    downHandler()
    return true
  }

  if (event.key === 'Enter') {
    enterHandler()
    return true
  }

  return false
}

const upHandler = () => {
  selectedIndex.value = (selectedIndex.value + props.items.length - 1) % props.items.length
}

const downHandler = () => {
  selectedIndex.value = (selectedIndex.value + 1) % props.items.length
}

const enterHandler = () => {
  selectItem(selectedIndex.value)
}

const selectItem = (index: number) => {
  const item = props.items[index] as any
  if (item) {
    props.command(item)
  }
}

watch(
  () => props.items,
  () => {
    selectedIndex.value = 0
  },
)

defineExpose({
  onKeyDown,
})
</script>

<template>
  <div class="slash-menu-popup bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 shadow-xl rounded-lg overflow-hidden min-w-[200px] max-h-[300px] overflow-y-auto p-1">
    <div v-if="items.length">
      <button
        v-for="(item, index) in items"
        :key="index"
        :class="{ 'is-selected': index === selectedIndex }"
        class="flex items-center w-full px-2 py-1.5 text-sm text-left hover:bg-zinc-100 dark:hover:bg-zinc-700 rounded transition-colors"
        @click="selectItem(index)"
      >
        <span v-if="(item as any).icon" class="mr-2 text-zinc-500 dark:text-zinc-400 w-5 h-5 flex items-center justify-center">
            <!-- Icon placeholder or SVG -->
            {{ (item as any).icon }}
        </span>
        <div class="flex flex-col">
            <span class="font-medium text-zinc-800 dark:text-zinc-200">{{ (item as any).title }}</span>
            <span v-if="(item as any).description" class="text-xs text-zinc-500 dark:text-zinc-400">{{ (item as any).description }}</span>
        </div>
      </button>
    </div>
    <div v-else class="text-sm text-zinc-500 dark:text-zinc-400 p-2 text-center">
      No result
    </div>
  </div>
</template>

<style scoped>
.is-selected {
  @apply bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400;
}
</style>
