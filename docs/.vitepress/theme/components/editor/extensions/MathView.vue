<script setup lang="ts">
import { NodeViewWrapper, nodeViewProps } from '@tiptap/vue-3'
import { ref, computed, onMounted, watch, nextTick } from 'vue'
import katex from 'katex'

const props = defineProps(nodeViewProps)

const isEditing = ref(false)
const inputRef = ref<HTMLTextAreaElement | null>(null)
const previewRef = ref<HTMLElement | null>(null)

const src = computed({
  get: () => props.node.attrs.src,
  set: (value) => {
    props.updateAttributes({ src: value })
  },
})

const renderMath = () => {
  if (previewRef.value) {
    try {
      katex.render(src.value || 'Eq', previewRef.value, {
        throwOnError: false,
        displayMode: props.node.attrs.display === 'block',
      })
    } catch (e: any) {
      previewRef.value.textContent = e.message
    }
  }
}

const startEditing = () => {
  if (props.editor.isEditable) {
    isEditing.value = true
    nextTick(() => {
        inputRef.value?.focus()
    })
  }
}

const stopEditing = () => {
  isEditing.value = false
}

// Watch for changes and re-render
watch(src, renderMath)
onMounted(renderMath)

// Re-render when display mode changes
watch(() => props.node.attrs.display, renderMath)

</script>

<template>
  <node-view-wrapper class="math-node inline-block relative cursor-pointer" :class="{ 'is-editing': isEditing, 'is-block': node.attrs.display === 'block' }">
    <!-- Preview Mode -->
    <div
      v-show="!isEditing"
      ref="previewRef"
      class="math-preview px-1 rounded hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors"
      @click="startEditing"
    ></div>

    <!-- Edit Mode -->
    <div v-show="isEditing" class="math-edit flex items-center">
      <span class="text-zinc-400 select-none mr-1 font-mono text-sm">$$</span>
      <textarea
        ref="inputRef"
        v-model="src"
        class="bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-700 rounded px-1 font-mono text-sm focus:outline-none focus:border-blue-500 min-w-[100px] resize-x overflow-hidden h-[1.5em]"
        @blur="stopEditing"
        @keydown.enter.stop
      />
      <span class="text-zinc-400 select-none ml-1 font-mono text-sm">$$</span>
    </div>
  </node-view-wrapper>
</template>

<style scoped>
.math-node.is-block {
  display: flex !important;
  justify-content: center;
  width: 100%;
  margin: 1em 0;
}
.math-edit textarea {
    field-sizing: content; /* New CSS property for auto-resize */
}
</style>
