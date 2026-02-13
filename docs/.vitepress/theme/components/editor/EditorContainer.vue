<script setup lang="ts">
import { ref } from 'vue'
import VditorEditor from './VditorEditor.vue';

const content = ref('# Hello Editor\n\nStart via slash command `/` or just type...')
</script>

<template>
  <div class="editor-container max-w-4xl mx-auto py-10 px-4">
    <div class="mb-6">
      <h1 class="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-purple-600 mb-2">
        WYSIWYG Markdown Editor
      </h1>
      <p class="text-zinc-500 dark:text-zinc-400">
        A block-based editor with markdown shortcuts.
      </p>
    </div>

    <VditorEditor v-model="content" height="600px" />

    <div class="mt-8 p-4 bg-zinc-100 dark:bg-zinc-800 rounded-lg">
      <h3 class="text-xs font-bold text-zinc-500 uppercase mb-2">Markdown Output</h3>
      <pre class="text-sm font-mono overflow-auto max-h-40 whitespace-pre-wrap">{{ content }}</pre>
    </div>
  </div>
</template>
