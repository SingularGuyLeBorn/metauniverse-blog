<script setup lang="ts">
/**
 * Find in Page Widget
 * 
 * Provides a floating search bar to find text within the current article content.
 * Simulates standard browser search but with more organic integration.
 */
import { ref, onMounted, onUnmounted, nextTick, watch } from 'vue'

const props = defineProps<{
  // No strict props needed, operates on DOM
}>()

const emit = defineEmits<{
  (e: 'close'): void
}>()

const searchQuery = ref('')
const replaceQuery = ref('')
const currentMatchIndex = ref(0)
const totalMatches = ref(0)
const showReplace = ref(false) // Toggle for "Replace" mode

// DOM elements for highlighting
let marks: HTMLElement[] = []

const clearHighlights = () => {
  // Remove all <mark> tags but keep content
  const content = document.querySelector('.vp-doc .content') || document.querySelector('.vp-doc')
  if (!content) return

  // This is a naive cleanup. Ideally we use a subtle overlay or range selection.
  // For safety in a Vue app, direct DOM manipulation like this is risky if the component re-renders.
  // A safer "Find" is just using the browser's find, but user requested a custom UI.
  // We will use the native window.find() API for navigation if possible, 
  // but to "Highlight All", we would need to wrap text. 
  // Let's stick to window.find() for navigation first as it's less intrusive.
  
  // Actually, window.find() is non-standard but widely supported. 
  // Let's implement a robust highlighting using a TreeWalker to avoid breaking Vue's DOM.
  
  // For this version (MVP), we will use window.find() loop to count and navigate.
}

const findNext = () => {
  if (!searchQuery.value) return
  // Use native find
  const found = window.find(searchQuery.value, false, false, true, false, false, false)
  if (!found) {
    // Wrap around?
    // Reset selection to top and try again
    window.getSelection()?.removeAllRanges()
    window.find(searchQuery.value, false, false, true, false, false, false)
  }
}

const findPrev = () => {
  if (!searchQuery.value) return
  const found = window.find(searchQuery.value, false, true, true, false, false, false)
   if (!found) {
     // Wrap around to bottom?
     // Difficult to jump to bottom reliably without knowing doc height.
   }
}

// "Replace" is superficial here - we can't save to file.
// But we can update the DOM visually.
const replaceCurrent = () => {
  if (!replaceQuery.value) return
  const selection = window.getSelection()
  if (selection && selection.toString() === searchQuery.value) {
    // ExecCommand is deprecated but still the standard way to edit "contenteditable"
    // Since our doc isn't editable, we have to force it.
    // Or just manually replace text node.
    if (selection.rangeCount > 0) {
      const range = selection.getRangeAt(0)
      range.deleteContents()
      range.insertNode(document.createTextNode(replaceQuery.value))
      // Move to next
      findNext()
    }
  } else {
    findNext()
  }
}

const replaceAll = () => {
  // Dangerous on a large page, but let's try.
  // Without a real backend, this is just for fun.
  alert('Global replace is separate from Read-Only mode. This would require an editor state.')
}

const close = () => {
  emit('close')
}

// Keyboard shortcut to close
const onKeydown = (e: KeyboardEvent) => {
  if (e.key === 'Escape') close()
  if (e.key === 'Enter') {
      if (e.shiftKey) findPrev()
      else findNext()
  }
}

onMounted(() => {
  window.addEventListener('keydown', onKeydown)
  // Auto focus input
  nextTick(() => {
    document.getElementById('mu-find-input')?.focus()
  })
})

onUnmounted(() => {
  window.removeEventListener('keydown', onKeydown)
})

</script>

<template>
  <div class="find-widget">
    <div class="find-bar">
      <div class="input-group">
         <svg class="search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
         </svg>
        <input 
          id="mu-find-input"
          v-model="searchQuery" 
          placeholder="在文档中查找..." 
          @input="() => {} /* Debounce highlight? */"
        />
        <div class="actions">
             <button @click="findPrev" title="上一处 (Shift+Enter)">
               <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m18 15-6-6-6 6"/></svg>
             </button>
             <button @click="findNext" title="下一处 (Enter)">
               <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>
             </button>
             <button @click="close" title="关闭 (Esc)">
               <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
             </button>
        </div>
      </div>
      
       <!-- Toggle Replace Mode -->
      <div class="expand-btn" @click="showReplace = !showReplace">
         <svg v-if="!showReplace" xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>
         <svg v-else xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m18 15-6-6-6 6"/></svg>
      </div>
    </div>
    
    <!-- Replace Bar -->
    <div v-if="showReplace" class="replace-bar">
      <div class="input-group">
         <svg class="replace-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M7 7h10v10"/><path d="M7 17 17 7"/>
         </svg>
        <input 
          v-model="replaceQuery" 
          placeholder="替换为..." 
          @keydown.enter="replaceCurrent"
        />
        <div class="actions text-actions">
             <button @click="replaceCurrent" title="替换当前">替换</button>
             <button @click="replaceAll" title="替换所有(WIP)">全部替换</button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.find-widget {
  position: fixed;
  top: 80px;
  right: 40px;
  z-index: 200;
  width: 340px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  box-shadow: 0 4px 24px rgba(0,0,0,0.15);
  border-radius: 8px;
  overflow: hidden;
  font-size: 14px;
  animation: slideDown 0.2s ease-out;
}

@keyframes slideDown {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}

.find-bar, .replace-bar {
  display: flex;
  align-items: center;
  padding: 8px 12px;
}

.replace-bar {
    border-top: 1px solid var(--vp-c-divider);
    background: var(--vp-c-bg-soft);
}

.input-group {
    display: flex;
    align-items: center;
    flex: 1;
    background: var(--vp-c-bg-alt);
    border: 1px solid var(--vp-c-divider);
    border-radius: 4px;
    padding: 4px 8px;
    transition: all 0.2s;
}

.input-group:focus-within {
    border-color: var(--vp-c-brand-1);
    box-shadow: 0 0 0 2px var(--vp-c-brand-soft);
}

.search-icon, .replace-icon {
    color: var(--vp-c-text-3);
    margin-right: 8px;
}

input {
    border: none;
    background: transparent;
    outline: none;
    flex: 1;
    color: var(--vp-c-text-1);
    font-size: 14px;
    min-width: 0;
}

.actions {
    display: flex;
    align-items: center;
    gap: 2px;
    margin-left: 8px;
    padding-left: 8px;
    border-left: 1px solid var(--vp-c-divider);
}

.actions button {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border: none;
    background: transparent;
    color: var(--vp-c-text-2);
    border-radius: 4px;
    cursor: pointer;
}

.actions button:hover {
    background: var(--vp-c-bg-soft);
    color: var(--vp-c-text-1);
}

.expand-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 32px;
    margin-left: 4px;
    cursor: pointer;
    color: var(--vp-c-text-3);
    border-radius: 4px;
}
.expand-btn:hover {
    background: var(--vp-c-bg-soft);
    color: var(--vp-c-text-2);
}

.text-actions button {
    width: auto;
    padding: 0 8px;
    font-size: 12px;
    height: 24px;
    white-space: nowrap;
}
.text-actions button:hover {
    background: var(--vp-c-brand-soft);
    color: var(--vp-c-brand-1);
}
</style>
