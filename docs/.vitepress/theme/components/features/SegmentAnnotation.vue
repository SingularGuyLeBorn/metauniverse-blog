<template>
  <div class="segment-annotation-container">
    <Teleport to="body">
      <div v-if="activeParagraph" class="annotation-modal" @click.self="closeModal">
        <div class="modal-content">
          <h3>📝 Commit Annotation</h3>
          <div class="context-preview">
            "{{ activeParagraph.innerText.substring(0, 50) }}..."
          </div>
          <textarea v-model="currentNote" placeholder="Enter your observation (Supports Markdown)..."></textarea>
          <div class="actions">
            <button @click="closeModal" class="cancel-btn">Cancel</button>
            <button @click="saveNote" class="commit-btn">Git Commit</button>
          </div>
        </div>
      </div>
      
      <!-- Existing Annotations Indicator -->
      <div 
        v-for="note in annotations" 
        :key="note.id"
        class="annotation-marker"
        :style="{ top: note.top + 'px', left: note.left + 'px' }"
        @click="viewNote(note)"
        title="View Annotation"
      >
        💬
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useData } from 'vitepress'
import { useLocalStorage } from '@vueuse/core'

const { page } = useData()
const activeParagraph = ref<HTMLElement | null>(null)
const currentNote = ref('')
const storageKey = 'mu-annotations'
// Store annotations as: path -> paragraphIndex -> note
const allAnnotations = useLocalStorage<Record<string, any[]>>(storageKey, {})
const annotations = ref<any[]>([])

function getParagraphs() {
  return document.querySelectorAll('.vp-doc p')
}

function initInteractions() {
  const paragraphs = getParagraphs()
  
  paragraphs.forEach((p, index) => {
    // Add hover effect via class
    p.classList.add('annotatable-p')
    (p as HTMLElement).dataset.pIndex = String(index)
    
    // Remove old listeners to avoid dupes (simple safety)
    p.removeEventListener('click', handleParagraphClick)
    p.addEventListener('click', handleParagraphClick)
  })
  
  refreshMarkers()
}

function handleParagraphClick(e: Event) {
  // Only trigger if clicking the paragraph itself, mostly for demo
  // In a real app we might want a specific button
  if ((e as MouseEvent).altKey) { // Alt+Click to annotate
    e.preventDefault()
    activeParagraph.value = e.target as HTMLElement
    currentNote.value = ''
  }
}

function saveNote() {
  if (!activeParagraph.value || !currentNote.value) return
  
  const path = page.value.relativePath
  const pIndex = activeParagraph.value.dataset.pIndex
  
  if (!allAnnotations.value[path]) {
    allAnnotations.value[path] = []
  }
  
  allAnnotations.value[path].push({
    id: Date.now(),
    pIndex,
    text: currentNote.value,
    timestamp: Date.now()
  })
  
  closeModal()
  refreshMarkers()
}

function closeModal() {
  activeParagraph.value = null
  currentNote.value = ''
}

function viewNote(note: any) {
  alert(`Annotation:\n${note.text}`)
}

function refreshMarkers() {
  const path = page.value.relativePath
  const pageNotes = allAnnotations.value[path] || []
  const paragraphs = getParagraphs()
  
  annotations.value = pageNotes.map(note => {
    const p = paragraphs[parseInt(note.pIndex)] as HTMLElement
    if (!p) return null
    
    const rect = p.getBoundingClientRect()
    return {
      ...note,
      top: rect.top + window.scrollY,
      left: rect.right + 20 // Place marker to the right
    }
  }).filter(Boolean)
}

// Re-init on page change
watch(() => page.value.relativePath, () => {
  nextTick(() => {
    setTimeout(initInteractions, 500) // Wait for transition
  })
})

onMounted(() => {
  setTimeout(initInteractions, 1000)
  window.addEventListener('resize', refreshMarkers)
})

onUnmounted(() => {
  window.removeEventListener('resize', refreshMarkers)
})
</script>

<style>
/* Global styles for paragraphs */
.annotatable-p {
  position: relative;
  cursor: text;
}
.annotatable-p:hover::after {
  content: 'Alt+Click to Annotate';
  position: absolute;
  right: -120px;
  top: 0;
  font-size: 0.7rem;
  color: var(--vp-c-brand);
  opacity: 0.5;
}
</style>

<style scoped>
.annotation-modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0,0,0,0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background: var(--vp-c-bg);
  padding: 2rem;
  border-radius: 12px;
  width: 400px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}

.context-preview {
  font-size: 0.8rem;
  color: var(--vp-c-text-2);
  margin-bottom: 1rem;
  font-style: italic;
  border-left: 2px solid var(--vp-c-brand);
  padding-left: 0.5rem;
}

textarea {
  width: 100%;
  height: 100px;
  margin: 1rem 0;
  padding: 0.5rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
  background: var(--vp-c-bg-alt);
  color: var(--vp-c-text-1);
}

.actions {
  display: flex;
  justify-content: flex-end;
  gap: 1rem;
}

button {
  padding: 0.5rem 1rem;
  border-radius: 4px;
  cursor: pointer;
}

.commit-btn {
  background: var(--vp-c-brand);
  color: white;
}

.annotation-marker {
  position: absolute;
  cursor: pointer;
  font-size: 1.2rem;
  z-index: 100;
  transition: transform 0.2s;
}

.annotation-marker:hover {
  transform: scale(1.2);
}
</style>
