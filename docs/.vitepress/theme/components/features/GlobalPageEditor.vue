<script setup lang="ts">
import { ref, watch, onMounted, nextTick } from 'vue'
import { useData, useRoute } from 'vitepress'
import VditorEditor from '../editor/VditorEditor.vue'

const { page, frontmatter } = useData()
const route = useRoute()

const isEditable = ref(false)
const isLoading = ref(false)
const content = ref('')
const editorRef = ref<any>(null)
const saveStatus = ref<'saved' | 'saving' | 'error'>('saved')

// Routes that should be Read-Only
const READ_ONLY_ROUTES = ['/', '/about/', '/about/index.html']

const checkEditable = () => {
  const path = route.path
  // Remove .html for check
  const normalizedPath = path.replace(/\.html$/, '')
  
  // Home and About are Read Only
  if (path === '/' || normalizedPath === '/about' || normalizedPath.startsWith('/about/')) {
    isEditable.value = false
    document.body.classList.remove('is-global-editing')
    return
  }

  // All other pages are Editable
  isEditable.value = true
  document.body.classList.add('is-global-editing')
  loadContent()
}

const loadContent = async () => {
  isLoading.value = true
  try {
    // Determine path for API
    const targetPath = page.value.filePath 
    const res = await fetch(`/api/read-md?path=${targetPath}`)
    if (res.ok) {
        const data = await res.json()
        content.value = data.content
    } else {
        console.error('Failed to load content')
        content.value = '# Error loading content'
    }
  } catch (e) {
    console.error(e)
  } finally {
    isLoading.value = false
  }
}

const handleSave = async (val: string) => {
    saveStatus.value = 'saving'
    try {
        const targetPath = page.value.filePath
        const res = await fetch('/api/save-md', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filePath: targetPath,
                content: val,
                message: 'Auto save from Vditor'
            })
        })
        if (res.ok) {
            saveStatus.value = 'saved'
        } else {
            saveStatus.value = 'error'
        }
    } catch (e) {
        saveStatus.value = 'error'
    }
}

// Debounced Auto-save
let saveTimer: any = null
watch(content, (newVal) => {
    if (!isEditable.value || isLoading.value) return
    
    if (saveTimer) clearTimeout(saveTimer)
    saveStatus.value = 'saving'
    saveTimer = setTimeout(() => {
        handleSave(newVal)
    }, 2000)
})

watch(() => route.path, () => {
    checkEditable()
})

onMounted(() => {
    checkEditable()
})
</script>

<template>
  <Teleport to=".VPDoc" v-if="isEditable">
     <div class="global-editor-container" :class="{ 'is-loading': isLoading }">
        <!-- Top Info Bar (Path & Status) -->
        <div class="editor-info-bar">
            <div class="path-display">
                <span class="folder-icon">📂</span>
                {{ page.filePath }}
            </div>
            
            <div class="status-actions">
                <span class="save-status" :class="saveStatus">
                    <span class="status-dot"></span>
                    {{ saveStatus === 'saving' ? 'Saving...' : saveStatus === 'saved' ? 'All Saved' : 'Error' }}
                </span>
            </div>
        </div>

        <div v-if="isLoading" class="loading-state">
            <div class="spinner"></div>
            Loading Editor...
        </div>
        
        <VditorEditor 
            v-else
            v-model="content" 
            :status="saveStatus"
            height="100%"
            class="main-editor"
            @update:modelValue="content = $event"
        />
     </div>
  </Teleport>
</template>

<style scoped>
/* Global Layout Adjustments */
.global-editor-container {
    position: relative;
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    z-index: 10; /* Ensure above standard VP content */
    pointer-events: auto; /* Ensure clickable */
}

/* Info Bar */
.editor-info-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 20px;
    margin-bottom: 0;
    background: transparent;
    font-family: var(--vp-font-family-mono);
}

.path-display {
    font-size: 13px;
    color: var(--vp-c-text-2);
    display: flex;
    align-items: center;
    gap: 6px;
    background: var(--vp-c-bg-alt);
    padding: 4px 10px;
    border-radius: 6px;
}

.status-actions {
    display: flex;
    align-items: center;
    gap: 15px;
}

.save-status {
    font-size: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--vp-c-text-2);
    font-weight: 500;
}
.save-status.saving { color: var(--vp-c-brand); }
.save-status.error { color: var(--vp-c-danger); }

.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background-color: currentColor;
}

/* Loading */
.loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 400px;
    color: var(--vp-c-text-3);
    gap: 15px;
}
.spinner {
    width: 24px; height: 24px;
    border: 3px solid var(--vp-c-divider);
    border-top-color: var(--vp-c-brand);
    border-radius: 50%;
    animation: spin 1s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* Vditor Container tweaks */
.main-editor {
    flex: 1;
    /* Allow Vditor to take remaining height */
    min-height: 0; 
}
</style>

<style>
/* Global Overrides needed for "Full Page" feel */
body.is-global-editing .VPDoc {
    padding: 0 !important;
    max-width: 100% !important;
}

/* Strictly hide the original content container */
body.is-global-editing .VPDoc > .container {
    display: none !important;
}

/* Also hide any other direct children that are not our editor */
body.is-global-editing .VPDoc > div:not(.global-editor-container) {
    display: none !important;
}

/* Ensure our container is compatible */
body.is-global-editing .global-editor-container {
    display: flex !important;
}
</style>
