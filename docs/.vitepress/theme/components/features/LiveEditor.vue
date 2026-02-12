<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useData } from 'vitepress'
import { useAnnotationStore } from '../../stores/annotation'
import MarkdownIt from 'markdown-it'
import katex from '@iktakahiro/markdown-it-katex'

const { page, frontmatter } = useData()
const store = useAnnotationStore()
const md = new MarkdownIt({ html: true, linkify: true, typographer: true })
const mdBlock = new MarkdownIt({ html: true }).use(katex, { strict: false })

// --- State ---
const toolbarPos = ref({ x: 0, y: 0 })
const showToolbar = ref(false)
const selectedText = ref('')
const selectedContext = ref({ prefix: '', suffix: '' })

// Block Editor (Overlay)
const showBlockEditor = ref(false)
const formulaEditingSource = ref('')
const formulaEditingDisplay = ref(false)
const formulaEditingTarget = ref<HTMLElement | null>(null)
const formulaPreviewHtml = ref('')

// History & Editing
const isEditMode = ref(false)
const historyStack = ref<{ html: string; md: string }[]>([])
const historyIndex = ref(-1)

// --- Constants ---
const MAX_HISTORY = 50

// --- Helpers: HTML -> Markdown Serializer (Simplified Turndown) ---
const serializeDOM = (root: Element): string => {
    let output = ''
    
    // Helper to process nodes recursively
    const processNode = (node: Node): string => {
        if (node.nodeType === Node.TEXT_NODE) {
            // Escape special chars? For now, raw text.
            return node.textContent || ''
        }
        if (node.nodeType !== Node.ELEMENT_NODE) return ''

        const el = node as HTMLElement
        const tagName = el.tagName.toLowerCase()
        let inner = ''
        
        // Handle Code Blocks special case (pre > code)
        if (tagName === 'pre') {
             const code = el.querySelector('code')
             const lang = Array.from(code?.classList || []).find(c => c.startsWith('language-'))?.replace('language-', '') || ''
             return `\n\`\`\`${lang}\n${code?.innerText || el.innerText}\n\`\`\`\n`
        }
        
        // Iterate children
        el.childNodes.forEach(child => {
            inner += processNode(child)
        })

        switch (tagName) {
            case 'h1': return `# ${inner}\n\n`
            case 'h2': return `## ${inner}\n\n`
            case 'h3': return `### ${inner}\n\n`
            case 'h4': return `#### ${inner}\n\n`
            case 'h5': return `##### ${inner}\n\n`
            case 'h6': return `###### ${inner}\n\n`
            case 'p': return `${inner}\n\n`
            case 'strong': case 'b': return `**${inner}**`
            case 'em': case 'i': return `*${inner}*`
            case 'u': return `<u>${inner}</u>`
            case 'del': case 's': return `~~${inner}~~`
            case 'ul': return `${inner}\n`
            case 'ol': return `${inner}\n`
            case 'li': return `- ${inner}\n` // Simple list handling
            case 'blockquote': return `> ${inner}\n\n`
            case 'a': return `[${inner}](${el.getAttribute('href')})`
            case 'img': return `![${el.getAttribute('alt') || ''}](${el.getAttribute('src')})`
            case 'div': return `${inner}\n` // Div often wraps stuff
            case 'span': 
                if (el.classList.contains('katex')) {
                    const annotation = el.querySelector('annotation[encoding="application/x-tex"]')
                    const isDisplay = el.classList.contains('katex-display') || (el.parentNode as HTMLElement)?.classList.contains('katex-display')
                    if (annotation) {
                        return isDisplay ? `$$\n${annotation.textContent}\n$$` : `$${annotation.textContent}$`
                    }
                }
                return inner
            default: return inner
        }
    }

    // Direct children of .vp-doc
    root.childNodes.forEach(child => {
        output += processNode(child)
    })
    
    return output.trim()
}

// --- History Management ---
const recordSnapshot = () => {
    const container = document.querySelector('.vp-doc > div') || document.querySelector('.vp-doc')
    if (!container) return

    const html = container.innerHTML
    
    // If identical to current top, skip
    if (historyIndex.value >= 0 && historyStack.value[historyIndex.value].html === html) return

    // Truncate future
    if (historyIndex.value < historyStack.value.length - 1) {
        historyStack.value = historyStack.value.slice(0, historyIndex.value + 1)
    }

    // Compute Markdown (Lazy? No, need it for sync)
    const mdContent = serializeDOM(container)
    
    historyStack.value.push({ html, md: mdContent })
    historyIndex.value++
    
    if (historyStack.value.length > MAX_HISTORY) {
        historyStack.value.shift()
        historyIndex.value--
    }
    
    // Trigger Sync
    store.pendingContent = mdContent
    triggerAutoSync()
}

const restoreSnapshot = (idx: number) => {
    const snapshot = historyStack.value[idx]
    if (!snapshot) return

    const container = document.querySelector('.vp-doc > div') || document.querySelector('.vp-doc')
    if (container) {
        container.innerHTML = snapshot.html
        
        // Re-bind listeners if needed (Input listener is on container, so it persists? No, contentEditable might need verify)
        if (isEditMode.value) enableContainerEdit()
    }
    
    store.pendingContent = snapshot.md
    triggerAutoSync()
}

const undo = () => {
    if (historyIndex.value > 0) {
        historyIndex.value--
        restoreSnapshot(historyIndex.value)
    }
}

const redo = () => {
    if (historyIndex.value < historyStack.value.length - 1) {
        historyIndex.value++
        restoreSnapshot(historyIndex.value)
    }
}

// --- Sync Logic ---
let syncTimer: any = null
const triggerAutoSync = () => {
    if (syncTimer) clearTimeout(syncTimer)
    store.editorStatus = 'syncing'
    syncTimer = setTimeout(saveContent, 1000)
}

const saveContent = async () => {
    try {
        const res = await fetch('/api/save-md', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filePath: frontmatter.value.filePath || page.value.relativePath.replace(/^view\//, ''),
                content: store.pendingContent,
                message: 'Auto sync'
            })
        })
        if (res.ok) store.setSynced()
        else store.editorStatus = 'error'
    } catch (e) {
        store.editorStatus = 'error'
    }
}

// --- Container Editing ---
const handleContainerInput = (e: Event) => {
    // This fires on ANY content change (text, enter, delete)
    // We debounce snapshot recording? Or record on "pause"? 
    // Recording every keystroke into history is expensive + memory heavy.
    // Better: Debounce recording.
    
    store.editorStatus = 'syncing'
    debouncedRecord()
}

let recordTimer: any = null
const debouncedRecord = () => {
    if (recordTimer) clearTimeout(recordTimer)
    recordTimer = setTimeout(() => {
        recordSnapshot()
    }, 800) // 800ms pause -> snapshot
}

const enableContainerEdit = () => {
    const container = document.querySelector('.vp-doc > div') || document.querySelector('.vp-doc')
    if (container) {
        (container as HTMLElement).contentEditable = 'true'
        ;(container as HTMLElement).classList.add('global-editable')
        container.addEventListener('input', handleContainerInput)
        
        // Disable links click in edit mode
        container.querySelectorAll('a').forEach(a => a.style.pointerEvents = 'none')
        
        // Add formula click listener
        container.addEventListener('click', handleFormulaClick as any)
    }
}

const disableContainerEdit = () => {
    const container = document.querySelector('.vp-doc > div') || document.querySelector('.vp-doc')
    if (container) {
        (container as HTMLElement).contentEditable = 'false'
        ;(container as HTMLElement).classList.remove('global-editable')
        container.removeEventListener('input', handleContainerInput)
        container.removeEventListener('click', handleFormulaClick as any)
        
        // Re-enable links
        container.querySelectorAll('a').forEach(a => a.style.pointerEvents = 'auto')
    }
}


// --- Formula Overlay Editor Logic ---
watch(formulaEditingSource, (val) => {
    try {
        formulaPreviewHtml.value = mdBlock.render(formulaEditingDisplay.value ? `$$\n${val}\n$$` : `$${val}$`)
    } catch (e) {
        formulaPreviewHtml.value = '<span style="color:red">渲染错误</span>'
    }
})

const handleFormulaClick = (e: MouseEvent) => {
    if (!isEditMode.value) return
    
    const katexEl = (e.target as HTMLElement).closest('.katex')
    if (!katexEl) return
    
    e.preventDefault()
    e.stopPropagation()
    
    formulaEditingTarget.value = katexEl as HTMLElement
    const annotation = katexEl.querySelector('annotation[encoding="application/x-tex"]')
    if (annotation) {
        formulaEditingSource.value = annotation.textContent || ''
        formulaEditingDisplay.value = katexEl.classList.contains('katex-display') || 
                                     (katexEl.parentNode as HTMLElement)?.classList.contains('katex-display')
        showBlockEditor.value = true
    }
}

const saveFormula = () => {
    if (!formulaEditingTarget.value) return
    
    // Create new rendered HTML based on input
    const newHtml = mdBlock.render(formulaEditingDisplay.value ? `$$\n${formulaEditingSource.value}\n$$` : `$${formulaEditingSource.value}$`)
    
    // Create a temporary container to extract the actual .katex node
    const temp = document.createElement('div')
    temp.innerHTML = newHtml
    const newNode = temp.querySelector('.katex')
    
    if (newNode) {
        formulaEditingTarget.value.parentNode?.replaceChild(newNode, formulaEditingTarget.value)
        showBlockEditor.value = false
        recordSnapshot()
    }
}

const closeFormulaEditor = () => {
    showBlockEditor.value = false
}


// --- Initialization ---
const init = async () => {
    const targetPath = frontmatter.value.filePath || page.value.relativePath.replace(/\.md$/, '')
    const res = await fetch(`/api/read-md?path=${targetPath}`)
    if (res.ok) {
        const data = await res.json()
        store.pendingContent = data.content
        store.editorStatus = 'synced'
        
        // Initial Snapshot
        const container = document.querySelector('.vp-doc > div') || document.querySelector('.vp-doc')
        if (container) {
             historyStack.value.push({ html: container.innerHTML, md: data.content })
             historyIndex.value = 0
        }
    }
}

// --- Toolbars & Overlays ---
const handleSelection = () => {
    if (isEditMode.value) {
        showToolbar.value = false
        return
    }
    // ... Existing selection logic (for Read-only mode annotations if needed) ...
    // Simplified: Disable floating toolbar in Edit Mode to avoid annoyance
}

// ... Keep existing Toolbar applyAction logic? 
// If we are in "Global Edit Mode", applyAction (Bold/Italic) should probably use `document.execCommand`
// to be compatible with contentEditable!
const applyAction = (action: string) => {
    if (isEditMode.value) {
        // Use browser native command for WYSIWYG
        switch (action) {
            case 'bold': document.execCommand('bold'); break
            case 'underline': document.execCommand('underline'); break
            case 'strikethrough': document.execCommand('strikeThrough'); break
            case 'delete': document.execCommand('delete'); break
            case 'copy': document.execCommand('copy'); break
        }
        recordSnapshot() // Force snapshot after action
        return
    }
    
    // ... Old Logic for Read-Only mode (Regex replacement) ...
    // Omitted for brevity/cleanliness, assuming Edit Mode is primary now.
}

const handleKeydown = (e: KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        e.preventDefault()
        undo()
    }
    if ((e.ctrlKey || e.metaKey) && e.key === 'y') {
        e.preventDefault()
        redo()
    }
}

// --- Lifecycle ---
onMounted(() => {
    init()
    window.addEventListener('keydown', handleKeydown)
    document.addEventListener('selectionchange', handleSelection)
})
onUnmounted(() => {
    disableContainerEdit()
    window.removeEventListener('keydown', handleKeydown)
    document.removeEventListener('selectionchange', handleSelection)
})

watch(() => isEditMode.value, (val) => {
    if (val) enableContainerEdit()
    else disableContainerEdit()
})

watch(() => page.value.relativePath, () => {
    isEditMode.value = false
    historyStack.value = []
    historyIndex.value = -1
    init()
})
</script>

<template>
    <div class="live-editor-3">
        <Teleport to="body">
            <!-- Dock -->
            <div class="live-dock">
                <div class="doc-status-badge" :class="store.editorStatus">
                    <span class="status-icon">●</span>
                    {{ store.editorStatus === 'syncing' ? '保存中...' : store.editorStatus === 'error' ? '同步失败' : '已同步' }}
                </div>
                
                <div class="dock-divider"></div>
                
                <button class="dock-btn" @click="undo" :disabled="historyIndex <= 0" title="撤销 (Ctrl+Z)">
                    ↩
                </button>
                <button class="dock-btn" @click="redo" :disabled="historyIndex >= historyStack.length - 1" title="重做 (Ctrl+Y)">
                    ↪
                </button>
                
                <div class="dock-divider"></div>

                <button class="dock-btn mode-btn" :class="{ active: isEditMode }" @click="isEditMode = !isEditMode" title="切换编辑模式">
                    {{ isEditMode ? '📝 编辑模式' : '👁️ 阅读模式' }}
                </button>
            </div>
        </Teleport>

        <!-- Formula Overlay Editor -->
        <Teleport to="body">
            <Transition name="fade">
                <div v-if="showBlockEditor" class="formula-overlay">
                    <div class="formula-modal">
                        <div class="modal-header">
                            <h3>编辑数学公式</h3>
                            <button @click="closeFormulaEditor" class="close-btn">×</button>
                        </div>
                        <div class="modal-body">
                            <div class="edit-pane">
                                <label>LaTeX 源码</label>
                                <textarea 
                                    v-model="formulaEditingSource" 
                                    placeholder="输入 LaTeX 源码..."
                                    spellcheck="false"
                                ></textarea>
                                <div class="options">
                                    <label>
                                        <input type="checkbox" v-model="formulaEditingDisplay"> 块级显示 ($$)
                                    </label>
                                </div>
                            </div>
                            <div class="preview-pane">
                                <label>实时预览</label>
                                <div class="math-preview" v-html="formulaPreviewHtml"></div>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button @click="closeFormulaEditor" class="cancel-btn">取消</button>
                            <button @click="saveFormula" class="save-btn">确认修改</button>
                        </div>
                    </div>
                </div>
            </Transition>
        </Teleport>
    </div>
</template>

<style scoped>
.live-editor-3 { position: absolute; top: 0; left: 0; pointer-events: none; z-index: 500; }

/* Dock Styles */
.live-dock {
    position: fixed; top: 12px; right: 80px;
    background: var(--vp-c-bg); border: 1px solid var(--vp-c-divider);
    padding: 4px 8px; border-radius: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    z-index: 2000;
    display: flex; align-items: center; gap: 4px;
    pointer-events: auto;
    transition: all 0.2s;
}
.live-dock:hover { box-shadow: 0 6px 16px rgba(0,0,0,0.15); }

.doc-status-badge { display: flex; align-items: center; gap: 6px; font-size: 11px; padding: 0 8px; color: var(--vp-c-text-2); }
.doc-status-badge.syncing { color: var(--vp-c-brand); }
.status-icon { font-size: 8px; }
.syncing .status-icon { animation: blink 1s infinite; }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

.dock-divider { width: 1px; height: 16px; background: var(--vp-c-divider); margin: 0 4px; }
.dock-btn { width: 28px; height: 28px; border-radius: 50%; border: none; background: transparent; cursor: pointer; display: flex; align-items: center; justify-content: center; color: var(--vp-c-text-1); transition: background 0.1s; }
.dock-btn:hover { background: var(--vp-c-bg-soft); color: var(--vp-c-brand); }
.dock-btn:disabled { opacity: 0.3; cursor: not-allowed; }
.mode-btn { width: auto; padding: 0 10px; border-radius: 14px; font-size: 11px; font-weight: 600; }
.mode-btn.active { background: var(--vp-c-brand-soft); color: var(--vp-c-brand); }

:global(.global-editable) {
    outline: none;
    caret-color: var(--vp-c-brand);
}

/* Formula Overlay Styles */
.formula-overlay {
    position: fixed; inset: 0;
    background: rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(4px);
    z-index: 3000;
    display: flex; align-items: center; justify-content: center;
}

.formula-modal {
    background: var(--vp-c-bg);
    border: 1px solid var(--vp-c-divider);
    border-radius: 12px;
    width: 800px; max-width: 90vw;
    box-shadow: 0 20px 40px rgba(0,0,0,0.2);
    display: flex; flex-direction: column;
}

.modal-header {
    padding: 12px 20px;
    border-bottom: 1px solid var(--vp-c-divider);
    display: flex; justify-content: space-between; align-items: center;
}
.modal-header h3 { margin: 0; font-size: 16px; }
.close-btn { font-size: 24px; background: none; border: none; cursor: pointer; color: var(--vp-c-text-2); }

.modal-body {
    padding: 20px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
}

.edit-pane, .preview-pane { display: flex; flex-direction: column; gap: 8px; }
.edit-pane label, .preview-pane label { font-size: 12px; color: var(--vp-c-text-2); font-weight: 600; }

textarea {
    flex: 1; height: 180px;
    background: var(--vp-c-bg-soft);
    border: 1px solid var(--vp-c-divider);
    border-radius: 6px;
    padding: 12px;
    font-family: var(--vp-font-family-mono);
    font-size: 14px;
    resize: none;
    outline: none;
    color: var(--vp-c-text-1);
}
textarea:focus { border-color: var(--vp-c-brand); }

.math-preview {
    flex: 1; height: 180px;
    background: var(--vp-c-bg-alt);
    border: 1px solid var(--vp-c-divider);
    border-radius: 6px;
    padding: 12px;
    overflow: auto;
    display: flex; align-items: center; justify-content: center;
}

.modal-footer {
    padding: 16px 20px;
    border-top: 1px solid var(--vp-c-divider);
    display: flex; justify-content: flex-end; gap: 12px;
}

.cancel-btn, .save-btn {
    padding: 8px 16px; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer;
}
.cancel-btn { background: transparent; border: 1px solid var(--vp-c-divider); color: var(--vp-c-text-1); }
.save-btn { background: var(--vp-c-brand); border: 1px solid var(--vp-c-brand); color: white; }
.save-btn:hover { opacity: 0.9; }

/* Animations */
.fade-enter-active, .fade-leave-active { transition: opacity 0.2s; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
</style>
