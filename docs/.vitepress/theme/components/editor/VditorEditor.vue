<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import Vditor from 'vditor'
import 'vditor/dist/index.css'
import { useAppStore } from '../../stores/app'

const props = defineProps<{
  modelValue?: string
  placeholder?: string
  height?: number | string
  status?: string
}>()

const emit = defineEmits(['update:modelValue', 'after', 'focus', 'blur'])

const editorRef = ref<HTMLElement | null>(null)
const vditorInstance = ref<Vditor | null>(null)
const appStore = useAppStore()

const initVditor = () => {
  if (!editorRef.value) return

  vditorInstance.value = new Vditor(editorRef.value, {
    height: props.height || 600,
    width: 'auto',
    placeholder: props.placeholder || 'Start writing...',
    theme: appStore.isDark ? 'dark' : 'classic',
    mode: 'ir', // Instant Rendering (所见即所得)
    preview: {
      theme: {
        current: appStore.isDark ? 'dark' : 'light',
      },
      hljs: {
        style: appStore.isDark ? 'dracula' : 'github',
      },
      math: {
        engine: 'KaTeX',
        inlineDigit: true,
      },
      markdown: {
        toc: true,
        mark: true,
        footnotes: true,
        autoSpace: true,
      },
    },
    toolbar: [
      'emoji',
      'headings',
      'bold',
      'italic',
      'strike',
      'link',
      '|',
      'list',
      'ordered-list',
      'check',
      'outdent',
      'indent',
      '|',
      'quote',
      'line',
      'code',
      'inline-code',
      'insert-before',
      'insert-after',
      '|',
      'upload',
      // 'record',
      'table',
      '|',
      'undo',
      'redo',
      '|',
      'fullscreen',
      'edit-mode',
      {
        name: 'more',
        toolbar: [
          'both',
          'code-theme',
          'content-theme',
          'export',
          'outline',
          'preview',
        ],
      },
    ],
    upload: {
      accept: 'image/*,.mp3, .wav, .rar',
      token: 'test',
      url: '/api/upload/editor',
      linkToImgUrl: '/api/upload/fetch',
      filename(name: string) {
        return name.replace(/[^(a-zA-Z0-9\u4e00-\u9fa5\.)]/g, '').
          replace(/[\?\\/:|<>\*\[\]\(\)\$%\{\}@~]/g, '').
          replace('/\\s/g', '')
      },
    },
    toolbarConfig: {
      pin: true,
    },
    counter: {
      enable: true,
      type: 'markdown',
    },
    resize: {
      enable: false, // Auto height driven by container
    },
    outline: {
      enable: true,
      position: 'right',
    },
    cache: {
      enable: false,
    },
    value: props.modelValue || '',
    input: (value: string) => {
      emit('update:modelValue', value)
    },
    focus: (value: string) => {
      emit('focus', value)
    },
    blur: (value: string) => {
      emit('blur', value)
    },
    after: () => {
      emit('after')
    },
  })
}

// Watch theme changes to update Vditor theme
watch(
  () => appStore.isDark,
  (isDark) => {
    if (vditorInstance.value) {
      vditorInstance.value.setTheme(
        isDark ? 'dark' : 'classic',
        isDark ? 'dark' : 'light',
        isDark ? 'dracula' : 'github'
      )
    }
  }
)

// Status Indicator logic
const injectStatus = () => {
    if (!editorRef.value) return
    const toolbar = editorRef.value.querySelector('.vditor-toolbar')
    if (toolbar) {
        let statusEl = document.getElementById('vditor-custom-status')
        if (!statusEl) {
           statusEl = document.createElement('div')
           statusEl.id = 'vditor-custom-status'
           statusEl.className = 'vditor-custom-status'
           toolbar.appendChild(statusEl)
        }
        statusEl.innerHTML = getStatusHTML()
    }
}

const getStatusHTML = () => {
    const s = props.status || 'saved'
    const text = s === 'saving' ? 'Saving...' : s === 'saved' ? 'All saved' : 'Error'
    const color = s === 'saving' ? 'var(--vp-c-brand)' : s === 'error' ? 'var(--vp-c-danger)' : 'var(--vp-c-text-3)'
    return `<span style="color:${color}; font-size: 12px; display: flex; align-items: center; gap: 4px;">
        <span style="width:6px;height:6px;border-radius:50%;background:${color}"></span>${text}
    </span>`
}

// Fix Outline Click Navigation (Event Delegation + Text Sync)
// Fix Outline Click Navigation (Event Delegation + Text Sync)
const enableOutlineClick = () => {
    // Vditor re-renders outline on input, so we must observe it
    const observer = new MutationObserver(() => {
        const outlineContainer = document.querySelector('.vditor-outline')
        if (outlineContainer && !outlineContainer.getAttribute('data-click-bound')) {
            
            // Add delegated listener to the container KEY: Use Capture Phase to intercept before Vditor's native handler
            outlineContainer.addEventListener('click', (e) => {
                const target = e.target as HTMLElement
                // Vditor puts data-target-id on the span or element inside the item
                const targetItem = target.closest('[data-target-id]') as HTMLElement
                
                if (!targetItem) return

                // We must stop propagation to prevent Vditor's native handler (which might fail on auto-height) 
                // from catching this and doing nothing (or failing)
                e.preventDefault()
                e.stopPropagation()

                const id = targetItem.getAttribute('data-target-id') // Correct attribute name from Vditor source
                const text = targetItem.innerText.trim() 
                const editorContainer = editorRef.value as HTMLElement
                if (!editorContainer) return

                let targetHeader: HTMLElement | null = null

                // Strategy 1: Exact ID Match (Vditor generated ID)
                if (id) {
                    // Try global search first as Vditor IDs should be unique in DOM
                    targetHeader = document.getElementById(id)
                    
                    // Specific check inside editor if not found globally (shadow/scoped)
                    if (!targetHeader) {
                        targetHeader = editorContainer.querySelector(`[id="${id}"]`) as HTMLElement
                    }
                    
                     // Strategy 2: Vditor Data-Block-ID (IR Mode) - Fallback
                    if (!targetHeader) {
                        targetHeader = editorContainer.querySelector(`[data-block-id="${id}"]`) as HTMLElement
                    }
                }

                // Strategy 3: Text Match (Fallback if ID generation caused check failure)
                if (!targetHeader && text) {
                     const candidates = Array.from(editorContainer.querySelectorAll('h1, h2, h3, h4, h5, h6'))
                     targetHeader = candidates.find(h => {
                         const hText = h.textContent?.trim() || ''
                         return hText === text || hText.endsWith(text)
                     }) as HTMLElement
                }

                if (targetHeader) {
                    // Highlight the parent item (usually separate from the span with data-target-id)
                    // We try to find the visually "item" element to highlight
                    const visualItem = target.closest('.vditor-outline__item') || targetItem
                    
                    outlineContainer.querySelectorAll('.vditor-outline__item').forEach(i => i.classList.remove('vditor-outline__item--current'))
                    visualItem.classList.add('vditor-outline__item--current')

                    // Scroll Logic - smooth center
                    targetHeader.scrollIntoView({ behavior: 'smooth', block: 'center' })
                }
            }, true) // <--- Use Capture Phase
            
            // Mark as bound
            outlineContainer.setAttribute('data-click-bound', 'true')
        }
    })

    // Start observing the editor container to catch when Vditor creates/updates the outline
    const editorEl = editorRef.value
    if (editorEl) {
        observer.observe(editorEl, { childList: true, subtree: true })
    } else {
        observer.observe(document.body, { childList: true, subtree: true })
    }
}

watch(() => props.status, () => {
    injectStatus()
})

onMounted(() => {
  nextTick(() => {
     initVditor()
     setTimeout(() => {
         injectStatus()
         enableOutlineClick()
     }, 800)
  })
})

onBeforeUnmount(() => {
  if (vditorInstance.value) {
    vditorInstance.value.destroy()
    vditorInstance.value = null
  }
})

// Expose instance for parent access
defineExpose({
  vditor: vditorInstance,
})
</script>

<template>
  <div ref="editorRef" class="vditor-wrapper"></div>
</template>

<style scoped>
.vditor-wrapper {
  margin-top: 1rem;
}

/* Toolbar: Fixed below Navbar, respects sidebars if needed */
:deep(.vditor-toolbar) {
    position: fixed !important;
    top: var(--vp-nav-height) !important;
    left: var(--mu-sidebar-width) !important; /* Start after left sidebar */
    right: 0 !important;
    z-index: 95 !important; /* Above outline, near resizers */
    
    height: 40px !important;
    padding: 0 16px !important;
    line-height: 40px !important;
    
    background-color: var(--vp-c-bg) !important;
    border-bottom: 1px solid var(--vp-c-divider) !important;
    border-left: 1px solid var(--vp-c-divider) !important;
    border-radius: 0 !important;
    margin: 0 !important;
    
    display: flex !important;
    align-items: center;
    justify-content: flex-start;
    flex-wrap: nowrap !important;
    gap: 4px;
    
    box-shadow: 0 2px 8px rgba(0,0,0,0.02) !important;
    transition: left 0.2s ease;
}

:deep(.vditor-toolbar__item) {
    padding: 0 6px !important;
    flex-shrink: 0 !important;
}

:deep(.vditor-custom-status) {
    margin-left: auto !important; 
    margin-right: calc(var(--mu-aside-width) + 20px) !important; /* Avoid right resizer/outline */
}

/* Hide Native VitePress Elements */
:global(.vp-doc-aside), 
:global(.VPDocAside),
:global(.VPContent.has-aside .content-container) {
    max-width: 100% !important;
}

/* Editor Content: Transparent & Responsive */
:deep(.vditor-content) {
    background-color: transparent !important;
    padding-top: 50px !important; /* Space for toolbar */
    margin: 0 auto;
    transition: max-width 0.3s ease;
}

/* Prevent color change on focus */
:deep(.vditor--focus) {
    background-color: transparent !important;
    border-color: var(--vp-c-divider) !important;
}

:deep(.vditor-reset) {
    background-color: transparent !important;
    color: var(--vp-c-text-1) !important;
}

/* Width Modes from layoutStore */
:global([data-content-width="narrow"]) :deep(.vditor-content) {
    max-width: 720px !important;
}
:global([data-content-width="half"]) :deep(.vditor-content) {
    max-width: 900px !important;
}
:global([data-content-width="full"]) :deep(.vditor-content) {
    max-width: 100% !important;
}

/* Outline: Integrated Fixed Sidebar - Modern Redesign */
:deep(.vditor-outline) {
    position: fixed !important;
    top: calc(var(--vp-nav-height) + 40px) !important; 
    right: 0 !important;
    bottom: 0 !important;
    width: var(--mu-aside-width) !important; /* SYNC WITH GLOBAL LAYOUT */
    
    background-color: var(--vp-c-bg-alt) !important; /* Distinct background */
    border-left: 1px solid var(--vp-c-divider) !important;
    display: block !important;
    overflow-y: auto !important;
    z-index: 80 !important;
    padding: 20px 10px 20px 20px !important;
    transition: width 0.2s ease;
    
    /* Font settings */
    font-size: 13px !important;
    line-height: 1.5 !important;
}

/* Hide Vditor internal title if present */
:deep(.vditor-outline__title) {
    display: none !important;
}

/* Outline Item Styling */
:deep(.vditor-outline__item) {
    display: block !important;
    padding: 4px 12px 4px 12px !important;
    margin: 4px 0 !important;
    border-left: 2px solid transparent !important;
    color: var(--vp-c-text-2) !important;
    border-radius: 0 4px 4px 0 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
    overflow: hidden !important;
}

:deep(.vditor-outline__item:hover) {
    color: var(--vp-c-brand) !important;
    background-color: var(--vp-c-bg-soft) !important;
}

:deep(.vditor-outline__item--current) {
    color: var(--vp-c-brand) !important;
    border-left-color: var(--vp-c-brand) !important;
    background-color: var(--vp-c-brand-dimm) !important;
    font-weight: 500 !important;
}

/* Indentation levels (Vditor uses nesting or classes, we assume flat list with padding if IR) */
/* Vditor generates flattened list with style="padding-left: xx" usually. 
   We try to override or normalize if possible, but style attr is hard to override without !important on specific attr selector which is messy.
   Instead, we just enhance the item look. */

:deep(.vditor-outline__button) {
    display: none !important; /* Hide expand/collapse buttons if any */
}
</style>
