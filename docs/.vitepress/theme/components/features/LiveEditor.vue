<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useData } from 'vitepress'
import { useAnnotationStore } from '../../stores/annotation'
import MarkdownIt from 'markdown-it'

const { page } = useData()
const store = useAnnotationStore()
const md = new MarkdownIt({ html: true, linkify: true, typographer: true })

// 状态管理
const toolbarPos = ref({ x: 0, y: 0 })
const showToolbar = ref(false)
const selectedText = ref('')
const originalFullContent = ref('')
const isVirtualFile = ref(false)

// 获取源代码
const fetchSource = async () => {
    const res = await fetch(`/api/read-md?path=${page.value.relativePath}`)
    if (res.ok) {
        const data = await res.json()
        originalFullContent.value = data.content
        isVirtualFile.value = data.isVirtual
        if (!store.pendingContent) {
             store.stagedContent = data.content
        }
    }
}

// 即时渲染逻辑：直接更新 DOM 以避免 full reload
const updatePreview = () => {
    const container = document.querySelector('.vp-doc')
    if (container && store.pendingContent) {
        // 使用 markdown-it 渲染新内容
        // 注意：这只是局部 UI 欺骗，确保护理流畅度
        container.innerHTML = md.render(store.pendingContent)
    }
}

// 选词监听
const handleSelection = () => {
    const selection = window.getSelection()
    if (!selection || selection.isCollapsed || selection.toString().trim().length === 0) {
        showToolbar.value = false
        return
    }

    const range = selection.getRangeAt(0)
    const rect = range.getBoundingClientRect()
    
    selectedText.value = selection.toString().trim()
    toolbarPos.value = {
        x: rect.left + window.scrollX + rect.width / 2,
        y: rect.top + window.scrollY - 10
    }
    showToolbar.value = true
}

// Markdown 操作
const applyAction = async (action: string) => {
    if (!originalFullContent.value) await fetchSource()
    
    let content = store.pendingContent || originalFullContent.value
    let target = selectedText.value
    let replacement = ''

    switch (action) {
        case 'bold': replacement = `**${target}**`; break
        case 'strikethrough': replacement = `~~${target}~~`; break
        case 'underline': replacement = `<u>${target}</u>`; break
        case 'delete': replacement = ''; break
        case 'copy': 
            navigator.clipboard.writeText(target)
            showToolbar.value = false
            return
    }

    if (replacement !== undefined) {
        const newContent = content.replace(target, replacement)
        store.updatePendingContent(page.value.relativePath, newContent)
        updatePreview()
    }
    showToolbar.value = false
    window.getSelection()?.removeAllRanges()
}

// 保存
const saveChanges = async () => {
    const res = await fetch('/api/save-md', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            filePath: page.value.relativePath,
            content: store.pendingContent,
            message: 'Live Editor update'
        })
    })

    if (res.ok) {
        store.commitChanges()
        alert('修改已提交并 Git 存档！')
        location.reload()
    }
}

onMounted(() => {
    document.addEventListener('selectionchange', handleSelection)
    fetchSource()
})

onUnmounted(() => {
    document.removeEventListener('selectionchange', handleSelection)
})

watch(() => page.value.relativePath, () => {
    fetchSource()
    store.resetChanges()
})
</script>

<template>
    <div class="live-editor-container">
        <!-- 页面右上角状态显示 -->
        <Teleport to="body">
            <div class="doc-status-badge" :class="store.editorStatus">
                <span class="status-icon">●</span>
                {{ 
                    store.editorStatus === 'staged' ? '已暂存 (未同步)' : 
                    store.editorStatus === 'committed' ? '已发布 (Git 存档)' : '已同步 (源码同步)' 
                }}
            </div>
        </Teleport>

        <!-- 选词工具栏 -->
        <div v-if="showToolbar" class="editor-toolbar" :style="{ left: `${toolbarPos.x}px`, top: `${toolbarPos.y}px` }">
            <button @click="applyAction('copy')" title="复制">📋</button>
            <button @click="applyAction('bold')" title="加粗">B</button>
            <button @click="applyAction('underline')" title="下划线">U</button>
            <button @click="applyAction('strikethrough')" title="删除线">S</button>
            <button @click="applyAction('delete')" class="danger" title="删除">🗑️</button>
        </div>

        <!-- 底部状态进度条 -->
        <Transition name="slide">
            <div v-if="store.editorStatus === 'staged'" class="editor-progress-bar">
                <div class="progress-info">
                    <span class="pulse-dot"></span>
                    发现未处理的本地更改
                </div>
                <div class="progress-actions">
                    <button @click="location.reload()">放弃预览</button>
                    <button class="primary" @click="saveChanges">保存并提交 Git</button>
                </div>
            </div>
        </Transition>
    </div>
</template>

<style scoped>
.live-editor-container {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    z-index: 200;
    pointer-events: none;
}

/* 右上角状态 */
.doc-status-badge {
    position: fixed;
    top: 12px;
    right: 320px; /* 避开大纲 */
    background: var(--vp-c-bg-soft);
    border: 1px solid var(--vp-c-divider);
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    z-index: 100;
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--vp-c-text-2);
}

.doc-status-badge.staged .status-icon { color: #f59e0b; }
.doc-status-badge.committed .status-icon { color: #10b981; }
.doc-status-badge.none .status-icon { color: #94a3b8; }

.editor-toolbar {
    position: absolute;
    transform: translate(-50%, -100%);
    background: var(--vp-c-bg);
    border: 1px solid var(--vp-c-divider);
    padding: 4px;
    border-radius: 8px;
    display: flex;
    gap: 4px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    pointer-events: auto;
    z-index: 300;
}

.editor-toolbar button {
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: none;
    background: transparent;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
    transition: background 0.2s;
}

.editor-toolbar button:hover {
    background: var(--vp-c-bg-soft);
}

/* 进度条 */
.editor-progress-bar {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background: var(--vp-c-brand);
    color: white;
    padding: 6px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    z-index: 400;
    pointer-events: auto;
    font-size: 13px;
}

.progress-info { display: flex; align-items: center; gap: 8px; }
.pulse-dot { width: 6px; height: 6px; background: #fff; border-radius: 50%; animation: pulse 1.5s infinite; }
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }

.progress-actions { display: flex; gap: 10px; }
.progress-actions button {
    background: rgba(0,0,0,0.2);
    border: none;
    color: white;
    padding: 2px 12px;
    border-radius: 4px;
    cursor: pointer;
}
.progress-actions button.primary {
    background: white;
    color: var(--vp-c-brand);
    font-weight: 600;
}

.slide-enter-active, .slide-leave-active { transition: transform 0.3s; }
.slide-enter-from, .slide-leave-to { transform: translateY(100%); }
</style>
