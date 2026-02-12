<script setup lang="ts">
import { ref, watch, nextTick, onMounted, onUnmounted } from 'vue'
import { useData } from 'vitepress'
import { useAnnotationStore } from '../../stores/annotation'
import * as Diff from 'diff'

const { page, frontmatter } = useData()
const store = useAnnotationStore()
// Markdown rendering is no longer needed for diff view, we show raw text
// const md = new MarkdownIt({ html: true, linkify: true, typographer: true }).use(katex, { strict: false })

const historyList = ref<any[]>([])
const showModal = ref(false)
const selectedVersion = ref<any>(null)
const versionContent = ref('') // Left side (Old)
const currentContent = ref('') // Right side (New)

// Diff state
const leftLines = ref<any[]>([])
const rightLines = ref<any[]>([])

// Scroll sync refs
const leftScroller = ref<HTMLElement | null>(null)
const rightScroller = ref<HTMLElement | null>(null)
const isSyncingLeft = ref(false)
const isSyncingRight = ref(false)

const isFolderMode = ref(false)
const folderHistoryList = ref<any[]>([])

// Resizer state
const splitRatio = ref(50)
const isResizing = ref(false)
const syncScroll = ref(true)
const hoverLineIdx = ref(-1)

const startResize = () => { isResizing.value = true }
const stopResize = () => { isResizing.value = false }
const handleResizerMove = (e: MouseEvent) => {
    if (!isResizing.value) return
    const container = (e.currentTarget as HTMLElement)
    const rect = container.getBoundingClientRect()
    const x = e.clientX - rect.left
    const percentage = (x / rect.width) * 100
    if (percentage > 10 && percentage < 90) {
        splitRatio.value = percentage
    }
}

const hoverLine = (idx: number) => {
    hoverLineIdx.value = idx
}

// 暴露打开方法给外部
defineExpose({
    open: (path?: string, isDir: boolean = false) => {
        showModal.value = true
        isFolderMode.value = isDir
        if (path) {
            // Override page relative path detection if path is provided (e.g. from Context Menu)
            // But we need to be careful about currentContent fetching
            // For now, let's just set a temp variable or just rely on the logic below
            // Actually, best to just let the component handle it if we update a ref "targetPath"
            customTargetPath.value = path
        } else {
            customTargetPath.value = ''
        }
        
        if (isDir) {
           fetchFolderHistory()
        } else {
           fetchHistory()
           fetchCurrent()
        }
    }
})

const customTargetPath = ref('')

const fetchHistory = async () => {
    // 优先使用传入的路径，否则使用当前页面路径
    let targetPath = customTargetPath.value || frontmatter.value.filePath || page.value.relativePath.replace(/\.md$/, '')
    // 针对 Shadow Files (.py.md)，还原为原始文件路径
    targetPath = targetPath.replace(/\.md$/, '')
    
    const res = await fetch(`/api/list-history?path=${targetPath}`)
    if (res.ok) {
        const data = await res.json()
        historyList.value = data.history
    }
}

const fetchCurrent = async () => {
    let targetPath = customTargetPath.value || frontmatter.value.filePath || page.value.relativePath.replace(/\.md$/, '')
    targetPath = targetPath.replace(/\.md$/, '')
    
    const res = await fetch(`/api/read-md?path=${targetPath}`)
    if (res.ok) {
        const data = await res.json()
        currentContent.value = data.content
    }
}

const viewVersion = async (version: any) => {
    selectedVersion.value = version
    const res = await fetch(`/api/read-history?file=${version.name}`)
    if (res.ok) {
        const data = await res.json()
        versionContent.value = data.content
        computeDiff()
    }
}

const computeDiff = () => {
    if (!versionContent.value || !currentContent.value) return

    // 1. Get raw line-by-line diff
    const diff = Diff.diffLines(versionContent.value, currentContent.value)
    
    const left: any[] = []
    const right: any[] = []
    
    let leftLineNum = 1
    let rightLineNum = 1
    
    // Helper to push simple lines
    const pushSimple = (lines: string[], side: 'left' | 'right', type: 'normal' | 'add' | 'remove' | 'empty') => {
        lines.forEach(line => {
            const lineObj = { 
                num: type === 'empty' ? null : (side === 'left' ? leftLineNum++ : rightLineNum++), 
                parts: [{ content: line, type: type === 'empty' ? 'transparent' : type }], // Use parts for consistency
                type 
            }
            if (side === 'left') left.push(lineObj)
            else right.push(lineObj)
        })
    }

    // Process diff chunks
    for (let i = 0; i < diff.length; i++) {
        const part = diff[i]
        const lines = part.value.replace(/\n$/, '').split('\n')

        // Check for modification (Remove followed immediately by Add)
        if (part.removed && diff[i+1] && diff[i+1].added) {
            const nextPart = diff[i+1]
            const nextLines = nextPart.value.replace(/\n$/, '').split('\n')
            
            // We align lines as much as possible, then fallback to block diff
            const maxLines = Math.max(lines.length, nextLines.length)
            
            for (let j = 0; j < maxLines; j++) {
                const leftLineStr = lines[j]
                const rightLineStr = nextLines[j]
                
                // If both exist, do word diff
                if (leftLineStr !== undefined && rightLineStr !== undefined) {
                    const wordDiff = Diff.diffWords(leftLineStr, rightLineStr)
                    
                    const leftParts: any[] = []
                    const rightParts: any[] = []
                    
                    wordDiff.forEach(p => {
                        if (p.added) {
                            rightParts.push({ content: p.value, type: 'add-word' })
                        } else if (p.removed) {
                            leftParts.push({ content: p.value, type: 'remove-word' })
                        } else {
                            leftParts.push({ content: p.value, type: 'normal' })
                            rightParts.push({ content: p.value, type: 'normal' })
                        }
                    })
                    
                    left.push({ num: leftLineNum++, parts: leftParts, type: 'modify' })
                    right.push({ num: rightLineNum++, parts: rightParts, type: 'modify' })
                } 
                // Only Update (Remove side usually)
                else if (leftLineStr !== undefined) {
                    left.push({ num: leftLineNum++, parts: [{ content: leftLineStr, type: 'remove-word' }], type: 'remove' })
                    right.push({ num: null, parts: [{ content: '', type: 'transparent' }], type: 'empty' })
                }
                // Only Update (Add side usually)
                else if (rightLineStr !== undefined) {
                    left.push({ num: null, parts: [{ content: '', type: 'transparent' }], type: 'empty' })
                    right.push({ num: rightLineNum++, parts: [{ content: rightLineStr, type: 'add-word' }], type: 'add' })
                }
            }
            
            i++ // Skip next part since we handled it
        } else if (part.added) {
            pushSimple(lines, 'right', 'add')
            // Fill left with empty
            for (let k=0; k<lines.length; k++) left.push({ num: null, parts: [{ content: '', type: 'transparent' }], type: 'empty' })
        } else if (part.removed) {
            pushSimple(lines, 'left', 'remove')
            // Fill right with empty
            for (let k=0; k<lines.length; k++) right.push({ num: null, parts: [{ content: '', type: 'transparent' }], type: 'empty' })
        } else {
             // Unchanged
             lines.forEach(line => {
                left.push({ num: leftLineNum++, parts: [{ content: line, type: 'normal' }], type: 'normal' })
                right.push({ num: rightLineNum++, parts: [{ content: line, type: 'normal' }], type: 'normal' })
             })
        }
    }
    
    leftLines.value = left
    rightLines.value = right
}

// Sync Scroll
const handleScroll = (side: 'left' | 'right') => {
    const left = leftScroller.value
    const right = rightScroller.value
    if (!left || !right) return

    if (side === 'left') {
        if (isSyncingLeft.value) {
            isSyncingLeft.value = false
            return
        }
        if (!syncScroll.value) return
        isSyncingRight.value = true
        right.scrollTop = left.scrollTop
        right.scrollLeft = left.scrollLeft
    } else {
        if (isSyncingRight.value) {
            isSyncingRight.value = false
            return
        }
        if (!syncScroll.value) return
        isSyncingLeft.value = true
        left.scrollTop = right.scrollTop
        left.scrollLeft = right.scrollLeft
    }
}

const performRollback = async () => {
    if (!selectedVersion.value || !confirm('确定要将该文档回滚到此历史版本并提交 Git 吗？')) return
    
    // 针对 Shadow Files (.py.md)，还原为原始文件路径
    const targetPath = frontmatter.value.filePath || page.value.relativePath.replace(/\.md$/, '')

    const res = await fetch('/api/rollback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            filePath: targetPath,
            historyFile: selectedVersion.value.name
        })
    })

    if (res.ok) {
        alert('回滚成功！页面即将重载。')
        location.reload()
    }
}

const fetchFolderHistory = async () => {
    const targetPath = customTargetPath.value || page.value.relativePath.replace(/\/[^/]+$/, '') // Parent dir if not specified
    const res = await fetch(`/api/git/log-folder?path=${targetPath}`)
    if (res.ok) {
        const data = await res.json()
        folderHistoryList.value = data.history
    }
}

const handleGlobalOpen = (e: CustomEvent) => {
    const { path, isDir } = e.detail
    // Open modal with specific path
    showModal.value = true
    isFolderMode.value = !!isDir
    customTargetPath.value = path
    
    if (isDir) {
        fetchFolderHistory()
    } else {
        fetchHistory()
        fetchCurrent()
    }
}

onMounted(() => {
    window.addEventListener('open-history-viewer', handleGlobalOpen as EventListener)
})

onUnmounted(() => {
    window.removeEventListener('open-history-viewer', handleGlobalOpen as EventListener)
})

const performFolderRollback = async (commitHash: string) => {
     if (!confirm(`确定要将文件夹 "${customTargetPath.value}" 回滚到 Commit ${commitHash} 吗？\n注意：这将覆盖文件夹下的所有文件！`)) return
     
     const res = await fetch('/api/git/checkout', {
         method: 'POST',
         headers: { 'Content-Type': 'application/json' },
         body: JSON.stringify({
             commitHash,
             path: customTargetPath.value || page.value.relativePath.replace(/\/[^/]+$/, '')
         })
     })
     
     if (res.ok) {
         alert('文件夹回滚成功！全站即将刷新。')
         location.reload()
     } else {
         const err = await res.json()
         alert('回滚失败: ' + err.error)
     }
}

watch(() => page.value.relativePath, async () => {
    if (!showModal.value) return // If closed, just ignore
    
    // Check if we are in "Context Menu Mode" (customTargetPath is set)
    // If user navigates, we probably should close the modal or switch to new page
    // Let's reset to current page mode
    isFolderMode.value = false
    customTargetPath.value = ''
    
    // 立即清空状态
    historyList.value = []
    folderHistoryList.value = []
    selectedVersion.value = null
    versionContent.value = ''
    currentContent.value = ''
    leftLines.value = []
    rightLines.value = []
    
    await fetchHistory()
    await fetchCurrent()
})
</script>

<template>
    <div class="history-container">
        <Transition name="fade">
            <div v-if="showModal" class="history-overlay">
                <div class="history-panel">
                    <div class="panel-header">
                        <div class="header-left">
                            <span class="icon">{{ isFolderMode ? '📂' : '📜' }}</span>
                            <h3>{{ isFolderMode ? '文件夹Git历史' : '本地文件历史' }}: {{ customTargetPath || page.relativePath }}</h3>
                        </div>
                        <div class="header-right">
                            <button class="close-x" @click="showModal = false">×</button>
                        </div>
                    </div>
                    
                    <div class="panel-layout" v-if="!isFolderMode">
                        <!-- 左侧：版本列表 (IDE 风格) -->
                        <div class="version-list">
                            <div class="list-title">版本快照 (GIT)</div>
                            <div 
                                v-for="h in historyList" 
                                :key="h.name" 
                                class="version-item"
                                :class="{ active: selectedVersion?.name === h.name }"
                                @click="viewVersion(h)"
                            >
                                <div class="item-time">{{ new Date(h.time).toLocaleString() }}</div>
                                <div class="item-id">#{{ h.name.slice(0, 7) }}</div>
                            </div>
                            <div v-if="historyList.length === 0" class="empty-state">暂无版本记录</div>
                        </div>

                        <!-- 右侧：Code Diff Viewer -->
                        <div class="diff-viewer">
                            <div v-if="selectedVersion" class="diff-toolbar">
                                <span class="diff-id">正在对比: {{ selectedVersion.name }} (Old) vs Current (New)</span>
                                <div class="toolbar-actions">
                                    <label><input type="checkbox" v-model="syncScroll" /> 同步滚动</label>
                                    <button class="rollback-btn" @click="performRollback">回滚至此版本</button>
                                </div>
                            </div>
                            
                            <div v-if="leftLines.length > 0" class="diff-container" @mousemove="handleResizerMove" @mouseup="stopResize" @mouseleave="stopResize">
                                <!-- LEFT COLUMN: OLD VERSION -->
                                <div class="diff-pane left-pane" ref="leftScroller" @scroll="handleScroll('left')" :style="{ width: splitRatio + '%' }">
                                    <div class="code-line" 
                                         v-for="(line, idx) in leftLines" 
                                         :key="idx" 
                                         :class="[line.type, { 'hover-active': hoverLineIdx === idx }]"
                                         @mouseenter="hoverLine(idx)"
                                    >
                                        <div class="line-num">{{ line.num || '' }}</div>
                                        <div class="line-content" :class="{ 'empty-placeholder': line.type === 'empty' }">
                                            <pre><span 
                                                v-for="(part, pIdx) in line.parts" 
                                                :key="pIdx"
                                                :class="part.type"
                                            >{{ part.content }}</span></pre>
                                        </div>
                                    </div>
                                </div>

                                <!-- RESIZER HANDLE -->
                                <div class="diff-resizer" @mousedown="startResize"></div>
                                
                                <!-- RIGHT COLUMN: NEW VERSION -->
                                <div class="diff-pane right-pane" ref="rightScroller" @scroll="handleScroll('right')" :style="{ width: (100 - splitRatio) + '%' }">
                                    <div class="code-line" 
                                         v-for="(line, idx) in rightLines" 
                                         :key="idx" 
                                         :class="[line.type, { 'hover-active': hoverLineIdx === idx }]"
                                         @mouseenter="hoverLine(idx)"
                                    >
                                        <div class="line-num">{{ line.num || '' }}</div>
                                        <div class="line-content" :class="{ 'empty-placeholder': line.type === 'empty' }">
                                            <pre><span 
                                                v-for="(part, pIdx) in line.parts" 
                                                :key="pIdx"
                                                :class="part.type"
                                            >{{ part.content }}</span></pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div v-else class="diff-placeholder">
                                <div class="hint">请在左侧选择版本以启动 IDE 级双栏溯源对比</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- 文件夹历史视图 -->
                    <div class="folder-layout" v-else>
                         <table class="git-log-table">
                             <thead>
                                 <tr>
                                     <th>Hash</th>
                                     <th>Date</th>
                                     <th>Author</th>
                                     <th>Message</th>
                                     <th style="width: 100px;">Actions</th>
                                 </tr>
                             </thead>
                             <tbody>
                                 <tr v-for="commit in folderHistoryList" :key="commit.hash">
                                     <td class="mono">{{ commit.hash }}</td>
                                     <td>{{ commit.date }}</td>
                                     <td>{{ commit.author }}</td>
                                     <td>{{ commit.message }}</td>
                                     <td>
                                         <button class="small-btn" @click="performFolderRollback(commit.hash)">Restore</button>
                                     </td>
                                 </tr>
                                 <tr v-if="folderHistoryList.length === 0">
                                     <td colspan="5" style="text-align:center; padding: 20px;">暂无 Git 提交记录</td>
                                 </tr>
                             </tbody>
                         </table>
                    </div>
                </div>
            </div>
        </Transition>
    </div>
</template>

<style scoped>
.history-overlay { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0,0,0,0.8); z-index: 9999; display: flex; align-items: center; justify-content: center; backdrop-filter: blur(4px); pointer-events: auto; }
.history-panel { background: var(--vp-c-bg); width: 95vw; height: 90vh; border-radius: 12px; display: flex; flex-direction: column; overflow: hidden; box-shadow: 0 40px 100px rgba(0,0,0,0.6); }

.panel-header { padding: 12px 24px; border-bottom: 1px solid var(--vp-c-divider); display: flex; justify-content: space-between; align-items: center; background: var(--vp-c-bg-soft); }
.header-left { display: flex; align-items: center; gap: 10px; }
.header-left h3 { margin: 0; font-size: 15px; font-weight: 600; color: var(--vp-c-text-1); }
.close-x { border: none; background: none; font-size: 24px; cursor: pointer; color: var(--vp-c-text-3); }
.close-x:hover { color: var(--vp-c-brand); }

.panel-layout { flex: 1; display: flex; overflow: hidden; }
.version-list { width: 220px; border-right: 1px solid var(--vp-c-divider); background: var(--vp-c-bg-alt); display: flex; flex-direction: column; }
.list-title { padding: 10px 16px; font-size: 11px; font-weight: 700; color: var(--vp-c-text-3); background: rgba(0,0,0,0.05); }

.version-item { padding: 10px 16px; cursor: pointer; border-bottom: 1px solid var(--vp-c-divider); transition: background 0.2s; }
.version-item:hover { background: var(--vp-c-bg-soft); }
.version-item.active { background: var(--vp-c-brand-soft); border-left: 3px solid var(--vp-c-brand); }
.item-time { font-size: 12px; color: var(--vp-c-text-1); font-weight: 500; }
.item-id { font-size: 10px; color: var(--vp-c-text-3); font-family: var(--vp-font-family-mono); margin-top: 2px; }

.diff-viewer { flex: 1; display: flex; flex-direction: column; background: var(--vp-c-bg); position: relative; overflow: hidden; }
.diff-toolbar { padding: 8px 16px; border-bottom: 1px solid var(--vp-c-divider); display: flex; justify-content: space-between; align-items: center; background: var(--vp-c-bg-soft); }
.diff-id { font-size: 11px; font-family: var(--vp-font-family-mono); color: var(--vp-c-brand); }
.rollback-btn { background: var(--vp-c-brand); color: white; border: none; padding: 4px 12px; border-radius: 4px; font-size: 11px; font-weight: 600; cursor: pointer; }
.toolbar-actions { display: flex; gap: 10px; align-items: center; font-size: 12px; }

.diff-container { flex: 1; display: flex; overflow: hidden; position: relative; user-select: text; }
.diff-pane { overflow: auto; display: flex; flex-direction: column; background: var(--vp-c-bg); }
.diff-resizer { width: 4px; background: var(--vp-c-divider); cursor: col-resize; transition: background 0.2s; z-index: 10; flex-shrink: 0; }
.diff-resizer:hover, .diff-resizer:active { background: var(--vp-c-brand); }

.code-line { display: flex; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 13px; line-height: 20px; width: max-content; min-width: 100%; }
.code-line:hover, .code-line.hover-active { background: rgba(0,0,0,0.05); }
.dark .code-line:hover, .dark .code-line.hover-active { background: rgba(255,255,255,0.05); }

/* Highlighting - High Contrast */
.code-line.add { background: rgba(16, 185, 129, 0.2); } /* Green */
.code-line.remove { background: rgba(239, 68, 68, 0.2); } /* Red */
.code-line.empty { background: rgba(0,0,0,0.02); }
.code-line.modify { background: rgba(234, 179, 8, 0.1); } /* Yellow tint for modified lines */

/* Word Level Highlighting */
.add-word { background: rgba(16, 185, 129, 0.4); border-radius: 2px; }
.remove-word { background: rgba(239, 68, 68, 0.4); text-decoration: line-through; border-radius: 2px; }

/* Dark mode override for better contrast */
.dark .code-line.add { background: rgba(16, 185, 129, 0.15); }
.dark .code-line.remove { background: rgba(239, 68, 68, 0.15); }
.dark .code-line.modify { background: rgba(234, 179, 8, 0.15); }
.dark .add-word { background: rgba(16, 185, 129, 0.5); color: #eefff5; }
.dark .remove-word { background: rgba(239, 68, 68, 0.5); color: #ffeef0; }

/* Line Numbers */
.line-num { width: 45px; text-align: right; padding-right: 10px; color: var(--vp-c-text-3); user-select: none; border-right: 1px solid var(--vp-c-divider); background: var(--vp-c-bg-alt); flex-shrink: 0; }

/* Content */
.line-content { padding-left: 10px; white-space: pre; flex: 1; tab-size: 4; }
.line-content pre { margin: 0; font-family: inherit; }

/* Placeholder pattern */
.empty-placeholder { background: repeating-linear-gradient(45deg, #f3f4f6, #f3f4f6 10px, #e5e7eb 10px, #e5e7eb 20px); opacity: 0.8; }
.dark .empty-placeholder { background: repeating-linear-gradient(45deg, #1f2937, #1f2937 10px, #374151 10px, #374151 20px); }

.diff-placeholder { flex: 1; display: flex; align-items: center; justify-content: center; background: url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCI+PGNpcmNsZSBjeD0iMjAiIGN5PSIyMCIgcj0iMSIgZmlsbD0icmdiYSgwLDAsMCwwLjA1KSIvPjwvc3ZnPg=='); }
.hint { padding: 20px; background: var(--vp-c-bg-soft); border: 1px dashed var(--vp-c-divider); border-radius: 8px; color: var(--vp-c-text-3); font-size: 13px; }

.fade-enter-active, .fade-leave-active { transition: opacity 0.3s; }
.fade-enter-from, .fade-leave-to { opacity: 0; }

.folder-layout { flex: 1; overflow: auto; padding: 20px; background: var(--vp-c-bg); }
.git-log-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.git-log-table th, .git-log-table td { border-bottom: 1px solid var(--vp-c-divider); padding: 10px; text-align: left; }
.git-log-table th { background: var(--vp-c-bg-soft); font-weight: 600; }
.mono { font-family: var(--vp-font-family-mono); color: var(--vp-c-brand); }
.small-btn { font-size: 11px; padding: 2px 6px; border: 1px solid var(--vp-c-divider); background: transparent; cursor: pointer; border-radius: 4px; }
.small-btn:hover { background: var(--vp-c-bg-soft); color: var(--vp-c-brand); }
</style>
