<script setup lang="ts">
import { ref, onMounted, watch, computed } from 'vue'
import { useData } from 'vitepress'
import { useAnnotationStore } from '../../stores/annotation'

const { page } = useData()
const store = useAnnotationStore()

const historyList = ref<any[]>([])
const showModal = ref(false)
const selectedVersion = ref<any>(null)
const versionContent = ref('')
const originalContent = ref('')

const fetchHistory = async () => {
    const res = await fetch(`/api/list-history?path=${page.value.relativePath}`)
    if (res.ok) {
        const data = await res.json()
        historyList.value = data.history
    }
}

const fetchCurrent = async () => {
    const res = await fetch(`/api/read-md?path=${page.value.relativePath}`)
    if (res.ok) {
        const data = await res.json()
        originalContent.value = data.content
    }
}

const viewVersion = async (version: any) => {
    selectedVersion.value = version
    const res = await fetch(version.path)
    if (res.ok) {
        const data = await res.json()
        versionContent.value = data.content
    }
}

const performRollback = async () => {
    if (!selectedVersion.value || !confirm('确定要将该文档回滚到此历史版本并提交 Git 吗？')) return
    
    const res = await fetch('/api/rollback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            filePath: page.value.relativePath,
            historyFile: selectedVersion.value.name
        })
    })

    if (res.ok) {
        alert('回滚成功！页面即将重载。')
        location.reload()
    }
}

onMounted(() => {
    fetchHistory()
    fetchCurrent()
})

watch(() => page.value.relativePath, () => {
    fetchHistory()
    fetchCurrent()
})
</script>

<template>
    <div class="history-trigger">
        <button @click="showModal = true" title="查看本地版本控制历史">🌳 溯源</button>

        <Transition name="fade">
            <div v-if="showModal" class="history-modal-overlay" @click.self="showModal = false">
                <div class="history-modal">
                    <div class="modal-header">
                        <div class="header-main">
                            <h3>🌿 文档演进史: {{ page.title }}</h3>
                            <div class="sub-tip">点击条目查看与当前版本的差异</div>
                        </div>
                        <button class="close-btn" @click="showModal = false">×</button>
                    </div>
                    
                    <div class="modal-body">
                        <!-- 版本侧栏 -->
                        <div class="version-sidebar">
                            <div 
                                v-for="(h, index) in historyList" 
                                :key="h.name" 
                                class="history-card"
                                :class="{ active: selectedVersion?.name === h.name }"
                                @click="viewVersion(h)"
                            >
                                <div class="card-time">{{ new Date(h.time).toLocaleString() }}</div>
                                <div class="card-meta">
                                    <span>{{ index === 0 ? '最新备份' : '历史快讯' }}</span>
                                    <span class="commit-dot"></span>
                                </div>
                            </div>
                            <div v-if="historyList.length === 0" class="empty">暂无 Git 或本地备份历史</div>
                        </div>

                        <!-- 差异预览区 -->
                        <div class="diff-area">
                            <div v-if="selectedVersion" class="diff-header">
                                <span class="filename">{{ selectedVersion.name }}</span>
                                <button class="rollback-btn" @click="performRollback">一键回滚至此版本</button>
                            </div>
                            
                            <div v-if="versionContent" class="diff-container">
                                <!-- 此处应使用 vue-diff, 为简化逻辑先展示版本源码预览 -->
                                <div class="diff-view-placeholder">
                                    <div class="view-panel">
                                        <h4>历史版本 ({{ selectedVersion.name }})</h4>
                                        <pre><code>{{ versionContent }}</code></pre>
                                    </div>
                                    <div class="view-panel current">
                                        <h4>当前在线版本</h4>
                                        <pre><code>{{ originalContent }}</code></pre>
                                    </div>
                                </div>
                            </div>
                            <div v-else class="preview-placeholder">选择左侧时间轴查看文档差异</div>
                        </div>
                    </div>
                </div>
            </div>
        </Transition>
    </div>
</template>

<style scoped>
.history-trigger { position: fixed; right: 28px; bottom: 80px; z-index: 50; pointer-events: auto; }
.history-trigger button {
    background: var(--vp-c-bg);
    border: 1px solid var(--vp-c-divider);
    padding: 8px 16px;
    border-radius: 24px;
    font-size: 13px;
    cursor: pointer;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    transition: all 0.2s;
    font-weight: 600;
}
.history-trigger button:hover { border-color: var(--vp-c-brand); transform: scale(1.05); }

.history-modal-overlay { position: fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(0,0,0,0.7); display:flex; align-items:center; justify-content:center; z-index:1000; padding: 40px; }
.history-modal { background: var(--vp-c-bg); width: 95vw; height: 90vh; border-radius: 12px; display: flex; flex-direction: column; overflow: hidden; }

.modal-header { padding: 16px 24px; border-bottom: 1px solid var(--vp-c-divider); display: flex; justify-content: space-between; align-items: center; }
.header-main h3 { margin: 0; font-size: 18px; }
.sub-tip { font-size: 12px; color: var(--vp-c-text-3); margin-top: 4px; }
.close-btn { font-size: 28px; background: none; border: none; cursor: pointer; color: var(--vp-c-text-3); }

.modal-body { flex: 1; display: flex; overflow: hidden; }
.version-sidebar { width: 280px; border-right: 1px solid var(--vp-c-divider); overflow-y: auto; background: var(--vp-c-bg-alt); padding: 12px; }

.history-card {
    padding: 12px;
    border-radius: 8px;
    background: var(--vp-c-bg);
    border: 1px solid var(--vp-c-divider);
    margin-bottom: 12px;
    cursor: pointer;
    transition: all 0.2s;
}
.history-card:hover { border-color: var(--vp-c-brand); }
.history-card.active { border-color: var(--vp-c-brand); background: var(--vp-c-brand-soft); }
.card-time { font-size: 13px; font-weight: 500; margin-bottom: 6px; }
.card-meta { display: flex; justify-content: space-between; align-items: center; font-size: 11px; color: var(--vp-c-text-3); }
.commit-dot { width: 6px; height: 6px; background: #94a3b8; border-radius: 50%; }
.active .commit-dot { background: var(--vp-c-brand); }

.diff-area { flex: 1; display: flex; flex-direction: column; background: var(--vp-c-bg); }
.diff-header { padding: 8px 16px; background: var(--vp-c-bg-soft); display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--vp-c-divider); }
.filename { font-size: 12px; font-family: var(--vp-font-family-mono); color: var(--vp-c-text-2); }

.rollback-btn {
    background: var(--vp-c-brand);
    color: white;
    border: none;
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
}

.diff-view-placeholder { display: flex; flex: 1; overflow: hidden; }
.view-panel { flex: 1; display: flex; flex-direction: column; border-right: 1px solid var(--vp-c-divider); }
.view-panel.current { border-right: none; background: rgba(0,0,0,0.02); }
.view-panel h4 { margin: 0; padding: 10px 16px; font-size: 12px; background: var(--vp-c-bg-alt); }
.view-panel pre { flex: 1; margin: 0; padding: 16px; overflow: auto; font-size: 12px; line-height: 1.6; border-radius: 0; }

.preview-placeholder { flex: 1; display: flex; align-items: center; justify-content: center; color: var(--vp-c-text-3); font-style: italic; }

.fade-enter-active, .fade-leave-active { transition: opacity 0.3s; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
</style>
