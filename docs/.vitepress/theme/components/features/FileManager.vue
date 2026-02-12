<script setup lang="ts">
import { ref, onMounted } from 'vue'

const currentDir = ref('docs')
const items = ref<any[]>([])
const showManager = ref(false)

const fetchFiles = async () => {
    const res = await fetch(`/api/files/list?dir=${currentDir.value}`)
    if (res.ok) {
        items.value = await res.json()
    }
}

const navigate = (path: string) => {
    currentDir.value = path
    fetchFiles()
}

const goBack = () => {
    const parts = currentDir.value.split('/')
    if (parts.length > 1) {
        parts.pop()
        currentDir.value = parts.join('/')
        fetchFiles()
    }
}

const deleteItem = async (path: string) => {
    if (!confirm('确定删除此文件并提交 Git 吗？')) return
    const res = await fetch('/api/files/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filePath: path })
    })
    if (res.ok) fetchFiles()
}

const createNew = async () => {
    const name = prompt('输入新文件/文件夹名称:')
    if (!name) return
    const isDir = name.includes('.') ? false : confirm('是否为文件夹？')
    
    const res = await fetch('/api/files/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ parentDir: currentDir.value, name, isDir })
    })
    if (res.ok) fetchFiles()
}

onMounted(fetchFiles)
</script>

<template>
    <div class="file-manager-trigger">
        <button @click="showManager = !showManager" title="项目文件管理">📁 资源</button>

        <Transition name="fade">
            <div v-if="showManager" class="manager-modal" @click.self="showManager = false">
                <div class="manager-content">
                    <div class="manager-header">
                        <div class="path-bar">
                            <button @click="goBack" :disabled="currentDir === 'docs'">⬅</button>
                            <span>{{ currentDir }}</span>
                        </div>
                        <div class="mgr-actions">
                            <button @click="createNew" class="create-btn">+ 新建</button>
                            <button class="close" @click="showManager = false">×</button>
                        </div>
                    </div>

                    <div class="manager-list">
                        <div v-for="item in items" :key="item.path" class="file-item">
                            <div class="item-info" @click="item.isDir ? navigate(item.path) : null">
                                <span class="icon">{{ item.isDir ? '📁' : '📄' }}</span>
                                <span class="name">{{ item.name }}</span>
                            </div>
                            <div class="item-ops">
                                <button @click="deleteItem(item.path)" class="del-btn">🗑️</button>
                            </div>
                        </div>
                        <div v-if="items.length === 0" class="empty">目录为空</div>
                    </div>
                </div>
            </div>
        </Transition>
    </div>
</template>

<style scoped>
.file-manager-trigger { position: fixed; right: 28px; bottom: 130px; z-index: 45; }
.file-manager-trigger button {
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

.manager-modal { position: fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(0,0,0,0.4); z-index:1000; display:flex; align-items:center; justify-content:center; }
.manager-content { background: var(--vp-c-bg); width: 400px; height: 500px; border-radius: 12px; box-shadow: 0 12px 32px rgba(0,0,0,0.3); display: flex; flex-direction: column; overflow: hidden; }

.manager-header { padding: 12px; border-bottom: 1px solid var(--vp-c-divider); display: flex; justify-content: space-between; align-items: center; background: var(--vp-c-bg-soft); }
.path-bar { display: flex; align-items: center; gap: 8px; font-size: 12px; font-family: var(--vp-font-family-mono); overflow: hidden; }
.path-bar button { border: none; background: none; cursor: pointer; padding: 0 4px; }

.mgr-actions { display: flex; gap: 8px; align-items: center; }
.create-btn { font-size: 11px; background: var(--vp-c-brand); color: white; border: none; padding: 2px 8px; border-radius: 4px; cursor: pointer; }
.close { font-size: 20px; border: none; background: none; cursor: pointer; color: var(--vp-c-text-3); }

.manager-list { flex: 1; overflow-y: auto; padding: 8px; }
.file-item { display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; border-radius: 6px; transition: background 0.2s; }
.file-item:hover { background: var(--vp-c-bg-alt); }

.item-info { display: flex; align-items: center; gap: 10px; cursor: pointer; flex: 1; }
.icon { font-size: 14px; }
.name { font-size: 13px; color: var(--vp-c-text-1); }

.item-ops button { border: none; background: none; cursor: pointer; padding: 4px; opacity: 0; transition: opacity 0.2s; }
.file-item:hover .item-ops button { opacity: 1; }
.del-btn:hover { color: #ef4444; }

.empty { text-align: center; padding: 40px; color: var(--vp-c-text-3); font-size: 12px; }

.fade-enter-active, .fade-leave-active { transition: opacity 0.2s, transform 0.2s; }
.fade-enter-from, .fade-leave-to { opacity: 0; transform: scale(0.95); }
</style>
