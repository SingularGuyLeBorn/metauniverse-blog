<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vitepress'

const visible = ref(false)
const position = ref({ x: 0, y: 0 })
const targetPath = ref('')
const isDir = ref(false)

const router = useRouter()

const emit = defineEmits(['open-history', 'open-folder-history'])

const handleContextMenu = (e: MouseEvent) => {
    // 查找最近的 sidebar link
    const link = (e.target as HTMLElement).closest('.VPSidebarItem .VPLink')
    
    // 如果不是 link，检查是否是文件夹 header
    const folder = (e.target as HTMLElement).closest('.VPSidebarItem .text')
    
    if (link) {
        e.preventDefault()
        const href = link.getAttribute('href')
        if (href) {
            targetPath.value = href.replace(/\.html$/, '') // remove .html
            isDir.value = false
            showMenu(e)
        }
    } else if (folder) {
        // 对于文件夹，我们需要获取其下的第一个链接或者通过其他方式判断路径
        // VitePress 侧边栏结构比较复杂，文件夹通常没有直接的 href
        // 我们尝试从父级结构推断，或者暂不支持文件夹右键（V1）
        // 但用户明确要求了文件夹历史。
        // 我们可以尝试找 folder 的 text 内容，或者最近的路径。
        // 简单实现：暂时只支持文件，文件夹如果点击的是 link 指向 index.md 的话也可以。
        // 如果是纯折叠组，没有 path，很难获取真实物理路径。
        // v1: 仅支持文件 (VPLink)
    }
}

const showMenu = (e: MouseEvent) => {
    position.value = { x: e.clientX, y: e.clientY }
    visible.value = true
}

const closeMenu = () => {
    visible.value = false
}

const copyPath = () => {
    navigator.clipboard.writeText(targetPath.value)
    closeMenu()
}

const viewHistory = () => {
    // 触发全局事件或路由跳转?
    // HistoryViewer 是全局组件，我们可以通过 emit 冒泡，或者 EventBus
    // 由于 Layout 结构，我们可以 dispatch 一个自定义事件
    window.dispatchEvent(new CustomEvent('open-history-viewer', { detail: { path: targetPath.value } }))
    closeMenu()
}

onMounted(() => {
    document.addEventListener('contextmenu', handleContextMenu)
    document.addEventListener('click', closeMenu)
})

onUnmounted(() => {
    document.removeEventListener('contextmenu', handleContextMenu)
    document.removeEventListener('click', closeMenu)
})
</script>

<template>
    <div v-if="visible" class="ctx-menu" :style="{ top: `${position.y}px`, left: `${position.x}px` }">
        <div class="menu-item" @click="viewHistory">📜 查看文件变更历史</div>
        <div class="menu-item" @click="copyPath">📋 复制文件路径</div>
        <!-- 暂未实现文件夹历史，待后续 FolderHistoryViewer 就绪 -->
    </div>
</template>

<style scoped>
.ctx-menu {
    position: fixed;
    z-index: 10000;
    background: var(--vp-c-bg);
    border: 1px solid var(--vp-c-divider);
    border-radius: 6px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    padding: 4px;
    min-width: 140px;
}
.menu-item {
    padding: 6px 12px;
    font-size: 13px;
    cursor: pointer;
    border-radius: 4px;
    color: var(--vp-c-text-1);
}
.menu-item:hover {
    background: var(--vp-c-bg-soft);
    color: var(--vp-c-brand);
}
</style>
