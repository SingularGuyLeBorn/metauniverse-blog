import { defineStore } from 'pinia'
import { ref } from 'vue'

export type EditorStatus = 'none' | 'staged' | 'committed'

export interface EditorChange {
    path: string
    originalContent: string
    modifiedContent: string
    timestamp: number
    message: string
}

export const useAnnotationStore = defineStore('annotation', () => {
    // 基础状态
    const isEditing = ref(false)
    const editorStatus = ref<EditorStatus>('none')
    const currentPath = ref('')
    const pendingContent = ref('')
    const stagedContent = ref('') // 暂存区内容
    
    // 历史记录
    const history = ref<EditorChange[]>([])

    const setEditing = (val: boolean) => {
        isEditing.value = val
    }

    const updatePendingContent = (path: string, content: string) => {
        currentPath.value = path
        pendingContent.value = content
        editorStatus.value = 'staged'
    }

    const commitChanges = () => {
        editorStatus.value = 'committed'
        stagedContent.value = pendingContent.value
    }

    const resetChanges = () => {
        editorStatus.value = 'none'
        pendingContent.value = ''
        stagedContent.value = ''
    }

    return {
        isEditing,
        editorStatus,
        currentPath,
        pendingContent,
        stagedContent,
        history,
        setEditing,
        updatePendingContent,
        commitChanges,
        resetChanges
    }
})
