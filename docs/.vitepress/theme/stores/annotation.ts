import { defineStore } from 'pinia'
import { ref } from 'vue'

export type EditorStatus = 'synced' | 'syncing' | 'error'

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
    const editorStatus = ref<EditorStatus>('synced')
    const currentPath = ref('')
    const pendingContent = ref('')
    
    // Undo/Redo 栈
    const undoStack = ref<string[]>([])
    const redoStack = ref<string[]>([])
    const MAX_STACK_SIZE = 50
    
    // 历史记录
    const history = ref<EditorChange[]>([])

    const setEditing = (val: boolean) => {
        isEditing.value = val
    }

    const updatePendingContent = (path: string, content: string) => {
        // 只有内容真正改变时才记录快照
        if (content !== pendingContent.value) {
            // 在更新前，将当前旧内容压入撤回栈
            if (pendingContent.value) {
                undoStack.value.push(pendingContent.value)
                if (undoStack.value.length > MAX_STACK_SIZE) undoStack.value.shift()
                // 每次有新操作时，清空重做栈
                redoStack.value = []
            }
            
            currentPath.value = path
            pendingContent.value = content
            editorStatus.value = 'syncing'
        }
    }

    const undo = () => {
        if (undoStack.value.length === 0) return
        
        // 将当前内容压入重做栈
        redoStack.value.push(pendingContent.value)
        
        // 弹出撤回栈顶
        pendingContent.value = undoStack.value.pop()!
        editorStatus.value = 'syncing'
    }

    const redo = () => {
        if (redoStack.value.length === 0) return
        
        // 将当前内容压回撤回栈
        undoStack.value.push(pendingContent.value)
        
        // 弹出重做栈顶
        pendingContent.value = redoStack.value.pop()!
        editorStatus.value = 'syncing'
    }

    const setSynced = () => {
        editorStatus.value = 'synced'
    }

    const resetChanges = () => {
        editorStatus.value = 'synced'
        pendingContent.value = ''
    }

    return {
        isEditing,
        editorStatus,
        currentPath,
        pendingContent,
        history,
        undoStack,
        redoStack,
        setEditing,
        updatePendingContent,
        undo,
        redo,
        setSynced,
        resetChanges
    }
})
