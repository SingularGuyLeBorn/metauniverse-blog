import { ref, readonly, watch } from 'vue'
import { useUrlSearchParams, useDebounceFn, useLocalStorage } from '@vueuse/core'
import { compress, decompress } from 'lz-string'

export interface DeepLinkState {
  scrollPosition?: number
  activeSection?: string
  searchQuery?: string
  selectedTags?: string[]
  expandedPanels?: string[]
  customData?: Record<string, unknown>
}

export interface DeepLinkOptions {
  paramKey?: string
  enableStorage?: boolean
  storageKey?: string
  debounceDelay?: number
  compress?: boolean
}

const defaultOptions = {
  paramKey: 'state',
  enableStorage: true,
  storageKey: 'mu-deep-link',
  debounceDelay: 300,
  compress: true
}

/**
 * 深度链接状态管理 - 特性1
 * 将应用状态持久化到URL和localStorage
 */
export function useDeepLink(options: DeepLinkOptions = {}) {
  const opts = { ...defaultOptions, ...options }
  const params = useUrlSearchParams('hash-params')
  const storageState = opts.enableStorage 
    ? useLocalStorage<DeepLinkState>(opts.storageKey, {})
    : ref<DeepLinkState>({})
  const state = ref<DeepLinkState>({})

  const parseFromUrl = (): DeepLinkState => {
    const compressed = params[opts.paramKey] as string | undefined
    if (!compressed) return {}
    
    try {
      const json = opts.compress ? (decompress(compressed) || '') : decodeURIComponent(compressed)
      return json ? JSON.parse(json) : {}
    } catch {
      return {}
    }
  }

  const serializeToUrl = useDebounceFn((newState: DeepLinkState) => {
    try {
      const json = JSON.stringify(newState)
      const value = opts.compress ? compress(json) : encodeURIComponent(json)
      params[opts.paramKey] = value
    } catch (error) {
      console.error('[useDeepLink] Failed to serialize:', error)
    }
  }, opts.debounceDelay)

  const setState = (newState: DeepLinkState) => {
    state.value = newState
  }

  const mergeState = (partial: Partial<DeepLinkState>) => {
    state.value = { ...state.value, ...partial }
  }

  const clearState = () => {
    state.value = {}
    delete params[opts.paramKey]
    if (opts.enableStorage) {
      storageState.value = {}
    }
  }

  const share = (): string => window.location.href

  const copyShareLink = async (): Promise<boolean> => {
    try {
      await navigator.clipboard.writeText(share())
      return true
    } catch {
      return false
    }
  }

  const autoRestore = (): boolean => {
    const urlState = parseFromUrl()
    if (Object.keys(urlState).length > 0) {
      state.value = urlState
      return true
    }
    if (opts.enableStorage && Object.keys(storageState.value).length > 0) {
      state.value = storageState.value
      return true
    }
    return false
  }

  // 初始化
  autoRestore()

  // 监听变化并同步
  watch(state, (newState) => {
    serializeToUrl(newState)
    if (opts.enableStorage) {
      storageState.value = newState
    }
  }, { deep: true })

  return {
    state: readonly(state),
    setState,
    mergeState,
    clearState,
    share,
    copyShareLink,
    autoRestore
  }
}

export default useDeepLink
