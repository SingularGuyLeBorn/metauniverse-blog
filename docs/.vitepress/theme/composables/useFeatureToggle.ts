import { ref, computed, readonly } from 'vue'
import { useLocalStorage } from '@vueuse/core'

export type FeatureFlag = 
  | 'knowledge-graph'
  | 'tensor-viz'
  | 'rag-search'
  | 'git-annotations'
  | 'wasm-sandbox'
  | 'heatmap'
  | 'deep-link'
  | 'dual-layout'

export interface FeatureMeta {
  name: string
  description: string
  defaultEnabled: boolean
}

export const featureMeta: Record<FeatureFlag, FeatureMeta> = {
  'knowledge-graph': { name: '知识图谱', description: '双向链接关系图', defaultEnabled: true },
  'tensor-viz': { name: '张量可视化', description: '3D张量数据可视化', defaultEnabled: true },
  'rag-search': { name: '智能搜索', description: '向量全文检索', defaultEnabled: true },
  'git-annotations': { name: 'Git注解', description: '段落级评论', defaultEnabled: true },
  'wasm-sandbox': { name: '代码沙箱', description: '浏览器内运行代码', defaultEnabled: true },
  'heatmap': { name: '热力图', description: '阅读热度分析', defaultEnabled: false },
  'deep-link': { name: '深度链接', description: 'URL状态持久化', defaultEnabled: true },
  'dual-layout': { name: '布局切换', description: '多种阅读布局', defaultEnabled: true }
}

const defaultFeatures: Record<FeatureFlag, boolean> = {
  'knowledge-graph': true,
  'tensor-viz': true,
  'rag-search': true,
  'git-annotations': true,
  'wasm-sandbox': true,
  'heatmap': false,
  'deep-link': true,
  'dual-layout': true
}

/**
 * 特性开关管理
 */
export function useFeatureToggle() {
  const features = useLocalStorage<Record<FeatureFlag, boolean>>(
    'mu-features',
    defaultFeatures,
    { mergeDefaults: true }
  )
  
  const isEnabled = (flag: FeatureFlag) => 
    computed(() => features.value[flag] ?? featureMeta[flag].defaultEnabled)
  
  const toggle = (flag: FeatureFlag): boolean => {
    const newValue = !features.value[flag]
    features.value[flag] = newValue
    return newValue
  }
  
  const enable = (flag: FeatureFlag) => {
    features.value[flag] = true
  }
  
  const disable = (flag: FeatureFlag) => {
    features.value[flag] = false
  }
  
  const enableAll = () => {
    (Object.keys(featureMeta) as FeatureFlag[]).forEach(flag => {
      features.value[flag] = true
    })
  }
  
  const resetToDefaults = () => {
    features.value = { ...defaultFeatures }
  }
  
  const enabledFeatures = computed(() => 
    (Object.keys(featureMeta) as FeatureFlag[]).filter(flag => features.value[flag])
  )
  
  return {
    features: readonly(features),
    isEnabled,
    enabledFeatures,
    toggle,
    enable,
    disable,
    enableAll,
    resetToDefaults
  }
}

export default useFeatureToggle
