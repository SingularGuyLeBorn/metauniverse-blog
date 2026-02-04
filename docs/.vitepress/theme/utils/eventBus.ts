import mitt from 'mitt'

/**
 * 事件类型定义 - 类型安全的模块间通信
 */
export type Events = {
  // 应用事件
  'app:mounted': void
  'app:theme:change': { theme: 'light' | 'dark' }
  'app:layout:change': { mode: 'default' | 'focus' | 'wide' }
  
  // 路由事件
  'router:before': { from: string; to: string }
  'router:after': { path: string }
  
  // 搜索事件
  'search:open': void
  'search:close': void
  'search:query': { query: string }
  'search:results': { results: unknown[]; query: string }
  
  // 内容事件
  'content:loaded': { path: string }
  'content:section:enter': { id: string; title: string }
  
  // 知识图谱事件
  'graph:node:click': { id: string; label: string }
  'graph:hover': { id: string; label: string }
  
  // 热力图事件
  'heatmap:update': { data: unknown[] }
  
  // 特性事件
  'feature:toggle': { feature: string; enabled: boolean }
}

export const eventBus = mitt<Events>()

export function emit<K extends keyof Events>(
  event: K,
  ...args: Events[K] extends void ? [] : [payload: Events[K]]
): void {
  if (args.length === 0) {
    eventBus.emit(event, undefined as Events[K])
  } else {
    eventBus.emit(event, args[0] as Events[K])
  }
}

export function on<K extends keyof Events>(
  event: K,
  handler: (payload: Events[K]) => void
): () => void {
  eventBus.on(event, handler as (e: unknown) => void)
  return () => eventBus.off(event, handler as (e: unknown) => void)
}

export function once<K extends keyof Events>(
  event: K,
  handler: (payload: Events[K]) => void
): void {
  const unsubscribe = on(event, (payload) => {
    handler(payload)
    unsubscribe()
  })
}

export function waitFor<K extends keyof Events>(event: K): Promise<Events[K]> {
  return new Promise((resolve) => {
    once(event, resolve)
  })
}

export default eventBus
