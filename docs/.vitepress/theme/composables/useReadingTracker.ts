import { ref, onMounted, onUnmounted, type Ref } from 'vue'

export interface ParagraphMetrics {
  id: string
  startTime: number | null
  totalTime: number
  highlightCount: number
  scrollIntoViewCount: number
}

export interface SessionData {
  articleId: string
  sessionId: string
  timestamp: number
  duration: number
  paragraphs: Array<{
    id: string
    time: number
    highlights: number
    views: number
  }>
}

/**
 * 阅读追踪 Composable - 特性5 语义热力图
 * 使用 Intersection Observer 追踪段落阅读行为
 */
export function useReadingTracker(articleId: string) {
  const paragraphs = ref<Map<string, ParagraphMetrics>>(new Map())
  const sessionId = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
  const startTime = Date.now()
  const visibleParagraphs = ref<Set<string>>(new Set())
  const observer = ref<IntersectionObserver | null>(null)

  function initTracking() {
    const elements = document.querySelectorAll('.vp-doc p, .vp-doc h2, .vp-doc h3')
    
    observer.value = new IntersectionObserver(
      (entries) => {
        const now = Date.now()
        
        entries.forEach(entry => {
          const id = entry.target.getAttribute('data-paragraph-id')
          if (!id) return
          
          const metrics = paragraphs.value.get(id)
          if (!metrics) return

          if (entry.isIntersecting && entry.intersectionRatio >= 0.5) {
            if (!metrics.startTime) {
              metrics.startTime = now
              metrics.scrollIntoViewCount++
              visibleParagraphs.value.add(id)
            }
          } else {
            if (metrics.startTime) {
              metrics.totalTime += now - metrics.startTime
              metrics.startTime = null
              visibleParagraphs.value.delete(id)
            }
          }
        })
      },
      {
        root: null,
        rootMargin: '-10% 0px -10% 0px',
        threshold: [0, 0.25, 0.5, 0.75, 1]
      }
    )

    elements.forEach((el, index) => {
      const id = `p-${index}`
      el.setAttribute('data-paragraph-id', id)
      el.classList.add('heatmap-paragraph')
      
      paragraphs.value.set(id, {
        id,
        startTime: null,
        totalTime: 0,
        highlightCount: 0,
        scrollIntoViewCount: 0
      })
      
      observer.value?.observe(el)
    })
  }

  function trackHighlight(paragraphId: string) {
    const metrics = paragraphs.value.get(paragraphId)
    if (metrics) {
      metrics.highlightCount++
    }
  }

  function getSessionData(): SessionData {
    const now = Date.now()
    
    // 更新当前可见段落的时间
    visibleParagraphs.value.forEach(id => {
      const metrics = paragraphs.value.get(id)
      if (metrics?.startTime) {
        metrics.totalTime += now - metrics.startTime
        metrics.startTime = now
      }
    })

    return {
      articleId,
      sessionId,
      timestamp: now,
      duration: now - startTime,
      paragraphs: Array.from(paragraphs.value.values()).map(p => ({
        id: p.id,
        time: p.totalTime,
        highlights: p.highlightCount,
        views: p.scrollIntoViewCount
      }))
    }
  }

  function getHeatmapData() {
    const maxTime = Math.max(...Array.from(paragraphs.value.values()).map(p => p.totalTime), 1000)
    
    return Array.from(paragraphs.value.values()).map(p => ({
      id: p.id,
      intensity: Math.min(p.totalTime / maxTime, 1),
      time: p.totalTime,
      views: p.scrollIntoViewCount
    }))
  }

  onMounted(() => {
    initTracking()
  })

  onUnmounted(() => {
    observer.value?.disconnect()
  })

  return {
    paragraphs,
    visibleParagraphs,
    trackHighlight,
    getSessionData,
    getHeatmapData
  }
}

export default useReadingTracker
