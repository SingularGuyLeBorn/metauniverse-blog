/**
 * 热力图颜色工具
 */

export interface ColorStop {
  stop: number
  color: [number, number, number]
}

export const defaultGradient: ColorStop[] = [
  { stop: 0, color: [59, 130, 246] },    // blue-500
  { stop: 0.3, color: [34, 197, 94] },   // green-500
  { stop: 0.6, color: [234, 179, 8] },   // yellow-500
  { stop: 1, color: [239, 68, 68] }      // red-500
]

export function getHeatColor(intensity: number, gradient: ColorStop[] = defaultGradient): string {
  for (let i = 0; i < gradient.length - 1; i++) {
    const curr = gradient[i]
    const next = gradient[i + 1]
    
    if (intensity >= curr.stop && intensity <= next.stop) {
      const t = (intensity - curr.stop) / (next.stop - curr.stop)
      const r = Math.round(curr.color[0] + (next.color[0] - curr.color[0]) * t)
      const g = Math.round(curr.color[1] + (next.color[1] - curr.color[1]) * t)
      const b = Math.round(curr.color[2] + (next.color[2] - curr.color[2]) * t)
      return `rgba(${r}, ${g}, ${b}, ${0.3 + intensity * 0.7})`
    }
  }
  
  const last = gradient[gradient.length - 1]
  return `rgba(${last.color[0]}, ${last.color[1]}, ${last.color[2]}, 1)`
}

/**
 * 防抖函数
 */
export function debounce<T extends (...args: any[]) => void>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout>
  return (...args) => {
    clearTimeout(timeoutId)
    timeoutId = setTimeout(() => fn(...args), delay)
  }
}

/**
 * 节流函数
 */
export function throttle<T extends (...args: any[]) => void>(
  fn: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle = false
  return (...args) => {
    if (!inThrottle) {
      fn(...args)
      inThrottle = true
      setTimeout(() => inThrottle = false, limit)
    }
  }
}
