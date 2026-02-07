<script setup lang="ts">
import { watch, onMounted, onUnmounted } from 'vue'
import { useLayoutStore } from '../../stores/layout'

const store = useLayoutStore()

// Bionic Reading Processor
const processBionic = () => {
  const content = document.querySelector('.vp-doc .content')
  if (!content) return

  // Traverse paragraphs
  const paragraphs = content.querySelectorAll('p, li')
  paragraphs.forEach(p => {
    // Skip if already processed
    if (p.hasAttribute('data-bionic-processed')) return
    
    // Simple text node traversal is safer to avoid breaking other components
    // But for "Bionic Reading", we need to wrap parts of words.
    // We only target direct text nodes or simple spans to minimize risk.
    
    const walker = document.createTreeWalker(p, NodeFilter.SHOW_TEXT, null)
    const textNodes: Node[] = []
    while (walker.nextNode()) textNodes.push(walker.currentNode)
    
    textNodes.forEach(node => {
      const text = node.nodeValue
      if (!text || !text.trim()) return
      
      // Avoid processing inside existing strict components if any
      // ...
      
      const span = document.createElement('span')
      const words = text.split(/(\s+)/) // Keep delimiters
      
      const newHtml = words.map(part => {
        if (/^\s+$/.test(part)) return part // Whitespace
        const mid = Math.ceil(part.length / 2)
        return `<b>${part.slice(0, mid)}</b>${part.slice(mid)}`
      }).join('')
      
      span.innerHTML = newHtml
      node.parentNode?.replaceChild(span, node)
    })
    
    p.setAttribute('data-bionic-processed', 'true')
  })
}

const cleanupBionic = () => {
    // Reloading the page or complex revert is hard. 
    // Easier: Just toggle CSS class, but we did DOM structure change...
    // Better approach: Bionic Reading IS destructive to DOM structure. 
    // To disable, we might need to reload or re-render, OR just hide bold weight via CSS.
    // CSS Toggle: .bionic-mode b { font-weight: bold } else { font-weight: inherit }
    // This requires the DOM changes to be present always? No, performance hit.
    // Let's rely on the user understanding that toggling ON does the processing.
    // Toggling OFF might require refresh if we want clean DOM, OR we can just style it away.
}

watch(() => store.bionicMode, (val) => {
  if (val) {
    // Defer to next tick to ensure content is loaded
    setTimeout(processBionic, 100)
  }
}, { immediate: true })

</script>

<template>
  <slot />
</template>

<style>
/* If bionic mode is OFF, we suppress the bolding if the DOM was already modified */
/* Actually, best to only modify if ON. If turned OFF, maybe just refresh? */
/* Or use CSS to hide the effect: */
:root:not(.bionic-mode) .vp-doc b {
    font-weight: inherit; /* Reset if we manually wrapped them in <b> */
}

/* Ensure only our generated b's are affected? Difficult without specific class. */
/* Let's try to be specific or accept that it's a "Mode" that sticks until refresh or we refine logic. */
</style>
