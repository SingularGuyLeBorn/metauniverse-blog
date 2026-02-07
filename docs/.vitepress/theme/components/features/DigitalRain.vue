<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { useLayoutStore } from '../../stores/layout'

const store = useLayoutStore()
const canvas = ref<HTMLCanvasElement | null>(null)
let intervalId: any = null

const startRain = () => {
  if (!canvas.value) return
  const ctx = canvas.value.getContext('2d')
  if (!ctx) return

  canvas.value.width = window.innerWidth
  canvas.value.height = window.innerHeight

  const katakana = 'アァカサタナハマヤャラワガザダバパイィキシチニヒミリヰギジヂビピウゥクスツヌフムユュルグズブヅプエェケセテネヘメレヱゲゼデベペオォコソトノホモヨョロヲゴゾドボポXH0123456789'
  const latin = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
  const nums = '0123456789'
  const alphabet = katakana + latin + nums

  const fontSize = 16
  const columns = canvas.value.width / fontSize

  const rainDrops: number[] = []
  for (let x = 0; x < columns; x++) {
    rainDrops[x] = 1
  }

  const draw = () => {
    if (!ctx || !canvas.value) return
    ctx.fillStyle = 'rgba(0, 0, 0, 0.05)'
    ctx.fillRect(0, 0, canvas.value.width, canvas.value.height)

    ctx.fillStyle = '#0F0' // Green
    ctx.font = fontSize + 'px monospace'

    for (let i = 0; i < rainDrops.length; i++) {
      const text = alphabet.charAt(Math.floor(Math.random() * alphabet.length))
      ctx.fillText(text, i * fontSize, rainDrops[i] * fontSize)

      if (rainDrops[i] * fontSize > canvas.value.height && Math.random() > 0.975) {
        rainDrops[i] = 0
      }
      rainDrops[i]++
    }
  }
  
  intervalId = setInterval(draw, 30)
}

const stopRain = () => {
  if (intervalId) clearInterval(intervalId)
  if (canvas.value) {
    const ctx = canvas.value.getContext('2d')
    ctx?.clearRect(0, 0, canvas.value.width, canvas.value.height)
  }
}

watch(() => store.digitalRainMode, (val) => {
  if (val) startRain()
  else stopRain()
})

onMounted(() => {
  if (store.digitalRainMode) startRain()
  window.addEventListener('resize', () => {
      if (store.digitalRainMode) startRain()
  })
})

onUnmounted(() => {
  stopRain()
})
</script>

<template>
  <canvas 
    ref="canvas" 
    class="digital-rain-canvas"
    :class="{ active: store.digitalRainMode }"
  ></canvas>
</template>

<style scoped>
.digital-rain-canvas {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  z-index: -1; /* Behind everything */
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.5s;
  background: black;
}

.digital-rain-canvas.active {
  opacity: 1;
  z-index: 10000; /* On top for full effect? Or background? */
  /* User probably wants background, but Matrix is usually immersive */
  /* Let's make it background but high opacity, or overlay with mix-blend-mode */
  /* Task says "Code Rain Background". Background implies z-index -1 */
  z-index: 0; 
}

/* If active, we might need to make other backgrounds transparent? */
/* That's complex. Let's make it an overlay with low opacity? */
/* Or just z-index 9999 with pointer-events none and mix-blend-mode screen? */
.digital-rain-canvas.active {
    z-index: 9999;
    background: transparent;
    opacity: 0.15; /* Subtle overlay */
    mix-blend-mode: screen; 
}
/* Wait, if it's "Background", it should be behind content. */
/* Let's stick to z-index -1 and assume body bg allows it see through? */
/* content has bg color. So it won't be visible. */
/* So overlay is better. */
</style>
