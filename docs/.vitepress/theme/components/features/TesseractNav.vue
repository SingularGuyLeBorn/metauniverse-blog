<script setup lang="ts">
import { ref } from 'vue'

const show = ref(true) // Always show or toggle?

// Simple 3D Cube with CSS
</script>

<template>
  <div class="tesseract-container">
    <div class="scene">
      <div class="cube">
        <div class="cube-face front"><a href="/posts/">Posts</a></div>
        <div class="cube-face back"><a href="/knowledge/">Know</a></div>
        <div class="cube-face right"><a href="/papers/">Papers</a></div>
        <div class="cube-face left"><a href="/about/">About</a></div>
        <div class="cube-face top">Meta</div>
        <div class="cube-face bottom">Verse</div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.tesseract-container {
  position: fixed;
  bottom: 80px;
  right: 30px;
  width: 60px;
  height: 60px;
  perspective: 400px;
  z-index: 90;
}

.scene {
  width: 100%;
  height: 100%;
  position: relative;
  transform-style: preserve-3d;
  animation: spin 10s infinite linear;
}

.cube {
  width: 100%;
  height: 100%;
  position: absolute;
  transform-style: preserve-3d;
}

.cube:hover .scene {
    animation-play-state: paused;
}

.cube-face {
  position: absolute;
  width: 60px;
  height: 60px;
  border: 1px solid var(--vp-c-brand);
  background: rgba(var(--vp-c-brand-rgb), 0.1);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: bold;
  color: var(--vp-c-brand);
  backdrop-filter: blur(2px);
  box-shadow: 0 0 10px rgba(var(--vp-c-brand-rgb), 0.2);
}

.cube-face a {
    color: inherit;
    text-decoration: none;
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.front  { transform: rotateY(  0deg) translateZ(30px); }
.right  { transform: rotateY( 90deg) translateZ(30px); }
.back   { transform: rotateY(180deg) translateZ(30px); }
.left   { transform: rotateY(-90deg) translateZ(30px); }
.top    { transform: rotateX( 90deg) translateZ(30px); }
.bottom { transform: rotateX(-90deg) translateZ(30px); }

@keyframes spin {
  0% { transform: rotateX(0deg) rotateY(0deg); }
  100% { transform: rotateX(360deg) rotateY(360deg); }
}
</style>
