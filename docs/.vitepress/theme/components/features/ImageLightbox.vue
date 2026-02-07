<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'

const isOpen = ref(false)
const currentImage = ref<string | null>(null)
const imageAlt = ref<string>('')

const openLightbox = (e: MouseEvent) => {
  const target = e.target as HTMLElement
  if (target.tagName === 'IMG' && target.closest('.vp-doc')) {
    e.preventDefault()
    e.stopPropagation()
    const img = target as HTMLImageElement
    currentImage.value = img.src
    imageAlt.value = img.alt
    isOpen.value = true
    document.body.style.overflow = 'hidden'
  }
}

const closeLightbox = () => {
  isOpen.value = false
  currentImage.value = null
  document.body.style.overflow = ''
}

const onKeydown = (e: KeyboardEvent) => {
  if (e.key === 'Escape' && isOpen.value) {
    closeLightbox()
  }
}

onMounted(() => {
  if (typeof window !== 'undefined') {
    document.addEventListener('click', openLightbox)
    document.addEventListener('keydown', onKeydown)
  }
})

onUnmounted(() => {
  if (typeof window !== 'undefined') {
    document.removeEventListener('click', openLightbox)
    document.removeEventListener('keydown', onKeydown)
  }
})
</script>

<template>
  <Teleport to="body">
    <Transition name="fade">
      <div v-if="isOpen" class="lightbox-overlay" @click="closeLightbox">
        <div class="lightbox-content">
          <img :src="currentImage!" :alt="imageAlt" class="lightbox-image" @click.stop />
          <p v-if="imageAlt" class="lightbox-caption">{{ imageAlt }}</p>
        </div>
        <button class="close-btn" @click="closeLightbox">×</button>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.lightbox-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0, 0, 0, 0.9);
  z-index: 2000;
  display: flex;
  justify-content: center;
  align-items: center;
  cursor: zoom-out;
}

.lightbox-content {
  max-width: 90vw;
  max-height: 90vh;
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.lightbox-image {
  max-width: 100%;
  max-height: 85vh;
  object-fit: contain;
  box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
  cursor: default;
  border-radius: 4px;
}

.lightbox-caption {
  margin-top: 1rem;
  color: #fff;
  font-size: 0.9rem;
  text-align: center;
  font-family: var(--vp-font-family-mono);
}

.close-btn {
  position: absolute;
  top: 2rem;
  right: 2rem;
  background: none;
  border: none;
  color: rgba(255, 255, 255, 0.7);
  font-size: 3rem;
  line-height: 1;
  cursor: pointer;
  padding: 0;
  transition: color 0.2s;
}

.close-btn:hover {
  color: #fff;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
