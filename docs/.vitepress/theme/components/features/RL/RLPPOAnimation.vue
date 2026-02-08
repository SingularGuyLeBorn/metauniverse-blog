<template>
  <ClientOnly>
    <PPOOrchestrator />
  </ClientOnly>
</template>

<script setup>
import { defineAsyncComponent } from 'vue'

const PPOOrchestrator = defineAsyncComponent(() => 
  import('./ppo/PPOOrchestrator.vue')
)
</script>