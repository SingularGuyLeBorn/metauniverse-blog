<template>
  <div class="tensor-playground">
    <div class="playground-header">
      <div class="title">💠 Tensor Playground</div>
      <div class="subtitle">Interactive Transformer Matrix Visualization</div>
    </div>

    <div class="controls">
      <div class="control-group">
        <label>Batch Size (B): {{ batchSize }}</label>
        <input type="range" v-model.number="batchSize" min="1" max="8" />
      </div>
      <div class="control-group">
        <label>Sequence Length (L): {{ seqLength }}</label>
        <input type="range" v-model.number="seqLength" min="1" max="10" />
      </div>
      <div class="control-group">
        <label>Hidden Dim (D): {{ hiddenDim }}</label>
        <input type="range" v-model.number="hiddenDim" min="4" max="16" step="4" />
      </div>
    </div>

    <div class="visualization">
      <div class="matrix-container">
        <div class="matrix-label">Input (X)</div>
        <div class="matrix-info">Shape: [{{ batchSize }}, {{ seqLength }}, {{ hiddenDim }}]</div>
        <div class="matrix-visual" :style="matrixStyle">
          <div v-for="b in batchSize" :key="b" class="matrix-batch">
            <div v-for="l in seqLength" :key="l" class="matrix-row">
              <div v-for="d in hiddenDim" :key="d" class="matrix-cell" :style="{ opacity: Math.random() * 0.5 + 0.2 }"></div>
            </div>
          </div>
        </div>
      </div>
      
      <div class="arrow">⬇️ Wq, Wk, Wv Projection</div>

      <div class="matrix-container">
        <div class="matrix-label">Q, K, V Matrices</div>
        <div class="matrix-info">Shape: [{{ batchSize }}, {{ seqLength }}, {{ hiddenDim }}]</div>
        <div class="matrix-group">
          <div class="mini-matrix q-matrix">Q</div>
          <div class="mini-matrix k-matrix">K</div>
          <div class="mini-matrix v-matrix">V</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

const batchSize = ref(2)
const seqLength = ref(4)
const hiddenDim = ref(8)

const matrixStyle = computed(() => ({
  display: 'grid',
  gridTemplateColumns: `repeat(${hiddenDim.value}, 1fr)`,
  gap: '2px',
  width: '100%',
  maxWidth: '300px'
}))
</script>

<style scoped>
.tensor-playground {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 1rem 0;
  font-family: monospace;
}

.playground-header {
  margin-bottom: 1.5rem;
  text-align: center;
}

.title {
  font-size: 1.2rem;
  font-weight: bold;
  background: linear-gradient(120deg, #bd34fe 30%, #41d1ff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.subtitle {
  font-size: 0.8rem;
  color: var(--vp-c-text-2);
}

.controls {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
  padding-bottom: 1rem;
  border-bottom: 1px dashed var(--vp-c-divider);
}

.control-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.control-group label {
  font-size: 0.9rem;
  color: var(--vp-c-text-2);
}

.visualization {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
}

.matrix-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  width: 100%;
}

.matrix-label {
  font-weight: bold;
  color: var(--vp-c-text-1);
}

.matrix-info {
  font-size: 0.8rem;
  color: var(--vp-c-text-2);
}

.matrix-batch {
  margin-bottom: 4px;
  border: 1px solid var(--vp-c-divider);
  padding: 2px;
}

.matrix-row {
  display: contents; /* Allows cells to sit in the grid defined by parent */
}

.matrix-visual {
  /* Grid defined in inline style */
}

.matrix-cell {
  aspect-ratio: 1;
  background-color: #3b82f6;
  border-radius: 2px;
}

.arrow {
  font-size: 0.8rem;
  color: var(--vp-c-text-3);
}

.matrix-group {
  display: flex;
  gap: 1rem;
}

.mini-matrix {
  width: 60px;
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: bold;
  border-radius: 4px;
}

.q-matrix { background-color: #ef4444; }
.k-matrix { background-color: #eab308; }
.v-matrix { background-color: #22c55e; }
</style>
