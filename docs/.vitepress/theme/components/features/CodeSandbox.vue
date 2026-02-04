<template>
  <div class="code-sandbox">
    <div class="sandbox-header">
      <span class="sandbox-title">🚀 WASM Sandbox</span>
      <div class="sandbox-actions">
        <button class="run-btn" @click="runCode" :disabled="isRunning || isLoading">
          <span v-if="isLoading">Loading Pyodide...</span>
          <span v-else-if="isRunning">Running...</span>
          <span v-else>Run ▶</span>
        </button>
      </div>
    </div>
    
    <div class="code-editor">
      <textarea v-model="code" spellcheck="false"></textarea>
    </div>
    
    <div class="console-output" v-if="output">
      <div class="console-label">Output ></div>
      <pre>{{ output }}</pre>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

const props = defineProps({
  initialCode: {
    type: String,
    default: 'import math\n\ndef attention(q, k, v):\n    return f"Attention Score: {math.sqrt(q*k)}"\n\nprint("Hello from WebAssembly!")\nprint(attention(10, 10, 10))'
  }
})

const code = ref(props.initialCode)
const output = ref('')
const isRunning = ref(false)
const isLoading = ref(false)
const pyodide = ref<any>(null)

async function loadPyodide() {
  if (pyodide.value) return
  
  isLoading.value = true
  try {
    // Dynamically load Pyodide script
    if (!window.loadPyodide) {
      const script = document.createElement('script')
      script.src = 'https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js'
      document.head.appendChild(script)
      
      await new Promise((resolve) => {
        script.onload = resolve
      })
    }
    
    // Initialize Pyodide

    pyodide.value = await window.loadPyodide()
    isLoading.value = false
  } catch (e) {
    output.value = `Error loading Pyodide: ${e}`
    isLoading.value = false
  }
}

async function runCode() {
  if (!pyodide.value) {
    await loadPyodide()
  }
  
  if (!pyodide.value) return
  
  isRunning.value = true
  output.value = ''
  
  try {
    // Capture stdout
    pyodide.value.setStdout({ batched: (msg: string) => { output.value += msg + '\n' } })
    
    await pyodide.value.runPythonAsync(code.value)
  } catch (e) {
    output.value = String(e)
  } finally {
    isRunning.value = false
  }
}
</script>

<style scoped>
.code-sandbox {
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  overflow: hidden;
  margin: 1.5rem 0;
  background: var(--vp-c-bg-soft);
}

.sandbox-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 1rem;
  background: var(--vp-c-bg-alt);
  border-bottom: 1px solid var(--vp-c-divider);
}

.sandbox-title {
  font-size: 0.9rem;
  font-weight: bold;
  opacity: 0.8;
}

.run-btn {
  background: var(--vp-c-brand);
  color: white;
  border: none;
  padding: 0.25rem 0.75rem;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.8rem;
  transition: opacity 0.2s;
}

.run-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.code-editor textarea {
  width: 100%;
  min-height: 150px;
  padding: 1rem;
  background: #1e1e1e;
  color: #d4d4d4;
  border: none;
  font-family: 'Fira Code', monospace;
  font-size: 0.9rem;
  resize: vertical;
}

.console-output {
  background: #000;
  color: #4ade80;
  padding: 1rem;
  font-family: monospace;
  font-size: 0.85rem;
  border-top: 1px solid var(--vp-c-divider);
}

.console-label {
  opacity: 0.6;
  font-size: 0.75rem;
  margin-bottom: 0.5rem;
}
</style>
