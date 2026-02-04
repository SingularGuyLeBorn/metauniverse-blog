<template>
  <div class="rag-search-container">
    <button class="rag-trigger" @click="isOpen = true">
      🤖 Ask AI Assistant
    </button>

    <Teleport to="body">
      <div v-if="isOpen" class="rag-modal-overlay" @click.self="isOpen = false">
        <div class="rag-modal">
          <div class="rag-header">
            <h3>🧠 MetaUniverse RAG Agent</h3>
            <button class="close-btn" @click="isOpen = false">×</button>
          </div>
          
          <div class="chat-viewport">
            <div v-for="(msg, i) in history" :key="i" class="chat-message" :class="msg.role">
              <div class="avatar">{{ msg.role === 'user' ? '👤' : '🤖' }}</div>
              <div class="content">
                <div v-html="renderMarkdown(msg.content)"></div>
                <div v-if="msg.sources" class="sources">
                  <strong>References:</strong>
                  <ul>
                    <li v-for="source in msg.sources" :key="source.title">
                      <a :href="source.link">{{ source.title }}</a>
                    </li>
                  </ul>
                </div>
              </div>
            </div>
            <div v-if="isThinking" class="chat-message ai thinking">
              <div class="avatar">🤖</div>
              <div class="content">Thinking... Analysing vector database...</div>
            </div>
          </div>

          <div class="input-area">
            <input 
              v-model="query" 
              @keydown.enter="sendQuery" 
              placeholder="Ask me anything about these articles..." 
              :disabled="isThinking"
            />
            <button @click="sendQuery" :disabled="!query || isThinking">Send</button>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const isOpen = ref(false)
const query = ref('')
const isThinking = ref(false)
const history = ref([
  { role: 'ai', content: 'Hello! I am the Digital Twin of this blog. Ask me about Transformers, LLM architectures, or anything else written here.' }
])

function renderMarkdown(text: string) {
  // Simple replacement for demo
  return text.replace(/\*\*(.*?)\*\*/g, '<b>$1</b>')
}

// Simulated RAG Logic
async function sendQuery() {
  if (!query.value.trim()) return
  
  const userQ = query.value
  history.value.push({ role: 'user', content: userQ })
  query.value = ''
  isThinking.value = true
  
  // Fake network delay
  await new Promise(r => setTimeout(r, 1500))
  
  let response = "I'm not sure about that. Try asking about Transformers."
  let sources = []
  
  if (userQ.toLowerCase().includes('transformer') || userQ.toLowerCase().includes('attention')) {
    response = "The **Transformer** architecture relies heavily on the **Self-Attention** mechanism. Unlike RNNs, it processes the entire sequence in parallel, allowing for much faster training and better handling of long-range dependencies."
    sources = [
      { title: 'Transformer Architecture', link: '/posts/transformer' },
      { title: 'Attention is All You Need', link: 'https://arxiv.org/abs/1706.03762' }
    ]
  } else if (userQ.toLowerCase().includes('mla')) {
    response = "**MLA (Multi-Head Latent Attention)** is an optimization to reduce the KV Cache memory footprint while maintaining performance."
    sources = [
      { title: 'MLA Evolution', link: '#' }
    ]
  }
  
  history.value.push({ role: 'ai', content: response, sources })
  isThinking.value = false
}
</script>

<style scoped>
.rag-trigger {
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 20px;
  cursor: pointer;
  font-weight: bold;
  font-size: 0.9rem;
  transition: transform 0.2s;
  position: fixed;
  bottom: 20px;
  right: 20px;
  z-index: 100;
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
}

.rag-trigger:hover {
  transform: scale(1.05);
}

.rag-modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0,0,0,0.6);
  z-index: 2000;
  display: flex;
  align-items: center;
  justify-content: center;
}

.rag-modal {
  width: 500px;
  height: 600px;
  background: var(--vp-c-bg);
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  box-shadow: 0 10px 40px rgba(0,0,0,0.3);
}

.rag-header {
  padding: 1rem;
  background: var(--vp-c-bg-alt);
  border-bottom: 1px solid var(--vp-c-divider);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.rag-header h3 {
  margin: 0;
  font-size: 1.1rem;
}

.close-btn {
  background: none;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  color: var(--vp-c-text-2);
}

.chat-viewport {
  flex: 1;
  padding: 1rem;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.chat-message {
  display: flex;
  gap: 0.8rem;
  animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.chat-message.user {
  flex-direction: row-reverse;
}

.avatar {
  font-size: 1.5rem;
}

.content {
  background: var(--vp-c-bg-alt);
  padding: 0.8rem;
  border-radius: 8px;
  max-width: 80%;
  font-size: 0.9rem;
  line-height: 1.5;
}

.chat-message.user .content {
  background: var(--vp-c-brand);
  color: white;
}

.sources {
  margin-top: 0.5rem;
  font-size: 0.8rem;
  color: var(--vp-c-text-2);
  border-top: 1px solid var(--vp-c-divider);
  padding-top: 0.5rem;
}

.sources ul {
  margin: 0;
  padding-left: 1rem;
}

.input-area {
  padding: 1rem;
  border-top: 1px solid var(--vp-c-divider);
  display: flex;
  gap: 0.5rem;
}

input {
  flex: 1;
  padding: 0.6rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
  background: var(--vp-c-bg-alt);
}

.input-area button {
  padding: 0 1rem;
  background: var(--vp-c-brand);
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}
</style>
