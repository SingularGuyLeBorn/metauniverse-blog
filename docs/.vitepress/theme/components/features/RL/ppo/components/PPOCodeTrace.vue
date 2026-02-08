<template>
  <div class="code-trace-container">
    <div class="code-header">
      <span class="file-name">📄 02_Implementation.py</span>
      <span class="phase-badge">{{ phaseName }}</span>
    </div>
    <div class="code-content">
      <pre><code v-html="highlightedCode"></code></pre>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';

const props = defineProps<{
  phase: string;
}>();

const codeSnippets: Record<string, { name: string; code: string; highlight: number[] }> = {
  SCENE1_Init: { name: 'Env', code: `env = GridWorld()
state = env.reset()`, highlight: [2] },
  
  SCENE2_StateSpawn: { name: 'Observe', code: `s_t = torch.tensor(state)
# Shape: [4]`, highlight: [1] },

  SCENE3_Split: { name: 'Input', code: `# Dual Heads Input
actor_in = s_t
critic_in = s_t`, highlight: [2, 3] },

  SCENE4_Critic1: { name: 'Value Est', code: `with torch.no_grad():
    v_t = agent.get_value(s_t)
    # v_t = 12.50`, highlight: [2] },

  SCENE5_Actor: { name: 'Policy', code: `logits = agent.actor(s_t)
dist = Categorical(logits=logits)`, highlight: [1, 2] },

  SCENE6_ActionForm: { name: 'Sample', code: `a_t = dist.sample()
# a_t = [0, 1, 0, 0]`, highlight: [1] },

  SCENE7_Interact: { name: 'Env Step', code: `s_next, r_t, done, _ = env.step(a_t)
# r_t = +1.0`, highlight: [1] },

  SCENE8_Critic2: { name: 'Buffer & V_next', code: `buffer.add(s, a, r, s_next)
with torch.no_grad():
    v_next = agent.get_value(s_next)`, highlight: [1, 3] },

  SCENE9_Math: { name: 'GAE Update', code: `delta = r_t + gamma * v_next - v_t
adv = delta + gamma * lam * next_adv
agent.update(adv)`, highlight: [1, 2, 3] }
};

const currentSnippet = computed(() => {
  return codeSnippets[props.phase] || { name: 'Wait', code: '# Waiting...', highlight: [] };
});

const phaseName = computed(() => currentSnippet.value.name);

const highlightedCode = computed(() => {
  const lines = currentSnippet.value.code.split('\n');
  const highlightLines = currentSnippet.value.highlight;
  
  return lines.map((line, i) => {
    const lineNum = i + 1;
    const isHighlight = highlightLines.includes(lineNum);
    const lineNumStr = String(lineNum).padStart(2, ' ');
    const escapedLine = escapeHtml(line);
    const styledLine = syntaxHighlight(escapedLine);
    
    return `<span class="line ${isHighlight ? 'highlight' : ''}"><span class="line-num">${lineNumStr}</span>${styledLine}</span>`;
  }).join('\n');
});

function escapeHtml(str: string): string {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function syntaxHighlight(line: string): string {
  // Simple syntax highlighter
  line = line.replace(/\b(def|class|return|if|for|in|import|from|with|as|else|elif|None|True|False|self|agent|env)\b/g, '<span class="keyword">$1</span>');
  line = line.replace(/\b(int|float|str|bool|torch|tensor)\b/g, '<span class="type">$1</span>');
  line = line.replace(/(#.*)$/g, '<span class="comment">$1</span>');
  return line;
}
</script>

<style scoped>
.code-trace-container {
  font-family: 'JetBrains Mono', monospace;
  background: #1e293b;
  border-radius: 12px;
  overflow: hidden;
  height: 100%;
}
.code-header {
  background: #0f172a;
  padding: 12px;
  display: flex;
  justify-content: space-between;
  border-bottom: 1px solid #334155;
}
.file-name { color: #94a3b8; font-size: 12px; }
.phase-badge { background: #3b82f6; color: white; padding: 2px 8px; border-radius: 4px; font-size: 10px; }
.code-content { padding: 16px; overflow: auto; }
pre { margin: 0; font-size: 12px; line-height: 1.5; }
code { color: #e2e8f0; }
:deep(.line.highlight) { background: rgba(59, 130, 246, 0.2); border-left: 2px solid #3b82f6; }
:deep(.line-num) { color: #475569; margin-right: 12px; }
:deep(.keyword) { color: #c084fc; }
:deep(.type) { color: #22d3ee; }
:deep(.comment) { color: #64748b; }
</style>
