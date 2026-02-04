<template>
  <div class="related-references" v-if="hasReferences">
    <div class="references-header">
      <span class="header-icon">🔗</span>
      <span class="header-title">关联笔记</span>
      <button 
        v-if="allLinks.length > 0" 
        class="graph-toggle"
        @click="showGraph = !showGraph"
        :title="showGraph ? '隐藏知识图谱' : '显示知识图谱'"
      >
        {{ showGraph ? '📊 隐藏图谱' : '📊 展开图谱' }}
      </button>
    </div>

    <div class="references-content">
      <!-- 引用了（本文引用的其他文章） -->
      <div class="reference-section" v-if="outgoingLinks.length > 0">
        <div class="section-label">
          <span class="label-icon">📤</span>
          <span>引用了</span>
        </div>
        <div class="link-list">
          <a
            v-for="link in outgoingLinks"
            :key="link.id"
            :href="link.href"
            class="link-item"
          >
            <span class="link-icon">📄</span>
            <span class="link-text">{{ link.title }}</span>
          </a>
        </div>
      </div>

      <!-- 被引用（引用了本文的其他文章） -->
      <div class="reference-section" v-if="incomingLinks.length > 0">
        <div class="section-label">
          <span class="label-icon">📥</span>
          <span>被引用</span>
        </div>
        <div class="link-list">
          <a
            v-for="link in incomingLinks"
            :key="link.id"
            :href="link.href"
            class="link-item"
          >
            <span class="link-icon">📄</span>
            <span class="link-text">{{ link.title }}</span>
          </a>
        </div>
      </div>

      <!-- 无引用关系提示 -->
      <div class="no-references" v-if="outgoingLinks.length === 0 && incomingLinks.length === 0">
        <span class="empty-icon">📭</span>
        <span>暂无关联笔记</span>
      </div>
    </div>

    <!-- 知识图谱（可展开） -->
    <Transition name="fade">
      <div v-if="showGraph" class="graph-container">
        <div ref="graphContainer" class="graph-canvas"></div>
      </div>
    </Transition>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch, nextTick } from "vue";
import { useData, useRouter } from "vitepress";

interface LinkItem {
  id: string;
  title: string;
  href: string;
}

const { page, frontmatter } = useData();
const router = useRouter();

const showGraph = ref(false);
const graphContainer = ref<HTMLElement | null>(null);
const incomingLinks = ref<LinkItem[]>([]);

let cy: any = null;

// 从frontmatter获取outgoing links（本文引用的）
const outgoingLinks = computed<LinkItem[]>(() => {
  const wikiLinks = (frontmatter.value?.wikiLinks as string[]) || [];
  return wikiLinks.map((link) => ({
    id: link.toLowerCase().replace(/\s+/g, "-"),
    title: link,
    href: `/posts/${link.toLowerCase().replace(/\s+/g, "-")}.html`,
  }));
});

// 计算所有链接
const allLinks = computed(() => [...outgoingLinks.value, ...incomingLinks.value]);

// 是否有引用关系
const hasReferences = computed(() => {
  return frontmatter.value?.graph !== false;
});

// 初始化图谱
async function initGraph() {
  if (!graphContainer.value || allLinks.value.length === 0) return;

  try {
    const cytoscape = (await import("cytoscape")).default;
    const dagre = (await import("cytoscape-dagre")).default;

    cytoscape.use(dagre);

    const currentPath = page.value.relativePath.replace(/\.md$/, "");
    const currentTitle = page.value.title || currentPath;

    const nodes = [
      {
        data: { id: currentPath, label: currentTitle },
        classes: "current",
      },
      ...outgoingLinks.value.map((link) => ({
        data: { id: link.id, label: link.title },
        classes: "linked",
      })),
    ];

    const edges = outgoingLinks.value.map((link) => ({
      data: { source: currentPath, target: link.id },
    }));

    cy = cytoscape({
      container: graphContainer.value,
      elements: [...nodes, ...edges],
      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            "text-valign": "center",
            "text-halign": "center",
            "font-size": "10px",
            width: "50px",
            height: "50px",
            "background-color": "#64748b",
            color: "#fff",
            "text-wrap": "wrap",
            "text-max-width": "45px",
          },
        },
        {
          selector: "node.current",
          style: {
            "background-color": "#0ea5e9",
            "border-width": "3px",
            "border-color": "#0284c7",
          },
        },
        {
          selector: "node.linked",
          style: {
            "background-color": "#d946ef",
            cursor: "pointer",
          },
        },
        {
          selector: "edge",
          style: {
            width: 2,
            "line-color": "#94a3b8",
            "target-arrow-color": "#94a3b8",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
          },
        },
      ],
      layout: {
        name: "dagre",
        rankDir: "LR",
        nodeSep: 40,
        rankSep: 60,
      },
      userZoomingEnabled: true,
      userPanningEnabled: true,
      minZoom: 0.5,
      maxZoom: 2,
    });

    // 点击节点跳转
    cy.on("tap", "node.linked", (e: any) => {
      const nodeId = e.target.data("id");
      router.go(`/posts/${nodeId}.html`);
    });
  } catch (error) {
    console.error("[RelatedReferences] Failed to initialize graph:", error);
  }
}

function destroyGraph() {
  if (cy) {
    cy.destroy();
    cy = null;
  }
}

// 监听图谱显示状态
watch(showGraph, async (show) => {
  if (show) {
    await nextTick();
    initGraph();
  } else {
    destroyGraph();
  }
});

// 页面变化时重置
watch(
  () => page.value.relativePath,
  () => {
    showGraph.value = false;
    destroyGraph();
  }
);
</script>

<style scoped>
.related-references {
  margin-top: 3rem;
  padding: 1.5rem;
  background: var(--vp-c-bg-soft);
  border-radius: 0.75rem;
  border: 1px solid var(--vp-c-divider);
}

.references-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 1rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid var(--vp-c-divider);
}

.header-icon {
  font-size: 1.25rem;
}

.header-title {
  font-size: 1rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
  flex: 1;
}

.graph-toggle {
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  padding: 0.375rem 0.75rem;
  border-radius: 1rem;
  font-size: 0.75rem;
  cursor: pointer;
  transition: all 0.2s;
  color: var(--vp-c-text-2);
}

.graph-toggle:hover {
  background: var(--vp-c-bg-alt);
  color: var(--vp-c-text-1);
}

.references-content {
  display: flex;
  flex-wrap: wrap;
  gap: 1.5rem;
}

.reference-section {
  flex: 1;
  min-width: 200px;
}

.section-label {
  display: flex;
  align-items: center;
  gap: 0.375rem;
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--vp-c-text-2);
  margin-bottom: 0.75rem;
}

.label-icon {
  font-size: 0.875rem;
}

.link-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.link-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.75rem;
  background: var(--vp-c-bg);
  border-radius: 0.5rem;
  text-decoration: none;
  color: var(--vp-c-text-1);
  font-size: 0.875rem;
  transition: all 0.2s;
  border: 1px solid transparent;
}

.link-item:hover {
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-bg-alt);
}

.link-icon {
  font-size: 0.875rem;
}

.link-text {
  flex: 1;
}

.no-references {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: var(--vp-c-text-3);
  font-size: 0.875rem;
  padding: 0.5rem;
}

.empty-icon {
  font-size: 1rem;
}

.graph-container {
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid var(--vp-c-divider);
}

.graph-canvas {
  width: 100%;
  height: 250px;
  background: var(--vp-c-bg);
  border-radius: 0.5rem;
}

/* 过渡动画 */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease, transform 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}
</style>
