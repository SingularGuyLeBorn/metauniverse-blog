<template>
  <div class="knowledge-graph" v-if="nodes.length > 0">
    <div class="graph-header">
      <span class="graph-icon">🕸️</span>
      <span class="graph-title">相关文章</span>
    </div>
    <div ref="graphContainer" class="graph-container"></div>
    <div class="graph-legend">
      <div class="legend-item">
        <span class="legend-dot current"></span>
        <span>当前页</span>
      </div>
      <div class="legend-item">
        <span class="legend-dot linked"></span>
        <span>关联页</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from "vue";
import { useData, useRouter } from "vitepress";

interface GraphNode {
  id: string;
  label: string;
  isCurrent: boolean;
}

interface GraphEdge {
  source: string;
  target: string;
}

const { page } = useData();
const router = useRouter();
const graphContainer = ref<HTMLElement | null>(null);
const nodes = ref<GraphNode[]>([]);
const edges = ref<GraphEdge[]>([]);

let cy: any = null;

// 提取Wiki链接 [[PageName]] 或 [[PageName|Display Text]]
function extractWikiLinks(content: string): string[] {
  const regex = /\[\[([^\]|]+)(?:\|[^\]]+)?\]\]/g;
  const links: string[] = [];
  let match;
  while ((match = regex.exec(content)) !== null) {
    links.push(match[1].trim());
  }
  return links;
}

// 构建图谱数据
function buildGraphData() {
  const currentPath = page.value.relativePath.replace(/\.md$/, "");
  const currentTitle = page.value.title || currentPath;

  // 当前节点
  const currentNode: GraphNode = {
    id: currentPath,
    label: currentTitle,
    isCurrent: true,
  };

  // 从frontmatter获取关联页面
  const wikiLinks = (page.value.frontmatter?.wikiLinks as string[]) || [];

  const linkedNodes: GraphNode[] = wikiLinks.map((link) => ({
    id: link.toLowerCase().replace(/\s+/g, "-"),
    label: link,
    isCurrent: false,
  }));

  const graphEdges: GraphEdge[] = linkedNodes.map((node) => ({
    source: currentPath,
    target: node.id,
  }));

  nodes.value = [currentNode, ...linkedNodes];
  edges.value = graphEdges;
}

async function initGraph() {
  if (!graphContainer.value || nodes.value.length === 0) return;

  try {
    const cytoscape = (await import("cytoscape")).default;
    const dagre = (await import("cytoscape-dagre")).default;

    cytoscape.use(dagre);

    cy = cytoscape({
      container: graphContainer.value,
      elements: [
        ...nodes.value.map((node) => ({
          data: { id: node.id, label: node.label },
          classes: node.isCurrent ? "current" : "linked",
        })),
        ...edges.value.map((edge) => ({
          data: { source: edge.source, target: edge.target },
        })),
      ],
      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            "text-valign": "center",
            "text-halign": "center",
            "font-size": "10px",
            width: "60px",
            height: "60px",
            "background-color": "#64748b",
            color: "#fff",
            "text-wrap": "wrap",
            "text-max-width": "50px",
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
            "curve-style": "bezier",
          },
        },
      ],
      layout: {
        name: "dagre",
        rankDir: "TB",
        nodeSep: 30,
        rankSep: 50,
      },
      userZoomingEnabled: false,
      userPanningEnabled: false,
    });

    // 点击节点跳转
    cy.on("tap", "node.linked", (e: any) => {
      const nodeId = e.target.data("id");
      router.go(`/${nodeId}`);
    });
  } catch (error) {
    console.error("[KnowledgeGraph] Failed to initialize:", error);
  }
}

function destroyGraph() {
  if (cy) {
    cy.destroy();
    cy = null;
  }
}

onMounted(() => {
  buildGraphData();
  if (nodes.value.length > 1) {
    initGraph();
  }
});

onUnmounted(() => {
  destroyGraph();
});

watch(
  () => page.value.relativePath,
  () => {
    destroyGraph();
    buildGraphData();
    if (nodes.value.length > 1) {
      initGraph();
    }
  },
);
</script>

<style scoped>
.knowledge-graph {
  margin-top: 1.5rem;
  padding: 1rem;
  background: var(--vp-c-bg-soft);
  border-radius: 0.75rem;
  border: 1px solid var(--vp-c-divider);
}

.graph-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
}

.graph-icon {
  font-size: 1.125rem;
}

.graph-title {
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.graph-container {
  width: 100%;
  height: 200px;
  background: var(--vp-c-bg);
  border-radius: 0.5rem;
}

.graph-legend {
  display: flex;
  gap: 1rem;
  margin-top: 0.75rem;
  font-size: 0.75rem;
  color: var(--vp-c-text-2);
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 0.375rem;
}

.legend-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}

.legend-dot.current {
  background: #0ea5e9;
}

.legend-dot.linked {
  background: #d946ef;
}
</style>
