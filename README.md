# 🌌 MetaUniverse Blog

<p align="center">
  <img src="docs/public/logo.svg" width="120" height="120" alt="MetaUniverse Logo">
</p>

<p align="center">
  <strong>探索 AI 的无限可能 | 构建你的第二大脑</strong>
</p>

<p align="center">
  <a href="https://github.com/SingularGuyLeBorn/metauniverse-blog/actions"><img src="https://img.shields.io/github/actions/workflow/status/SingularGuyLeBorn/metauniverse-blog/deploy.yml?label=Build&style=flat-square" alt="Build Status"></a>
  <img src="https://img.shields.io/badge/VitePress-1.0.0-646cff?style=flat-square&logo=vite&logoColor=white" alt="VitePress">
  <img src="https://img.shields.io/badge/Vue.js-3.4-4FC08D?style=flat-square&logo=vue.js&logoColor=white" alt="Vue">
  <img src="https://img.shields.io/badge/TypeScript-5.4-3178C6?style=flat-square&logo=typescript&logoColor=white" alt="TypeScript">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License">
</p>

---

## 📖 简介 (Introduction)

**MetaUniverse** 是一个基于 [VitePress](https://vitepress.dev/) 构建的现代化 AI 技术博客。它不仅是一个静态文档站点，更是一个集成了多种动态交互特性的知识管理系统。我们致力于记录 LLM（Large Language Model）从 0 到 1 的全流程，分享前沿论文笔记，并沉淀系统化的技术知识。

## ✨ 核心特性 (Features)

本项目集成了八大核心特性，打造极致的阅读与交互体验：

1.  **🚀 Deep Link Persistence (深链接持久化)**
    *   所有用户配置（主题、布局模式）均自动持久化，刷新页面不丢失。
2.  **🌗 Dual Layout Mode (双重布局模式)**
    *   支持 **默认布局** 和 **专注模式 (Focus Mode)**，一键隐藏侧边栏，沉浸式阅读。悬浮式 Mode Switcher 组件支持拖拽与最小化。
3.  **🔍 RAG Search (RAG 搜索)**
    *   集成本地 RAG 搜索能力，支持根据内容相似度进行语义检索（模拟）。
4.  **📝 Paragraph-level Git Annotations (段落级 Git 注释)**
    *   支持 Alt + Click 选中文本段落，添加类似 Git Blame 的批注（模拟）。
5.  **🔥 Semantic Heatmap (语义热力图)**
    *   页面左侧实时显示阅读热力条，根据在该段落的停留时间呈现蓝-红渐变，直观展示阅读重点与进度。
6.  **🕸️ Bidirectional Knowledge Graph (双向知识图谱)**
    *   基于 WikiLinks (`[[Link]]`) 自动构建文档间的引用关系。文章底部自动展示 "Mentioned in" 和 "References" 关系图。
7.  **📊 Interactive Tensor Visualization (交互式张量可视化)**
    *   内置 Tensor Playground，可视化展示 Transformer 中的矩阵运算与 Attention 机制。
8.  **⚡ Real-time WASM Sandbox (实时 WASM 沙箱)**
    *   集成 Pyodide，支持在浏览器端直接运行 Python 代码，无需后端服务。

## 🎨 视觉与美化 (Visuals)

-   **Home Hero**: 首页集成动态打字机效果、3D 悬浮卡片导航及流体背景动画。
-   **Spotlight Effect**: 卡片组件支持 Apple 风格的鼠标跟随聚光灯效果。
-   **Scroll Progress**: 全局顶部彩虹色阅读进度条。
-   **Mermaid Integration**: 手动集成的 Mermaid.js 支持，完美渲染 Gantt、Pie 等复杂图表。
-   **Confetti**: 关于页头像彩蛋交互。

## 🛠️ 技术栈 (Tech Stack)

-   **Core**: [VitePress](https://vitepress.dev/) + [Vue 3](https://vuejs.org/)
-   **Language**: TypeScript
-   **Styling**: CSS Variables (Dark/Light mode support) + Glassmorphism
-   **Visualization**: [Mermaid.js](https://mermaid.js.org/), [Cytoscape.js](https://js.cytoscape.org/) (Graph), [Canvas-Confetti](https://github.com/catdad/canvas-confetti)
-   **Runtime**: Node.js >= 18

## 🚀 快速开始 (Getting Started)

### 本地运行

```bash
# 1. 克隆仓库
git clone https://github.com/SingularGuyLeBorn/metauniverse-blog.git
cd metauniverse-blog

# 2. 安装依赖 (推荐使用 npm)
npm install

# 3. 启动开发服务器
npm run dev
# 访问 http://localhost:5173
```

### 构建与部署

```bash
# 构建生产版本
npm run build

# 本地预览生产构建
npm run preview
```

## 📂 目录结构

```text
metauniverse-blog/
├── docs/
│   ├── .vitepress/           # 核心配置与主题
│   │   ├── config.ts         # VitePress 配置
│   │   └── theme/            # 自定义主题组件
│   ├── index.md              # 首页 (HomeHero)
│   ├── about/                # 关于页 (AboutProfile)
│   ├── papers/               # 论文阅读板块
│   ├── knowledge/            # 知识库板块
│   ├── essays/               # 杂谈板块
│   ├── thoughts/             # 随想板块
│   └── yearly/               # 年度总结
└── package.json
```

## 🤝 贡献 (Contributing)

欢迎提交 Issue 或 Pull Request！

## 📄 License

[MIT](./LICENSE) © 2024 MetaUniverse
