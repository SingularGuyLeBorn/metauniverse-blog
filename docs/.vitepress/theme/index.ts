import type { Theme } from "vitepress";
import { h, watch } from "vue";
import { createPinia } from "pinia";
import DefaultTheme from "vitepress/theme";
import { useData } from "vitepress";

// 样式
import "./style.css";
import "./styles/reading-mode.css";
import "./styles/zen-mode.css";

// 状态管理
import { useAppStore } from "./stores/app";
import { useLayoutStore } from "./stores/layout";

// 组件
import ModeSwitcher from "./components/features/ModeSwitcher.vue";
import ThemeToggle from "./components/widgets/ThemeToggle.vue";
import KnowledgeGraph from "./components/features/KnowledgeGraph.vue";
import SemanticHeatmap from "./components/features/SemanticHeatmap.vue";
import TensorPlayground from "./components/features/TensorPlayground.vue";
import SegmentAnnotation from "./components/features/SegmentAnnotation.vue";
import CodeSandbox from "./components/features/CodeSandbox.vue";
import RAGSearch from "./components/features/RAGSearch.vue";
import RelatedReferences from "./components/features/RelatedReferences.vue";
import Mermaid from "./components/Mermaid.vue";
import HomeHero from "./components/HomeHero.vue";
import AboutProfile from "./components/AboutProfile.vue";
import ScrollProgress from "./components/features/ScrollProgress.vue";
import NotFound from "./components/NotFound.vue";
import ArticleIndex from "./components/ArticleIndex.vue";
import KnowledgeDashboard from "./components/KnowledgeDashboard.vue";
import SectionDashboard from "./components/SectionDashboard.vue";
import PostsDashboard from "./components/PostsDashboard.vue";
import PapersDashboard from "./components/PapersDashboard.vue";
import EssaysDashboard from "./components/EssaysDashboard.vue";
import ThoughtsDashboard from "./components/ThoughtsDashboard.vue";
import YearlyDashboard from "./components/YearlyDashboard.vue";
import LayoutToolbar from "./components/features/LayoutToolbar.vue";
import SidebarToolbar from "./components/features/SidebarToolbar.vue";
import GlobalLayoutControl from "./components/features/GlobalLayoutControl.vue";
import ArticleMetadata from "./components/features/ArticleMetadata.vue";
import Breadcrumbs from "./components/features/Breadcrumbs.vue";
import ImageLightbox from "./components/features/ImageLightbox.vue";
import ImageCaptions from "./components/features/ImageCaptions.vue";
import ZenModeToggle from "./components/features/ZenModeToggle.vue";
import LinkPopover from "./components/features/LinkPopover.vue";
import LiveEditor from "./components/features/LiveEditor.vue";
import HistoryViewer from "./components/features/HistoryViewer.vue";

// PPO Visualization Components
import PPOHeader from "./components/features/RL/ppo/components/PPOHeader.vue";
import PPODependencyGraph from "./components/features/RL/ppo/components/PPODependencyGraph.vue"; // Re-import
import PPOInfo from "./components/features/RL/ppo/components/PPOInfo.vue";
import PPOCodeTrace from "./components/features/RL/ppo/components/PPOCodeTrace.vue";


export default {
  extends: DefaultTheme,
  NotFound, // Override default 404

  Layout: () => {
    const { frontmatter } = useData();

    return h(DefaultTheme.Layout, null, {
      // 侧边栏顶部工具栏 - Injected here
      "sidebar-nav-before": () => h(SidebarToolbar),

      // 顶部阅读进度条
      "layout-top": () => h(ScrollProgress),
      
      // 导航栏右侧 - 布局工具栏
      "nav-bar-content-after": () => h(LayoutToolbar),

      // 文档内容前 - 面包屑和元数据
      "doc-before": () => {
        const { frontmatter } = useData();
        if (frontmatter.value.layout !== 'home') {
          return [
            h(Breadcrumbs)
          ];
        }
        return null;
      },
      
      // 布局底部插槽 - 模式切换器、热力图、全局布局控制器、灯箱
      "layout-bottom": () =>
        h("div", { id: "mu-teleport-container" }, [
          h(ModeSwitcher),
          h(SemanticHeatmap),
          h(GlobalLayoutControl), // 全局布局控制器
          h(ImageLightbox), // 图片灯箱
          h(ImageCaptions), // 图片标题自动生成
          h(ZenModeToggle), // Zen Mode Exit Button
          h(LinkPopover), // Link Preview Card
          h(LiveEditor), // Live Markdown Editor
          h(HistoryViewer) // Version History Viewer
        ]),

      // 文档内容后 - 关联笔记引用
      "doc-after": () => {
        if (frontmatter.value.graph !== false) {
          return h(RelatedReferences);
        }
        return null;
      },
    });
  },

  enhanceApp({ app, router, siteData }) {
    // 创建Pinia实例
    const pinia = createPinia();
    app.use(pinia);

    // 注册全局组件
    app.component("ThemeToggle", ThemeToggle);
    app.component("KnowledgeGraph", KnowledgeGraph);
    app.component("TensorPlayground", TensorPlayground);
    app.component("SegmentAnnotation", SegmentAnnotation);
    app.component("CodeSandbox", CodeSandbox);
    app.component("RAGSearch", RAGSearch);
    app.component("Mermaid", Mermaid);
    app.component("HomeHero", HomeHero);
    app.component("AboutProfile", AboutProfile);
    app.component("ArticleIndex", ArticleIndex);
    app.component("KnowledgeDashboard", KnowledgeDashboard);
    app.component("SectionDashboard", SectionDashboard);
    app.component("PostsDashboard", PostsDashboard);
    app.component("PapersDashboard", PapersDashboard);
    app.component("EssaysDashboard", EssaysDashboard);
    app.component("ThoughtsDashboard", ThoughtsDashboard);
    app.component("YearlyDashboard", YearlyDashboard);
    app.component("ArticleMetadata", ArticleMetadata);
    app.component("Breadcrumbs", Breadcrumbs);
    app.component("ImageLightbox", ImageLightbox);
    app.component("ImageCaptions", ImageCaptions);

    // Register PPO Visualization Components
    app.component("PPOHeader", PPOHeader);
    app.component("PPODependencyGraph", PPODependencyGraph);
    app.component("PPOInfo", PPOInfo);
    app.component("PPOCodeTrace", PPOCodeTrace);
    
    // 客户端初始化
    if (typeof window !== "undefined") {
      // 初始化应用状态
      const appStore = useAppStore();
      const layoutStore = useLayoutStore(); // 初始化布局 store

      // 路由导航钩子
      router.onBeforeRouteChange = (to) => {
        appStore.setLoading(true);
      };

      router.onAfterRouteChanged = (to) => {
        appStore.setLoading(false);
      };
    }
  },

  setup() {
    // 客户端初始化逻辑
    if (typeof window !== "undefined") {
      const appStore = useAppStore();
      const layoutStore = useLayoutStore();

      // 应用主题到DOM
      watch(
        () => appStore.isDark,
        (isDark) => {
          document.documentElement.classList.toggle("dark", isDark);
        },
        { immediate: true },
      );

      // 应用布局模式到DOM
      watch(
        () => appStore.layoutMode,
        (mode) => {
          document.documentElement.setAttribute("data-layout", mode);
        },
        { immediate: true },
      );

      // 监听 Zen Mode
      watch(
        () => layoutStore.zenMode,
        (isZen) => {
          document.documentElement.classList.toggle("zen-mode", isZen);
        },
        { immediate: true }
      );
    }
  },
} as Theme;
