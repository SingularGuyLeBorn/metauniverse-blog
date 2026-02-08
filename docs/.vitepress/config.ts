import { defineConfig } from 'vitepress'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { generateFullSidebar } from './utils/sidebar'

// 获取 docs 目录路径
const __dirname = path.dirname(fileURLToPath(import.meta.url))
const docsDir = path.resolve(__dirname, '..')

// 自动生成 sidebar
const autoSidebar = generateFullSidebar(docsDir)


export default defineConfig({
  lang: 'zh-CN',
  title: 'MetaUniverse',
  titleTemplate: ':title | MetaUniverse',
  description: '大模型技术博客 - 探索AI的无限可能',
  
  lastUpdated: true,
  cleanUrls: true,
  ignoreDeadLinks: true,
  
    markdown: {
      theme: {
        light: 'github-light',
        dark: 'github-dark'
      },
      languages: [
        JSON.parse(fs.readFileSync(path.resolve(__dirname, 'ptx.json'), 'utf-8'))
      ],
      lineNumbers: true, // 显示行号
      math: true, // 启用数学公式支持
      image: {
        lazyLoading: true // 图片懒加载
      },
      // 外部链接图标
      externalLinks: {
        target: '_blank',
        rel: 'noopener noreferrer'
      },
      // 目录配置
      toc: {
        level: [2, 3, 4]
      },
      container: {
        tipLabel: '💡 提示',
        warningLabel: '⚠️ 警告',
        dangerLabel: '🚨 危险',
        infoLabel: 'ℹ️ 信息',
        detailsLabel: '▶️ 详细信息'
      },
      config: (md) => {
        // 自定义 WikiLink 插件 [[Link]] -> <a href="/posts/link">Link</a>
        md.core.ruler.push('wiki_link', (state) => {
          state.tokens.forEach(token => {
            if (token.type === 'inline' && token.children) {
              for (let i = 0; i < token.children.length; i++) {
                const child = token.children[i];
                if (child.type === 'text' && child.content) {
                  // 排除 [[TOC]]
                  const regex = /\[\[(?!TOC\]\])([^\]|]+)(?:\|[^\]]+)?\]\]/g;
                  let match;
                  // 简单的文本替换逻辑
                }
              }
            }
          })
        })
        
        // 使用简单的正则替换插件
        const defaultRender = md.renderer.rules.text || function(tokens, idx, options, env, self) {
          return self.renderToken(tokens, idx, options);
        };
        
        md.renderer.rules.text = function(tokens, idx, options, env, self) {
          const content = tokens[idx].content;
          // 排除 [[TOC]]
          if (content.match(/\[\[(?!TOC\]\])([^\]|]+)(?:\|([^\]]+))?\]\]/)) {
            return content.replace(/\[\[(?!TOC\]\])([^\]|]+)(?:\|([^\]]+))?\]\]/g, (match, p1, p2) => {
              const link = p1.trim().toLowerCase().replace(/\s+/g, '-');
              const text = p2 ? p2.trim() : p1.trim();
              return `<a href="/posts/${link}.html" class="wiki-link">${text}</a>`;
            });
          }
          return defaultRender(tokens, idx, options, env, self);
        };

        // Mermaid 代码块拦截
        const defaultFence = md.renderer.rules.fence || function(tokens, idx, options, env, self) {
          return self.renderToken(tokens, idx, options);
        };
        md.renderer.rules.fence = (tokens, idx, options, env, self) => {
          const token = tokens[idx]
          if (token.info.trim() === 'mermaid') {
            return `<Mermaid code="${encodeURIComponent(token.content)}" />`
          }
          return defaultFence(tokens, idx, options, env, self)
        }
      }
    },
    
    transformPageData(pageData) {
      // 提取 WikiLinks 到 frontmatter 供图谱使用
      const pd = pageData as any
      let content = pd.content

      // 如果 runtime 中 pageData 没有 content，尝试从文件读取
      if (!content && pd.filePath) {
        try {
          content = fs.readFileSync(pd.filePath, 'utf-8')
        } catch (e) {
          // ignore error
        }
      }
      content = content || '';
      const regex = /\[\[([^\]|]+)(?:\|[^\]]+)?\]\]/g;
      const links = new Set<string>();
      let match;
      while ((match = regex.exec(content)) !== null) {
        links.add(match[1].trim());
      }
      
      pageData.frontmatter.wikiLinks = Array.from(links);
      pageData.frontmatter.graph = true; // 默认开启图谱

      // 计算字数和阅读时间
      const pureContent = content.replace(/<[^>]*>/g, '').replace(/\[\[.*?\]\]/g, '');
      const cnMatches = pureContent.match(/[\u4e00-\u9fa5]/g);
      const enMatches = pureContent.match(/[a-zA-Z0-9]+/g);
      const cnCount = cnMatches ? cnMatches.length : 0;
      const enCount = enMatches ? enMatches.length : 0;
      const wordCount = cnCount + enCount;
      const readingTime = Math.ceil(wordCount / 400); // 假设阅读速度 400字/分钟

      pageData.frontmatter.wordCount = wordCount;
      pageData.frontmatter.readingTime = readingTime;
    },
  
  head: [
    ['meta', { name: 'theme-color', content: '#0ea5e9' }],
    ['meta', { name: 'mobile-web-app-capable', content: 'yes' }],
    ['meta', { name: 'apple-mobile-web-app-capable', content: 'yes' }], // 保持兼容性
    ['link', { rel: 'icon', href: '/logo.svg' }],
    ['link', { rel: 'preconnect', href: 'https://cdn.jsdelivr.net' }],
    // 初始化脚本 - 避免模式切换闪烁
    ['script', {}, `
      (function() {
        try {
          const mode = localStorage.getItem('mu-layout') || 'default';
          document.documentElement.setAttribute('data-layout', mode);
          const theme = localStorage.getItem('mu-theme');
          if (theme === 'dark' || (!theme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
            document.documentElement.classList.add('dark');
          }
        } catch (e) {}
      })();
    `]
  ],
  
  themeConfig: {
    logo: '/logo.svg',
    
    nav: [
      { text: '首页', link: '/' },
      { text: '技术文章', link: '/posts/' },
      { text: '论文阅读', link: '/papers/' },
      { text: '知识库', link: '/knowledge/' },
      { text: '杂谈', link: '/essays/' },
      { text: '随想', link: '/thoughts/' },
      { text: '年度总结', link: '/yearly/' },
      { text: '关于我', link: '/about/' }
    ],
    
    sidebar: {
      ...autoSidebar,
      // about 页面保持静态配置
      '/about/': [
        {
          text: '关于我',
          items: [
            { text: '个人档案', link: '/about/' }
          ]
        }
      ]
    },
    
    socialLinks: [
      { icon: 'github', link: 'https://github.com/yourusername' }
    ],
    
    editLink: {
      pattern: 'https://github.com/yourusername/metauniverse-blog/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页'
    },
    
    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索文档',
            buttonAriaLabel: '搜索文档'
          },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: {
              selectText: '选择',
              navigateText: '切换',
              closeText: '关闭'
            }
          }
        }
      }
    },
    
    footer: {
      message: '基于 VitePress 构建 | MetaUniverse 八大特性系统',
      copyright: 'Copyright © 2024 MetaUniverse'
    },
    
    outline: {
      label: '页面导航',
      level: [2, 3]
    },
    
    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'short'
      }
    },
    
    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    },
    
    returnToTopLabel: '返回顶部',
    sidebarMenuLabel: '菜单',
    darkModeSwitchLabel: '外观'
  },
  
  vite: {
    resolve: {
      alias: {
        '@components': './.vitepress/theme/components',
        '@composables': './.vitepress/theme/composables',
        '@stores': './.vitepress/theme/stores',
        '@utils': './.vitepress/theme/utils'
      }
    },
    
    build: {
      chunkSizeWarningLimit: 1000
    },
    
    ssr: {
      noExternal: ['flexsearch']
    },
    
    optimizeDeps: {
      include: [
        'vue',
        '@vueuse/core',
        'pinia',
        'flexsearch',
        'fuse.js',
        'lz-string',
        'mitt'
      ]
    }
  }
})
