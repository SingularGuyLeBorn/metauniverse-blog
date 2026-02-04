import { defineConfig } from 'vitepress'
import fs from 'node:fs'

export default defineConfig({
  lang: 'zh-CN',
  title: 'MetaUniverse',
  titleTemplate: ':title | MetaUniverse',
  description: '大模型技术博客 - 探索AI的无限可能',
  
  lastUpdated: true,
  cleanUrls: true,
  
    markdown: {
      theme: {
        light: 'github-light',
        dark: 'github-dark'
      },
      lineNumbers: true, // 显示行号
      math: true, // 启用数学公式支持
      image: {
        lazyLoading: true // 图片懒加载
      },
      container: {
        tipLabel: '提示',
        warningLabel: '警告',
        dangerLabel: '危险',
        infoLabel: '信息',
        detailsLabel: '详细信息'
      },
      config: (md) => {
        // 自定义 WikiLink 插件 [[Link]] -> <a href="/posts/link">Link</a>
        md.core.ruler.push('wiki_link', (state) => {
          state.tokens.forEach(token => {
            if (token.type === 'inline' && token.children) {
              for (let i = 0; i < token.children.length; i++) {
                const child = token.children[i];
                if (child.type === 'text' && child.content) {
                  const regex = /\[\[([^\]|]+)(?:\|[^\]]+)?\]\]/g;
                  let match;
                  // 简单的文本替换逻辑 (注: 生产环境建议编写完整的 Tokenizer)
                  // 这里为了简化演示，我们假设 [[ ]] 不会跨 Token 分割
                  // 实际上 markdown-it 的 text token 可能会被 formatter 分割，但简单场景够用
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
          if (content.match(/\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/)) {
            return content.replace(/\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g, (match, p1, p2) => {
              const link = p1.trim().toLowerCase().replace(/\s+/g, '-');
              const text = p2 ? p2.trim() : p1.trim();
              return `<a href="/posts/${link}.html" class="wiki-link">${text}</a>`;
            });
          }
          return defaultRender(tokens, idx, options, env, self);
        };
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
    },
  
  head: [
    ['meta', { name: 'theme-color', content: '#0ea5e9' }],
    ['meta', { name: 'apple-mobile-web-app-capable', content: 'yes' }],
    ['link', { rel: 'icon', href: '/favicon.ico' }],
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
      { text: '文章', link: '/posts/' },
      { text: '笔记', link: '/notes/' },
      { text: '关于', link: '/about' }
    ],
    
    sidebar: {
      '/posts/': [
        {
          text: '文章',
          items: [
            { text: '全部文章', link: '/posts/' },
            { text: 'Hello World', link: '/posts/hello-world' },
            { text: 'Transformer 架构', link: '/posts/transformer' },
            { text: 'Markdown 演示', link: '/posts/markdown-demo' }
          ]
        }
      ],
      '/notes/': [
        {
          text: '原子笔记',
          items: [
            { text: '全部笔记', link: '/notes/' }
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
      noExternal: ['@vueuse/core', 'pinia', 'flexsearch']
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
