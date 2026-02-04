import { defineConfig } from 'vitepress'

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
    lineNumbers: true,
    math: true
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
            { text: 'Transformer 架构', link: '/posts/transformer' }
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
      chunkSizeWarningLimit: 1000,
      rollupOptions: {
        output: {
          manualChunks: {
            'vue-vendor': ['vue'],
            'vueuse': ['@vueuse/core'],
            'pinia': ['pinia'],
            'search': ['flexsearch', 'fuse.js'],
            'graph': ['cytoscape', 'cytoscape-dagre']
          }
        }
      }
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
