import { defineConfig } from 'vitepress'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { generateFullSidebar } from './utils/sidebar'
import katex from '@iktakahiro/markdown-it-katex'

// 获取 docs 目录路径
const __dirname = path.dirname(fileURLToPath(import.meta.url))
const docsDir = path.resolve(__dirname, '..')

// 自动生成 sidebar
// 自动生成 sidebar
const autoSidebar = generateFullSidebar(docsDir)

// 自定义 Vite 插件：虚拟 Markdown 转换
// virtualMarkdownPlugin removed (replaced by dynamic routes)


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
      math: false, // 禁用自带的 MathJax3，改用 KaTeX 插件
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

        // 启用 KaTeX
        md.use(katex, { strict: false })
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
    // KaTeX for PPOInfo.vue
    ['link', { rel: 'stylesheet', href: 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css' }],
    ['script', { src: 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js' }],
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
      noExternal: ['flexsearch', '@iktakahiro/markdown-it-katex']
    },
    
    optimizeDeps: {
      include: [
        'vue',
        '@vueuse/core',
        'pinia',
        'flexsearch',
        'fuse.js',
        'lz-string',
        'mitt',
        'markdown-it',
        'katex',
        '@iktakahiro/markdown-it-katex'
      ]
    },

    // 自定义插件：处理批注数据的本地保存
    plugins: [
      {
        name: 'markdown-editor-api',
        configureServer(server) {
          server.middlewares.use((req, res, next) => {
            // 安全限制：仅允许 docs/ 及其子目录
            const isPathSafe = (p: string) => {
               const absolute = path.resolve(__dirname, '..', p.replace(/^\//, ''))
               const docsRoot = path.resolve(__dirname, '..', 'docs')
               return absolute.startsWith(docsRoot)
            }

            // [移除] 虚拟 Markdown 转换拦截 (影子文件访问) - 已由 virtual-markdown-plugin 接管

            // 读取源码原文
            if (req.url?.startsWith('/api/read-md?path=') && req.method === 'GET') {
              const url = new URL(req.url, `http://${req.headers.host}`)
              const filePath = url.searchParams.get('path')
              if (!filePath) {
                res.statusCode = 400
                return res.end('Path missing')
              }
              try {
                // 1. 去除 view/ 前缀 (针对虚拟路由)
                let targetPath = filePath.replace(/^view\//, '').replace(/^\//, '')
                let fullPath = path.resolve(__dirname, '..', targetPath)

                // 2. 智能文件查找策略
                if (!fs.existsSync(fullPath)) {
                  // 尝试 A: 去除 .md 后缀
                  if (targetPath.endsWith('.md')) {
                    const noMd = targetPath.replace(/\.md$/, '')
                    const noMdPath = path.resolve(__dirname, '..', noMd)
                    if (fs.existsSync(noMdPath)) {
                       fullPath = noMdPath
                       targetPath = noMd
                    } else {
                       // 尝试 B: 去除 .md 后再尝试添加源码后缀 (.py, .ipynb, .java, etc.)
                       // 针对 LiveEditor 请求了 .../foo.md 但实际文件是 .../foo.py 的情况
                       const extensions = ['.py', '.ipynb', '.java', '.cpp', '.ts', '.js', '.go', '.rs', '.c', '.cs']
                       for (const ext of extensions) {
                         const withExt = noMdPath + ext
                         if (fs.existsSync(withExt)) {
                           fullPath = withExt
                           targetPath = noMd + ext
                           break
                         }
                       }
                    }
                  } 
                  // 尝试 C: 本身没有后缀，尝试添加后缀
                  else {
                     const extensions = ['.py', '.ipynb', '.md', '.java', '.cpp', '.ts', '.js', '.go', '.rs', '.c', '.cs']
                       for (const ext of extensions) {
                         const withExt = fullPath + ext
                         if (fs.existsSync(withExt)) {
                           fullPath = withExt
                           targetPath = targetPath + ext
                           break
                         }
                       }
                  }
                }

                if (!fs.existsSync(fullPath)) {
                   res.statusCode = 404
                   return res.end(JSON.stringify({ error: 'File not found', path: targetPath }))
                }

                const content = fs.readFileSync(fullPath, 'utf-8')
                res.setHeader('Content-Type', 'application/json')
                // isVirtual: 前端路径与物理路径不一致即视为虚拟
                res.end(JSON.stringify({ content, isVirtual: targetPath !== filePath }))
              } catch (e) {
                res.statusCode = 500
                res.end(JSON.stringify({ error: 'Failed to read file' }))
              }
            } 
            // 保存并全自动化 Git Commit
            else if (req.url === '/api/save-md' && req.method === 'POST') {
              let body = ''
              req.on('data', chunk => { body += chunk })
              req.on('end', () => {
                try {
                  const { filePath, content, message } = JSON.parse(body)
                  
                  // 1. 去除 view/ 前缀 (针对虚拟路由)
                  let targetPath = filePath.replace(/^view\//, '').replace(/^\//, '')

                  // 2. 自动处理虚拟路径
                  if (targetPath.endsWith('.md') && !fs.existsSync(path.resolve(__dirname, '..', targetPath))) {
                     targetPath = targetPath.replace(/\.md$/, '')
                  }

                  const fullPath = path.resolve(__dirname, '..', targetPath)
                  
                  // 1. 历史备份
                  const historyDir = path.resolve(__dirname, 'history')
                  if (!fs.existsSync(historyDir)) fs.mkdirSync(historyDir, { recursive: true })
                  const fileName = path.basename(targetPath)
                  const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
                  const historyPath = path.join(historyDir, `${fileName}_${timestamp}.md`)
                  if (fs.existsSync(fullPath)) fs.writeFileSync(historyPath, fs.readFileSync(fullPath))

                  // 2. 写入文件
                  fs.writeFileSync(fullPath, content)
                  
                  // 3. Git 自动化操作
                  import('node:child_process').then(({ execSync }) => {
                    try {
                      execSync(`git add "${fullPath}"`, { encoding: 'utf8' })
                      execSync(`git commit -m "Auto-edit: ${fileName} - ${message || 'No message'}"`, { encoding: 'utf8' })
                    } catch(gitError: any) {
                      console.warn('Git commit failed (likely no changes or git not init):', gitError.message)
                    }
                  })
                  
                  res.statusCode = 200
                  res.end(JSON.stringify({ success: true, historyFile: historyPath }))
                } catch (e) {
                  res.statusCode = 500
                  res.end(JSON.stringify({ error: 'Failed to save file' }))
                }
              })
            } 
            // Git 回滚接口
            else if (req.url === '/api/rollback' && req.method === 'POST') {
              let body = ''
              req.on('data', chunk => { body += chunk })
              req.on('end', () => {
                try {
                  const { filePath, historyFile } = JSON.parse(body)
                  
                  // 去除 view/ 前缀
                  let targetPath = filePath.replace(/^view\//, '').replace(/^\//, '')
                  if (targetPath.endsWith('.md') && !fs.existsSync(path.resolve(__dirname, '..', targetPath))) {
                      targetPath = targetPath.replace(/\.md$/, '')
                  }

                  const fullPath = path.resolve(__dirname, '..', targetPath)
                  const historyPath = path.resolve(__dirname, 'history', historyFile)
                  
                  if (fs.existsSync(historyPath)) {
                    const content = fs.readFileSync(historyPath, 'utf-8')
                    fs.writeFileSync(fullPath, content)
                    
                    // 记录回滚
                    import('node:child_process').then(({ execSync }) => {
                      try {
                        execSync(`git add "${fullPath}"`)
                        execSync(`git commit -m "Rollback: ${path.basename(filePath)} from ${historyFile}"`)
                      } catch(e) {}
                    })
                    
                    res.statusCode = 200
                    res.end(JSON.stringify({ success: true }))
                  } else {
                    res.statusCode = 404
                    res.end('History file not found')
                  }
                } catch(e) {
                  res.statusCode = 500
                  res.end('Rollback failed')
                }
              })
            }
            // 列出历史记录文件
            else if (req.url?.startsWith('/api/list-history?path=') && req.method === 'GET') {
              const url = new URL(req.url, `http://${req.headers.host}`)
              const filePath = url.searchParams.get('path')
              if (!filePath) return res.end('Path missing')
              
              // 去除 view/ 前缀
              let targetPath = filePath.replace(/^view\//, '').replace(/^\//, '')
              // 统一去掉可能存在的 .md 后缀 (历史记录按文件名匹配)
              const baseName = path.basename(targetPath).replace(/\.md$/, '')

              const historyDir = path.resolve(__dirname, 'history')
              if (!fs.existsSync(historyDir)) return res.end(JSON.stringify({ history: [] }))
              
              const files = fs.readdirSync(historyDir)
                .filter(f => f.startsWith(baseName))
                .map(f => {
                    const stat = fs.statSync(path.join(historyDir, f))
                    return { name: f, time: stat.mtime }
                })
                .sort((a, b) => b.time.getTime() - a.time.getTime())
                
              res.setHeader('Content-Type', 'application/json')
              res.end(JSON.stringify({ history: files }))
            }
            // --- 全能文件管理系统 (CRUD) ---

            // 1. 列出目录文件
            else if (req.url?.startsWith('/api/files/list') && req.method === 'GET') {
              const url = new URL(req.url, `http://${req.headers.host}`)
              const dir = url.searchParams.get('dir') || 'docs'
              if (!isPathSafe(dir)) {
                res.statusCode = 403; // Forbidden
                return res.end('Access Denied');
              }
              const fullPath = path.resolve(__dirname, '..', dir.replace(/^\//, ''))
              try {
                const items = fs.readdirSync(fullPath, { withFileTypes: true }).map(it => ({
                  name: it.name,
                  isDir: it.isDirectory(),
                  path: path.join(dir, it.name).replace(/\\/g, '/')
                }))
                res.setHeader('Content-Type', 'application/json')
                res.end(JSON.stringify(items))
              } catch (e) {
                res.statusCode = 500;
                res.end(JSON.stringify({ error: 'Failed to list directory' }));
              }
            }
            // 2. 创建文件/文件夹
            else if (req.url === '/api/files/create' && req.method === 'POST') {
              let body = ''
              req.on('data', c => body += c)
              req.on('end', () => {
                try {
                  const { parentDir, name, isDir } = JSON.parse(body)
                  const target = path.join(parentDir, name)
                  if (!isPathSafe(target)) {
                    res.statusCode = 403;
                    return res.end('Access Denied');
                  }
                  const fullPath = path.resolve(__dirname, '..', target.replace(/^\//, ''))
                  if (isDir) fs.mkdirSync(fullPath, { recursive: true })
                  else fs.writeFileSync(fullPath, '')
                  res.setHeader('Content-Type', 'application/json')
                  res.end(JSON.stringify({ success: true }))
                } catch (e) {
                  res.statusCode = 500;
                  res.end(JSON.stringify({ error: 'Failed to create file/directory' }));
                }
              })
            }
            // 3. 删除文件/文件夹
            else if (req.url === '/api/files/delete' && req.method === 'POST') {
              let body = ''
              req.on('data', c => body += c)
              req.on('end', () => {
                try {
                  const { filePath } = JSON.parse(body)
                  if (!isPathSafe(filePath)) {
                    res.statusCode = 403;
                    return res.end('Access Denied');
                  }
                  const fullPath = path.resolve(__dirname, '..', filePath.replace(/^\//, ''))
                  if (fs.existsSync(fullPath)) {
                     if (fs.statSync(fullPath).isDirectory()) fs.rmSync(fullPath, { recursive: true })
                     else fs.unlinkSync(fullPath)
                  }
                  // Git Commit 删除
                  import('node:child_process').then(({ execSync }) => {
                     try { execSync(`git add -A ; git commit -m "Delete: ${filePath}"`) } catch(e){}
                  })
                  res.setHeader('Content-Type', 'application/json')
                  res.end(JSON.stringify({ success: true }))
                } catch (e) {
                  res.statusCode = 500;
                  res.end(JSON.stringify({ error: 'Failed to delete file/directory' }));
                }
              })
            }
            // 4. 重命名/移动
            else if (req.url === '/api/files/rename' && req.method === 'POST') {
               let body = ''
               req.on('data', c => body += c)
               req.on('end', () => {
                 try {
                   const { oldPath, newPath } = JSON.parse(body)
                   if (!isPathSafe(oldPath) || !isPathSafe(newPath)) {
                     res.statusCode = 403;
                     return res.end('Access Denied');
                   }
                   const fullOld = path.resolve(__dirname, '..', oldPath.replace(/^\//, ''))
                   const fullNew = path.resolve(__dirname, '..', newPath.replace(/^\//, ''))
                   fs.renameSync(fullOld, fullNew)
                   import('node:child_process').then(({ execSync }) => {
                      try { execSync(`git add -A ; git commit -m "Rename: ${oldPath} -> ${newPath}"`) } catch(e){}
                   })
                   res.setHeader('Content-Type', 'application/json')
                   res.end(JSON.stringify({ success: true }))
                 } catch (e) {
                   res.statusCode = 500;
                   res.end(JSON.stringify({ error: 'Failed to rename/move file/directory' }));
                 }
               })
            }
            // 读取具体的历史备份内容
            else if (req.url?.startsWith('/api/read-history?file=') && req.method === 'GET') {
              const url = new URL(req.url, `http://${req.headers.host}`)
              const file = url.searchParams.get('file')
              try {
                const fullPath = path.resolve(__dirname, 'history', file!)
                const content = fs.readFileSync(fullPath, 'utf-8')
                res.setHeader('Content-Type', 'application/json')
                res.end(JSON.stringify({ content }))
              } catch (e) {
                res.statusCode = 500
                res.end(JSON.stringify({ error: 'Failed to read history file' }))
              }
            }
            // 5. 文件夹历史记录 (Recursive Git Log)
            else if (req.url?.startsWith('/api/git/log-folder?path=') && req.method === 'GET') {
              const url = new URL(req.url, `http://${req.headers.host}`)
              const folderPath = url.searchParams.get('path')
              if (!folderPath) return res.end('Path missing')

              try {
                // 安全检查
                if (!isPathSafe(folderPath)) {
                    res.statusCode = 403;
                    return res.end('Access Denied');
                }
                
                const fullPath = path.resolve(__dirname, '..', folderPath.replace(/^\//, ''))
                
                // 使用 git log --name-status 递归获取目录下变更
                import('node:child_process').then(({ execSync }) => {
                    try {
                        // 格式: hash|date|author|message
                        const log = execSync(`git log --pretty=format:"%h|%ad|%an|%s" --date=iso "${fullPath}"`, { encoding: 'utf8' })
                        
                        const history = log.split('\n').filter(Boolean).map(line => {
                            const [hash, date, author, message] = line.split('|')
                            return { hash, date, author, message }
                        })
                        
                        res.setHeader('Content-Type', 'application/json')
                        res.end(JSON.stringify({ history }))
                    } catch (e: any) {
                        res.statusCode = 500
                        res.end(JSON.stringify({ error: 'Git log failed: ' + e.message }))
                    }
                })
              } catch (e) {
                 res.statusCode = 500
                 res.end(JSON.stringify({ error: 'Failed to read folder history' }))
              }
            }
            else {
              next()
            }
          })
        }
      },
// virtualMarkdownPlugin() removed
    ]
  }
})
