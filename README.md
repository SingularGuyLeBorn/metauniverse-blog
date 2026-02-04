# MetaUniverse Blog

一个基于 VitePress 构建的 AI 技术博客。

## 🚀 快速开始

```bash
# 安装依赖
npm install

# 启动开发服务器
npm run dev

# 构建生产版本
npm run build

# 预览生产版本
npm run preview
```

## 📁 目录结构

```
metauniverse-blog/
├── docs/                    # 文档源文件
│   ├── .vitepress/         # VitePress 配置
│   │   └── config.ts       # 站点配置
│   ├── public/             # 静态资源
│   ├── posts/              # 博客文章
│   ├── about.md            # 关于页面
│   └── index.md            # 首页
├── .github/
│   └── workflows/          # GitHub Actions
│       └── deploy.yml      # 自动部署配置
└── package.json
```

## 🔧 GitHub Pages 部署配置

1. 在 GitHub 上创建仓库
2. 推送代码到 `main` 分支
3. 进入仓库 Settings → Pages
4. Build and deployment → Source 选择 **GitHub Actions**
5. 每次推送到 `main` 分支都会自动部署

## 📝 添加新文章

在 `docs/posts/` 目录下创建新的 `.md` 文件，然后在 `docs/.vitepress/config.ts` 中更新侧边栏配置。

## 🎨 自定义主题

编辑 `docs/.vitepress/config.ts` 可以自定义：
- 导航栏
- 侧边栏
- 社交链接
- 页脚信息

## 📄 License

MIT
