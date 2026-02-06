import os
import re

# Base path for the new knowledge base
base_path = r"d:\ALL IN AI\metauniverse-blog\docs\knowledge\llm-galaxy"

# The raw catalog text provided by the user
raw_catalog = """
1. 导论与基础 (Introduction & Fundamentals)
1.1 学习路线与知识图谱
1.2 科普与行业杂谈
1.3 发展历程与趋势展望

2. 核心原理与架构 (Core Principles & Architecture)
2.1 深度学习基础组件 (Fundamental DL Components)
2.1.1 前馈网络 (FFN) 与激活函数 (Activations)
2.1.2 归一化层 (Normalization Layers)
2.1.3 残差连接 (Residual Connections)
2.1.4 位置编码 (Positional Encoding - RoPE, ALiBi)
2.2 Transformer核心 (The Transformer Core)
2.2.1 自注意力机制 (Self-Attention)
2.2.2 多头注意力变体 (MHA MQA GQA MLA)
2.2.4 稀疏注意力 (NSA MoBA)
2.2.5 线性注意力
2.3 前沿架构与变体 (Advanced Architectures & Variants)
2.3.1 混合专家模型 (MoE - Mixture of Experts)
2.3.2 状态空间模型 (SSM - Mamba)
2.3.3 线性RNN (RWKV)
2.3.4 新兴架构探索 (Google MoR)

3. 预训练 (Pre-training)
3.1 预训练数据
3.1.1 数据收集
3.1.2 数据蒸馏
3.2 分词器 (Tokenizer)
3.3 预训练全流程 (End-to-End Pre-training Workflow)
3.3.1 训练 Tokenizer
3.3.2 确定模型结构
3.3.3 训练框架选择 (Training Frameworks)
3.3.4 预训练策略
3.3.5 训练容灾与监控
3.3.6 Scaling Law 研究与应用
3.3.7 继续预训练
3.4 预训练评估 (Evaluation)
3.4.1 困惑度 (PPL)
3.4.2 通用基准 (Benchmarks)
3.4.3 大海捞针测试 (Needle in a Haystack)
3.4.4 特定能力评估

4. 后训练 (Post-training)
4.1 后训练数据
4.2 SFT (监督微调)
4.3 PEFT (参数高效微调)
4.4 对齐技术 (Alignment Technologies)
4.4.1 基于奖励模型的RL (RLHF/PPO/GRPO)
4.4.2 无奖励模型的对齐 (DPO/ORPO/KTO)
4.4.3 RLAIF
4.4.4 其他对齐技术

5. 主流模型全解 (Mainstream Models)
5.1 模型对比总结
5.2 国内大模型
5.3 国外大模型
5.4 大模型发展历程

6. 训练与推理优化 (Optimization)
6.1 显存与计算优化
6.1.1 FlashAttention
6.1.2 PagedAttention
6.1.3 混合精度训练
6.2 分布式训练 (Distributed Training)
6.3 推理优化 (Inference Optimization)
6.4 模型压缩 (Model Compression)
6.5 优化器 (Optimizers)

7. LLM应用开发 (Application Development)
7.1 Prompt工程
7.2 RAG (检索增强生成)
7.3 Agent (智能体)
7.4 Function Calling
7.5 MCP (Model Context Protocol)

8. 多模态 (Multi-modality)
8.1 核心概念与架构
8.2 视觉语言模型 (VLMs)
8.3 音频与语音模型
8.4 视频理解模型
8.5 多模态对齐技术

9. AI工程化与基础设施 (Infrastructure)
9.1 硬件基础 (GPU/TPU)
9.2 系统软件 (CUDA/Docker)
9.3 分布式训练框架 (DeepSpeed)
9.4 推理服务框架 (vLLM)

10. 综述与前沿论文 (Surveys & Papers)
10.1 经典综述
10.2 年度总结
10.3 技术演进报告

11. 实践与资源 (Practice & Resources)
11.1 项目实践
11.2 面试集锦
11.3 资源索引

12. 技术回顾与展望 (Review & Outlook)
12.1 大佬访谈
12.2 技术趋势
12.3 生态推演
"""

def clean_name(name):
    # Remove special chars but keep spaces/hyphens for readability, then simple sanitization
    name = re.sub(r'[\\/:*?"<>|]', '', name)
    return name.strip()

def create_structure():
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Create root index
    with open(os.path.join(base_path, "index.md"), "w", encoding="utf-8") as f:
        f.write("---\ntitle: 大模型超银河传说\nlayout: page\nsidebar: false\n---\n\n<ClientOnly>\n  <KnowledgeDashboard />\n</ClientOnly>\n\n# 大模型超银河传说\n\n> 探索大模型的星辰大海。\n")

    lines = raw_catalog.split('\n')
    
    # Simple stack to keep track of current directory context
    # stack elements: (level, path)
    # level: hierarchy level (1 for "1.", 2 for "1.1", 3 for "1.1.1")
    stack = [] 

    # Helper to get numeric hierarchical level
    def get_level(line):
        match = re.match(r'^(\d+(\.\d+)*)', line)
        if match:
            dots = match.group(1).count('.')
            return dots + 1
        return 0

    # Helper to get proper directory/file name prefix
    def get_prefix_and_title(line):
        match = re.match(r'^(\d+(\.\d+)*)\s+(.*)', line)
        if match:
            num = match.group(1)
            title = match.group(3)
            # Format numbers to be sortable: 1 -> 01, 1.1 -> 01.01 approx
            # Actually, standard string sort might fail locally, but let's keep original numbering str for clarity
            # Or formatted: 1 -> 01, 2 -> 02. 
            parts = num.split('.')
            formatted_parts = [p.zfill(2) for p in parts]
            dirname = "-".join(formatted_parts)
            return dirname, title, num
        return None, line, "0"

    current_path = base_path

    for line in lines:
        line = line.strip()
        if not line:
            continue

        level = get_level(line)
        dir_prefix, raw_title, num_str = get_prefix_and_title(line)
        
        if not dir_prefix:
            continue

        title_clean = clean_name(raw_title)
        folder_name = f"{dir_prefix}-{title_clean}"
        # Truncate if too long (rare)
        if len(folder_name) > 50:
             folder_name = folder_name[:50]

        # Determine if this item should be a folder or a file
        # Logic: If the NEXT line is a child of THIS line, then THIS line is a folder.
        # Otherwise, strictly speaking, it could be a file.
        # BUT, the user prompt implies a structure where everything listable is a node.
        # To support "Folder/index.md" structure which is flexible:
        # We will make EVERYTHING a folder with index.md, EXCEPT leaf nodes if we knew them.
        # But since we are parsing stream-wise, let's just make EVERYTHING a folder for safety?
        # No, that's too nested.
        # Standard approach: Chapters (Level 1) are folders. Sections (Level 2) are folders.
        # Sub-sections (Level 3) are files?
        # Let's check max depth. Looks like max level is 3 (x.x.x).
        
        # Strategy:
        # Level 1 (e.g. "1. xxx"): Folder
        # Level 2 (e.g. "1.1 xxx"): Folder (if it has children) or File?
        # Many 2.x items have 2.x.x children.
        # Items like "1.2 科普" seem to be leaves.
        # To be safe and consistent with the sidebar generator:
        # We can implement a "Lazy" approach or just make Level 1 and 2 Folders, and Level 3 Files.
        # Exception: If Level 2 has no children, it stays a folder with index.md (acts like file).
        
        # Actually, let's process level logic:
        # Stack management:
        # If current level > stack top level, push.
        # If current level <= stack top level, pop until we find parent.
        
        # Reset current path to root for calculation
        
        # We need a proper tree data structure first to know if a node is a leaf.
        pass

    # Re-parsing into a tree to determine Leaf vs Folder
    nodes = []
    for line in lines:
        line = line.strip()
        if not line: continue
        level = get_level(line)
        if level > 0:
            dir_prefix, raw_title, num_str = get_prefix_and_title(line)
            nodes.append({
                "level": level,
                "title": clean_name(raw_title),
                "num_str": num_str,
                "is_folder": False # Will determine later
            })

    # Determine folders
    for i in range(len(nodes) - 1):
        if nodes[i+1]["level"] > nodes[i]["level"]:
            nodes[i]["is_folder"] = True
    
    # Level 1 is always a folder for organization
    for node in nodes:
        if node["level"] == 1:
            node["is_folder"] = True

    # Now generation
    # stack to hold paths: {level: path}
    path_stack = {0: base_path}

    for node in nodes:
        level = node["level"]
        parent_path = path_stack[level - 1]
        
        # Format name
        safe_title = node["title"].split(' ')[0] # simplified for path
        # Use full num_str hierarchy for sorting
        dirname = f"{node['num_str']}-{safe_title}"
        dirname = re.sub(r'[^\w\-\.]', '', dirname) # Remove non-ascii for path safety if wanted, but Chinese is fine
        dirname = f"{node['num_str']}-{node['title']}"  # Keep full title for clarity
        dirname = re.sub(r'[\\/:*?"<>|]', '', dirname).strip() # Sanitize
        
        full_path = os.path.join(parent_path, dirname)
        
        if node["is_folder"]:
            if not os.path.exists(full_path):
                os.makedirs(full_path)
            
            # Create index.md for the folder
            index_path = os.path.join(full_path, "index.md")
            with open(index_path, "w", encoding="utf-8") as f:
                f.write(f"---\ntitle: {node['title']}\n---\n\n# {node['title']}\n\n*暂无内容，敬请期待...*\n")
            
            path_stack[level] = full_path
        else:
            # Create .md file
            file_path = full_path + ".md"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"---\ntitle: {node['title']}\n---\n\n# {node['title']}\n\n*正文内容待补充...*\n")

if __name__ == "__main__":
    create_structure()
