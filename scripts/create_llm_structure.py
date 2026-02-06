import os
import shutil

# Base path for the knowledge base
BASE_DIR = r"d:\ALL IN AI\metauniverse-blog\docs\knowledge\llm-mastery"

# The structure definition
# Key is folder name (or file name if ends with .md)
# Value is list of children or None if it's a file
structure = {
    "1. 导论与基础": [
        {
            "1.1 学习路线与知识图谱": [
                "人工智能学习路线.md",
                "The Document is All You Need.md"
            ]
        },
        "1.2 科普与行业杂谈.md",
        "1.3 发展历程与趋势展望.md"
    ],
    "2. 核心原理与架构": [
        {
            "2.1 深度学习基础组件": [
                "2.1.1 前馈网络 (FFN) 与激活函数.md",
                "2.1.2 归一化层 (Normalization).md",
                "2.1.3 残差连接 (Residual).md",
                "2.1.4 位置编码 (Positional Encoding).md"
            ]
        },
        {
            "2.2 Transformer核心": [
                "2.2.1 自注意力机制 (Self-Attention).md",
                {
                    "2.2.2 多头注意力变体": [
                        "MLA专题.md"
                    ]
                },
                {
                    "2.2.4 稀疏注意力": [
                        "MoBA架构深度解析.md",
                        "原生稀疏注意力机制 NSA.md"
                    ]
                },
                "2.2.5 线性注意力.md"
            ]
        },
        {
            "2.3 前沿架构与变体": [
                {
                    "2.3.1 混合专家模型 (MoE)": [
                        "DeepSeek-MoE.md",
                        "MoE的工程实践.md"
                    ]
                },
                "2.3.2 状态空间模型 (SSM - Mamba).md",
                "2.3.3 线性RNN (RWKV).md",
                "2.3.4 新兴架构探索.md"
            ]
        }
    ],
    "3. 预训练": [
        {
            "3.1 预训练数据": [
                "3.1.1 预训练数据收集.md",
                "3.1.2 数据蒸馏.md"
            ]
        },
        "3.2 分词器(Tokenizer).md",
        {
            "3.2 预训练全流程": [
                "3.2.1 训练Tokenizer.md",
                "3.2.2 确定模型结构与参数.md",
                "3.2.3 训练框架选择.md",
                "3.2.4 预训练策略.md",
                "3.2.5 训练容灾与监控.md",
                "3.2.6 Scaling Law 研究与应用.md",
                "3.2.7 继续预训练.md"
            ]
        },
        {
            "3.3 预训练评估": [
                "3.3.1 困惑度 (PPL).md",
                "3.3.2 通用基准 (Benchmarks).md",
                "3.3.3 大海捞针测试.md",
                "3.3.4 特定能力评估.md"
            ]
        }
    ],
    "4. 后训练": [
        "尺度定律的再发现.md",
        "4.1 后训练数据.md",
        "4.2 SFT (监督微调).md",
        "4.3 PEFT (参数高效微调).md",
        {
            "4.4 对齐技术": [
                "强化学习的数学原理.md",
                {
                    "4.4.1 基于奖励模型的RL": [
                        "TRPO.md", "PPO.md", "GRPO.md", "GSPO.md", "GMPO.md"
                    ]
                },
                {
                    "4.4.2 无奖励模型的对齐": [
                        "ORPO.md", "DPO.md"
                    ]
                },
                "4.4.3 RLAIF.md",
                "4.4.4 其他对齐技术.md"
            ]
        }
    ],
    "5. 主流模型全解": [
        "5.1 模型对比总结.md", "5.2 国内大模型.md", "5.3 国外大模型.md", "5.4 大模型发展历程.md"
    ],
    "6. 训练与推理优化": [
        "FlashAttention.md",
        "无界的想平线.md",
        "PagedAttention原理.md",
        "训练框架.md",
        "推理框架.md",
        "显存占用分析.md",
        {
            "6.1 训练优化": [
                 "6.1.1 PagedAttention原理(副本).md", "6.1.2 分布式训练.md", "6.1.3 混合精度训练.md"
            ]
        },
        "6.2 推理优化.md",
        "6.3 模型压缩.md",
        "6.5 优化器.md"
    ],
    "7. LLM应用开发": [
        "7.1 Prompt工程.md", "7.2 RAG.md", "7.3 Agent.md", "7.4 Function Calling.md", "7.5 MCP.md"
    ],
    "8. 多模态": [
        "8.1 核心概念与架构.md", "8.2 视觉语言模型.md", "8.3 音频与语音模型.md", "8.4 视频理解模型.md", "8.5 多模态对齐技术.md"
    ],
    "9. AI工程化与基础设施": [
        "9.1 硬件基础.md", "9.2 系统软件.md", "9.3 分布式训练框架.md", "9.4 推理服务框架.md"
    ],
    "10. 综述与前沿论文": [
        "从OCR到OCR-2.0.md", "LLM技术ICL_Principle.md", "2024年终总结.md", "全面综述.md", "Llama3技术整理.md", "2023年Agent调研.md", "Prefix场景Attention优化.md", "工具学习综述.md", "架构全景图.md"
    ],
    "11. 实践与资源": [
        "11.1 项目实践与代码库.md", "11.2 面试集锦.md", "11.3 资源索引.md"
    ],
    "12. 技术回顾与展望": [
        "Self-MoA.md", "Jeff Dean演讲.md", "DeepSeek硬核解读.md", "2025技术演进.md", "OpenAI愿景推演.md", "密度法则.md", "2024多模态回顾.md", "通向AGI的四层阶梯.md"
    ],
    "系列专栏": [
        "大模型面试题.md", "面经分享.md", "未命名文档1.md", "未命名文档2.md"
    ]
}

def create_structure(base, content):
    if not os.path.exists(base):
        os.makedirs(base)
    
    # Handle list vs dict
    items = []
    if isinstance(content, dict):
        items = content.items()
    elif isinstance(content, list):
        # normalize to list of (name, subcontent)
        temp_items = []
        for item in content:
            if isinstance(item, str):
                temp_items.append((item, None))
            elif isinstance(item, dict):
                for k, v in item.items():
                    temp_items.append((k, v))
        items = temp_items
    
    for name, sub_content in items:
        # Check if it's a file
        if name.endswith('.md'):
            file_path = os.path.join(base, name)
            if not os.path.exists(file_path):
                print(f"Creating file: {file_path}")
                with open(file_path, 'w', encoding='utf-8') as f:
                    title = name.replace('.md', '')
                    f.write(f"---\ntitle: {title}\n---\n\n# {title}\n\nRunning content...\n")
        else:
            # It's a directory
            dir_path = os.path.join(base, name)
            if not os.path.exists(dir_path):
                print(f"Creating directory: {dir_path}")
                os.makedirs(dir_path)
            
            if sub_content:
                create_structure(dir_path, sub_content)
    
    # Create a simple index.md for the folder itself if it doesn't exist
    # This is optional now with our sidebar fix, but good for navigation click
    # But wait, if we create index.md, the sidebar will link to it. 
    # If we DON'T, it will be a collapsible label. 
    # For "1. 导论与基础", maybe we want a summary? 
    # Let's NOT create index.md for folders unless specified, to test the sidebar fix.

def main():
    print("Starting structure generation...")
    
    # Clean up target dir if it exists to ensure freshness? 
    # The user said "re-implement", maybe clean is better.
    # But safety first, let's just overwrite/add.
    
    # Create root index
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
        
    root_index = os.path.join(BASE_DIR, "index.md")
    with open(root_index, 'w', encoding='utf-8') as f:
        f.write("---\ntitle: LLM从入门到入土\n---\n\n# LLM从入门到入土\n\n这是一个非常详尽的LLM知识库结构。\n")

    create_structure(BASE_DIR, structure)
    print("Done!")

if __name__ == "__main__":
    main()
