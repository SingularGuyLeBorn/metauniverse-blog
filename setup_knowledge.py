
import os

base_path = r"d:\ALL IN AI\metauniverse-blog\docs\knowledge\llm-training"

structure = {
    "00-preface": {
        "title": "00. 序章",
        "files": {
            "01-intro": "什么是 LLM",
            "02-roadmap": "学习路线图"
        }
    },
    "01-environment": {
        "title": "01. 环境搭建",
        "files": {
            "01-gpu": "GPU 选型与租赁 (AutoDL)",
            "02-docker": "Docker 容器化环境配置",
            "03-python": "Python & PyTorch 环境"
        }
    },
    "02-theory": {
        "title": "02. 理论基础",
        "files": {
            "01-transformer": "Transformer 宏观架构",
            "02-attention": "Attention 机制 (MHA, MQA, GQA)",
            "03-position": "位置编码 (RoPE, ALiBi)",
            "04-activation": "激活函数 (SwiGLU, GeLU)",
            "05-norm": "归一化 (LayerNorm, RMSNorm)"
        }
    },
    "03-pretraining": {
        "title": "03. 预训练",
        "files": {
            "01-data-pipeline": "数据管线 (清洗, 去重)",
            "02-tokenizer": "Tokenizer 原理 (BPE, SentencePiece)",
            "03-distributed": "分布式并行 (DP, TP, PP, ZeRO)"
        }
    },
    "04-fine-tuning": {
        "title": "04. 微调技术",
        "files": {
            "01-sft-basics": "SFT 指令微调基础",
            "02-peft": "参数高效微调 (LoRA, QLoRA)",
            "03-frameworks": "分布式微调框架 (Llama-Factory)"
        }
    },
    "05-alignment": {
        "title": "05. 人类对齐",
        "files": {
            "01-rlhf-intro": "RLHF 完整流程",
            "02-reward-model": "Reward Model 训练",
            "03-ppo": "PPO 算法详解",
            "04-dpo": "DPO (Direct Preference Optimization)"
        }
    },
    "06-inference": {
        "title": "06. 推理优化",
        "files": {
            "01-kv-cache": "KV Cache 与 PagedAttention",
            "02-quantization": "量化技术 (GPTQ, AWQ, GGUF)",
            "03-vllm": "vLLM 部署实战"
        }
    }
}

def create_structure():
    # Create base dir if not exists (it should exist)
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Note: We are strictly creating FILES (.md), not folders for the leaf nodes
    # based on the new sidebar.ts logic that supports .md files directly.
    # But to match the "Folder Tree" VS Code look, creating Folders + index.md is also fine.
    # Let's stick to the .md file approach for leaf nodes to be cleaner, 
    # OR folders if the user prefers "Folder Mode".
    # User said: "VS Code left side tree". Files show up in trees too.
    # Let's do folders + index.md for Chapter roots, and .md files for articles.

    # First, handle the root index.md
    with open(os.path.join(base_path, "index.md"), "w", encoding="utf-8") as f:
        f.write("---\ntitle: LLM 从入门到入土\ndate: 2024-02-06\n---\n\n# LLM 从入门到入土\n\n> 那些年，我们一起追过的 Transformer。\n")

    for folder_key, folder_data in structure.items():
        folder_path = os.path.join(base_path, folder_key)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Create Folder Index
        with open(os.path.join(folder_path, "index.md"), "w", encoding="utf-8") as f:
            f.write(f"---\ntitle: {folder_data['title']}\n---\n\n# {folder_data['title']}\n")

        # Create Article Files
        for file_key, file_title in folder_data["files"].items():
            file_path = os.path.join(folder_path, file_key + ".md")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"---\ntitle: {file_title}\n---\n\n# {file_title}\n\n*待补充内容*\n")

if __name__ == "__main__":
    create_structure()
