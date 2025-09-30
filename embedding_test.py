import torch
import numpy as np
from modelscope import AutoModel, AutoTokenizer, snapshot_download
from sklearn.metrics.pairwise import cosine_similarity

# 定义模型名称和缓存路径
MODEL_NAME = "iic/nlp_corom_sentence-embedding_chinese-base"
CACHE_DIR = "./model_cache"

# 下载并加载模型
def load_model(model_name, cache_dir):
    print(f"Downloading/loading model {model_name} to {cache_dir}...")
    snapshot_download(model_id=model_name, cache_dir=cache_dir, revision="v1.0.0")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

# 获取句子的嵌入向量
def get_embedding(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embedding.flatten()

# 主函数
def main():
    # 加载模型
    tokenizer, model, device = load_model(MODEL_NAME, CACHE_DIR)
    
    # 定义测试句子（语义相似但文字不同，以及不相关的句子）
    sentences = [
        "哈哈哈哈",
        "太逗",
       "笑死",
        "哈哈笑",
        "好好笑"
    ]
    
    # 获取每个句子的嵌入向量
    print("\nGenerating embeddings for sentences...")
    embeddings = []
    for i, sentence in enumerate(sentences):
        embedding = get_embedding(sentence, tokenizer, model, device)
        embeddings.append(embedding)
        # 仅显示前5个维度以简化输出
        print(f"Sentence {i+1}: {sentence}")
        print(f"Embedding (first 10dims): {embedding[:10]}\n"   )
    
    # 转换为numpy数组
    embeddings = np.array(embeddings)
    
    # 计算余弦相似度
    print("Computing cosine similarity between sentence pairs...")
    similarity_matrix = cosine_similarity(embeddings)
    
    # 输出相似度矩阵
    print("\nCosine Similarity Matrix:")
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            print(f"Similarity between Sentence {i+1} and Sentence {j+1}: {similarity_matrix[i][j]:.4f}")
        print()

if __name__ == "__main__":
    main()