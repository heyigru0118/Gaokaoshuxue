import os
import re
import uuid
import networkx as nx
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from modelscope import AutoModel, AutoTokenizer, snapshot_download
import google.generativeai as genai
import torch
import numpy as np
from pathlib import Path
import time
import multiprocessing as mp
from chromadb import PersistentClient
from chromadb.config import Settings as ChromaSettings
import hashlib
import json
import matplotlib.pyplot as plt

# Define model name and cache directory
MODEL_NAME = "iic/nlp_gte_sentence-embedding_chinese-base"
CACHE_DIR = "./model_cache"

# Ensure the cache directory exists
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Pre-download the model to ensure caching
def ensure_model_download(model_name: str, cache_dir: str):
    print(f"Checking if model {model_name} exists in {cache_dir}...")
    try:
        snapshot_download(
            model_id=model_name,
            cache_dir=cache_dir,
            revision="v1.0.0"
        )
        print(f"Model {model_name} is cached in {cache_dir}.")
    except Exception as e:
        print(f"Failed to download model {model_name}: {e}")
        exit(1)

# Custom ModelScope embedding adapter
class ModelScopeEmbedding(BaseEmbedding):
    def __init__(self, model_name: str, cache_folder: str = "./model_cache"):
        super().__init__()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_folder,
                local_files_only=True
            )
            self._model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_folder,
                local_files_only=True
            )
            self._model.eval()
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)
        except Exception as e:
            print(f"Failed to load ModelScope model: {e}")
            exit(1)

    def _get_text_embedding(self, text: str) -> list[float]:
        inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding.flatten().tolist()

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

# Topic keyword dictionary for tagging
TOPIC_KEYWORDS = {
    "集合": ["集合", "并集", "交集", "补集", "子集", "全集"],
    "立体几何": ["立体几何", "棱柱", "棱锥", "球", "体积", "表面积", "截面", "线面角", "二面角"],
    "平面向量": ["平面向量", "向量", "夹角", "单位向量"],
    "函数与导数": ["函数", "单调性", "导数", "极值", "拐点", "对称性", "周期性", "奇偶性", "最小值", "最大值", "最值"],
    "三角函数": ["三角函数", "正弦", "余弦", "正切", "弧度", "周期"],
    "数列": ["数列", "等差", "等比", "递推", "通项", "求和"],
    "概率统计": ["概率", "统计", "期望", "方差", "分布", "随机变量"],
    "解析几何": ["解析几何", "圆锥曲线", "椭圆", "双曲线", "抛物线", "焦点", "准线", "渐近线", "圆", "轨迹", "距离", "斜率"],
    "不等式": ["不等式", "均值不等式", "柯西", "绝对值", "解不等式"],
    "复数": ["复数", "共轭", "复平面"]
}

# Helper functions for parallel processing
def process_file(file_path: str, separator: str) -> tuple[str, list[str]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        chunks = content.split(separator)
        return file_path, chunks
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return file_path, []

def extract_metadata(metadata: dict,text: str) -> dict:
    filename = metadata["file_name"]
    name = filename.replace(".md", "")
    year_match = re.search(r"(20\d{2})年", name)
    year = year_match.group(0) if year_match else "unknown"
    category = name.replace(year, "").strip() if year_match else "unknown"
    question_number = metadata["chunk_index"]
    tags = []
    text_lower = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            tags.append(topic)
    tags = tags if tags else ["其他"]
    tags_str = ", ".join(tags)
    return {
        "year": year,
        "category": category,
        "question_number": question_number,
        "tags": tags_str,
        "chunk_id": str(uuid.uuid4())
    }

def generate_embedding(embed_model, text: str) -> list[float]:
    return embed_model._get_text_embedding(text)

def compute_document_hash(text: str) -> str:
    """Compute SHA-256 hash of document text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# 数据摄取管道的转换器
class DataIngestionPipeline:
    def __init__(self, directory_path: str, separator: str = "【分割符】", embed_model=None, num_processes: int = 1, document_store_path: str = "./document_cache.json"):
        self.directory_path = directory_path
        self.separator = separator
        self.embed_model = embed_model
        self.documents = []
        self.num_processes = num_processes
        self.chroma_client = PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name="high_school_math")
        self.document_store_path = document_store_path
        # Timing results
        self.load_time = 0
        self.metadata_time = 0
        self.embedding_time = 0
        self.store_time = 0
        self.total_time = 0
        # Load cached documents
        self.cached_documents = self._load_cached_documents()

    def _load_cached_documents(self) -> dict[str, dict]:
        """Load cached document information from file."""
        if os.path.exists(self.document_store_path):
            try:
                with open(self.document_store_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cached documents: {e}")
        return {}

    def _save_cached_documents(self):
        """Save processed document information to file."""
        with open(self.document_store_path, 'w', encoding='utf-8') as f:
            json.dump({doc.metadata["chunk_id"]: {
                "text": doc.text,
                "metadata": doc.metadata,
                "embedding": doc.embedding
            } for doc in self.documents}, f, ensure_ascii=False, indent=4)

    def run(self):
        start_time = time.time()
        print("Starting data ingestion pipeline...")
        self.load_and_split_files()
        self.extract_metadata()
        self.generate_embeddings()
        self.store_to_chroma()
        self._save_cached_documents()
        end_time = time.time()
        self.total_time = end_time - start_time
        print(f"Pipeline completed. Processed {len(self.documents)} documents in {self.total_time:.2f} seconds.")
        return self.documents

    def load_and_split_files(self):
        start_time = time.time()
        print(f"Loading and splitting Markdown files from {self.directory_path} with {self.num_processes} processes...")
        file_paths = [os.path.join(root, file) for root, _, files in os.walk(self.directory_path) for file in files if file.endswith(".md")]
        with mp.Pool(processes=self.num_processes) as pool:
            results = pool.starmap(process_file, [(file_path, self.separator) for file_path in file_paths])
        for file_result in results:
            if file_result:
                file_path, chunks = file_result
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        doc_hash = compute_document_hash(chunk.strip())
                        if doc_hash not in self.cached_documents or self.cached_documents[doc_hash]["text"] != chunk.strip():
                            doc = Document(
                                text=chunk.strip(),
                                metadata={
                                    "file_path": file_path,
                                    "file_name": Path(file_path).name,
                                    "chunk_index": i
                                }
                            )
                            self.documents.append(doc)
                        else:
                            # Restore from cache
                            cached_doc = self.cached_documents[doc_hash]
                            doc = Document(
                                text=cached_doc["text"],
                                metadata=cached_doc["metadata"],
                                embedding=cached_doc["embedding"]
                            )
                            self.documents.append(doc)
        end_time = time.time()
        self.load_time = end_time - start_time
        print(f"Loaded and split {len(self.documents)} chunks from Markdown files in {self.load_time:.2f} seconds.")

    def extract_metadata(self):
        start_time = time.time()
        print(f"Extracting metadata for documents with {self.num_processes} processes...")
        with mp.Pool(processes=self.num_processes) as pool:
            metadata_results = pool.starmap(
                extract_metadata,
                [(doc.metadata, doc.text) for doc in self.documents if "metadata" not in doc.metadata or "embedding" not in doc.metadata]
            )
        updated_documents = []
        for doc, meta_update in zip(self.documents, metadata_results + [{} for _ in range(len(self.documents) - len(metadata_results))]):
            new_doc = Document(
                text=doc.text,
                metadata={**doc.metadata, **meta_update} if "metadata" not in doc.metadata or "embedding" not in doc.metadata else doc.metadata
            )
            updated_documents.append(new_doc)
        self.documents = updated_documents
        end_time = time.time()
        self.metadata_time = end_time - start_time
        print(f"Metadata extracted for {len(self.documents)} documents in {self.metadata_time:.2f} seconds.")

    def generate_embeddings(self):
        start_time = time.time()
        print(f"Generating embeddings for documents with {self.num_processes} processes...")
        if not self.embed_model:
            print("No embedding model provided, skipping embedding generation.")
            return
        with mp.Pool(processes=self.num_processes) as pool:
            embeddings = pool.starmap(generate_embedding, [(self.embed_model, doc.text) for doc in self.documents if "embedding" not in doc.metadata])
        updated_documents = []
        for doc, embedding in zip(self.documents, embeddings + [None for _ in range(len(self.documents) - len(embeddings))]):
            new_doc = Document(
                text=doc.text,
                metadata=doc.metadata.copy(),
                embedding=embedding if "embedding" not in doc.metadata else doc.embedding
            )
            updated_documents.append(new_doc)
        self.documents = updated_documents
        end_time = time.time()
        self.embedding_time = end_time - start_time
        print(f"Embeddings generated for {len(self.documents)} documents in {self.embedding_time:.2f} seconds.")

    def store_to_chroma(self):
        start_time = time.time()
        print("Storing documents to Chroma database...")
        for doc in self.documents:
            self.collection.add(
                embeddings=[doc.embedding],
                documents=[doc.text],
                metadatas=[doc.metadata],
                ids=[doc.metadata["chunk_id"]]
            )
        end_time = time.time()
        self.store_time = end_time - start_time
        print(f"Documents stored to Chroma database in {self.store_time:.2f} seconds.")

def plot_comparison(results):
    process_counts =[2]
    stages = ['Load & Split', 'Extract Metadata', 'Generate Embeddings', 'Store to Chroma', 'Total']
    
    load_times = [results[p]['load'] for p in process_counts]
    metadata_times = [results[p]['metadata'] for p in process_counts]
    embedding_times = [results[p]['embedding'] for p in process_counts]
    store_times = [results[p]['store'] for p in process_counts]
    total_times = [results[p]['total'] for p in process_counts]

    x = np.arange(len(process_counts))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 2*width, load_times, width, label='Load & Split', color='skyblue')
    ax.bar(x - width, metadata_times, width, label='Extract Metadata', color='lightgreen')
    ax.bar(x, embedding_times, width, label='Generate Embeddings', color='salmon')
    ax.bar(x + width, store_times, width, label='Store to Chroma', color='lightcoral')
    ax.bar(x + 2*width, total_times, width, label='Total', color='gold')

    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Performance Comparison with Different Number of Processes')
    ax.set_xticks(x)
    ax.set_xticklabels(process_counts)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.savefig('process_comparison.png')

def main():
    data_directory = "G:\data_text\data_pre\cache"
    process_counts = [2]
    timing_results = {}

    for num_processes in process_counts:
        print(f"\nRunning pipeline with {num_processes} processes...")
        pipeline = DataIngestionPipeline(
            directory_path=data_directory,
            separator="【分割符】",
            embed_model=embed_model,
            num_processes=num_processes
        )
        documents = pipeline.run()

        if not documents:
            print("No documents loaded. Exiting.")
            exit(1)

        timing_results[num_processes] = {
            'load': pipeline.load_time,
            'metadata': pipeline.metadata_time,
            'embedding': pipeline.embedding_time,
            'store': pipeline.store_time,
            'total': pipeline.total_time
        }

    # Plot the comparison
    plot_comparison(timing_results)
    print("Comparison plot saved as 'process_comparison.png'")

if __name__ == "__main__":
    # Pre-download the model
    ensure_model_download(MODEL_NAME, CACHE_DIR)

    # Set global embedding model
    global embed_model
    embed_model = ModelScopeEmbedding(
        model_name=MODEL_NAME,
        cache_folder=CACHE_DIR
    )
    Settings.embed_model = embed_model

    # Set global LLM (optional for this experiment, but keeping for consistency)
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDviK1KV_qyq9gi6NLKbZcTyg0j-5toUTg"
    try:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        Settings.llm = GoogleGenAI(model="gemini-1.5-pro", api_key=os.environ["GOOGLE_API_KEY"])
        print("LLM loaded successfully")
    except Exception as e:
        print(f"Failed to set LLM: {e}. API is unavailable.")
        exit(1)

    main()