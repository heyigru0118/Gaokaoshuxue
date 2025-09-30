import os
import re
import uuid
import networkx as nx
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.prompts import PromptTemplate
from modelscope import AutoModel, AutoTokenizer, snapshot_download
import google.generativeai as genai
import torch
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Any, List, Optional, Dict
import time
import multiprocessing as mp
from chromadb import PersistentClient
from chromadb.config import Settings as ChromaSettings
import hashlib
import json
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.schema import QueryBundle
from llama_index.core.base.response.schema import StreamingResponse
from llama_index.core.callbacks import CallbackManager
from llama_index.core.schema import TextNode, NodeWithScore

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# 打印脚本绝对路径
print(f"运行脚本: {os.path.abspath(__file__)}")

# 配置环境变量
os.environ["GOOGLE_API_KEY"] = "AIzaSyDviK1KV_qyq9gi6NLKbZcTyg0j-5toUTg"  # 替换为有效密钥

# 存储对话历史
conversation_history = {}

# 定义模型名称和缓存目录
MODEL_NAME = "iic/nlp_gte_sentence-embedding_chinese-base"
CACHE_DIR = "./model_cache"

# 确保缓存目录存在
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# 预下载模型
def ensure_model_download(model_name: str, cache_dir: str):
    print(f"检查模型 {model_name} 是否存在于 {cache_dir}...")
    try:
        snapshot_download(model_id=model_name, cache_dir=cache_dir, revision="v1.0.0")
        print(f"模型 {model_name} 已缓存至 {cache_dir}。")
    except Exception as e:
        print(f"下载模型 {model_name} 失败: {e}")
        exit(1)

# 自定义 ModelScope 嵌入适配器
class ModelScopeEmbedding(BaseEmbedding):
    def __init__(self, model_name: str, cache_folder: str = "./model_cache"):
        super().__init__()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_folder, local_files_only=True)
            self._model = AutoModel.from_pretrained(model_name, cache_dir=cache_folder, local_files_only=True)
            self._model.eval()
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)
        except Exception as e:
            print(f"加载 ModelScope 模型失败: {e}")
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

# 主题关键词字典
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

# 并行处理辅助函数
def process_file(file_path: str, separator: str) -> tuple[str, list[str]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        chunks = content.split(separator)
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        print(f"文件 {file_path}: {len(chunks)} 个分块")
        for i, chunk in enumerate(chunks):
            print(f"分块 {i}: {chunk[:100]}...")
        return file_path, chunks[1:] if len(chunks) > 1 else chunks  # 跳过第0个分块
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return file_path, []

def extract_metadata(metadata: Dict, text: str) -> Dict:
    filename = metadata["file_name"]
    name = filename.replace(".md", "")
    year_match = re.search(r"(20\d{2})年", name)
    year = year_match.group(0) if year_match else "unknown"
    category = name.replace(year, "").strip() if year_match else "unknown"
    print(f"{filename} 的原始文本: {text[:200]}...")
    number_match = re.search(r"第(\d+)题|\[(\d+)\]|题目(\d+)|Question\s*(\d+)|(\d+)\.|\s*(\d+)\s*[-:]", text, re.IGNORECASE)
    question_number = next((g for g in number_match.groups() if g), str(metadata["chunk_index"] + 1)) if number_match else str(metadata["chunk_index"] + 1)
    tags = []
    text_lower = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            tags.append(topic)
    tags = tags if tags else ["其他"]
    tags_str = ", ".join(tags)
    meta = {
        "year": year,
        "category": category,
        "question_number": question_number,
        "tags": tags_str,
        "chunk_id": str(uuid.uuid4())
    }
    print(f"{filename} 提取的元数据: {meta}")
    return meta

def generate_embedding(embed_model, text: str) -> List[float]:
    return embed_model._get_text_embedding(text)

def compute_document_hash(text: str) -> str:
    normalized_text = re.sub(r'\s+', ' ', text.strip())
    return hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()

# 数据摄入流水线
class DataIngestionPipeline:
    def __init__(self, directory_path: str, separator: str = "【分割线】", embed_model=None, num_processes: int = 2, document_store_path: str = "./document_cache.json"):
        self.directory_path = directory_path
        self.separator = separator
        self.embed_model = embed_model
        self.documents = []
        self.num_processes = num_processes
        self.chroma_client = PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name="high_school_math")
        self.document_store_path = document_store_path
        self.cached_documents = self._load_cached_documents()

    def _load_cached_documents(self) -> Dict[str, Dict]:
        if os.path.exists(self.document_store_path):
            try:
                with open(self.document_store_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载缓存文档时出错: {e}")
        return {}

    def _save_cached_documents(self):
        with open(self.document_store_path, 'w', encoding='utf-8') as f:
            json.dump({doc.metadata["chunk_id"]: {
                "text": doc.text,
                "metadata": doc.metadata,
                "embedding": doc.embedding,
                "hash": compute_document_hash(doc.text)
            } for doc in self.documents}, f, ensure_ascii=False, indent=4)

    def run(self):
        start_time = time.time()
        print("启动数据摄入流水线...")
        self.load_and_split_files()
        self.extract_metadata()
        self.generate_embeddings()
        self.store_to_chroma()
        self._save_cached_documents()
        end_time = time.time()
        total_time = end_time - start_time
        print(f"流水线完成。处理了 {len(self.documents)} 个文档，耗时 {total_time:.2f} 秒。")
        return self.documents

    def load_and_split_files(self):
        start_time = time.time()
        print(f"从 {self.directory_path} 加载并分割 Markdown 文件，使用 {self.num_processes} 个进程...")
        file_paths = [os.path.join(root, file) for root, _, files in os.walk(self.directory_path) for file in files if file.endswith(".md")]
        with mp.Pool(processes=self.num_processes) as pool:
            results = pool.starmap(process_file, [(file_path, self.separator) for file_path in file_paths])
        failed_files = []
        for file_result in results:
            if file_result:
                file_path, chunks = file_result
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        doc_hash = compute_document_hash(chunk.strip())
                        cached_doc = None
                        for cached_id, cached_data in self.cached_documents.items():
                            if cached_data.get("hash") == doc_hash and cached_data["text"] == chunk.strip():
                                cached_doc = cached_data
                                break
                        if cached_doc:
                            doc = Document(text=cached_doc["text"], metadata=cached_doc["metadata"], embedding=cached_doc["embedding"])
                        else:
                            doc = Document(text=chunk.strip(), metadata={"file_path": file_path, "file_name": Path(file_path).name, "chunk_index": i})
                        self.documents.append(doc)
            else:
                failed_files.append(file_result[0])
        if failed_files:
            print(f"警告: 无法处理 {len(failed_files)} 个文件: {failed_files}")
        end_time = time.time()
        print(f"加载并分割 {len(self.documents)} 个分块，耗时 {end_time - start_time:.2f} 秒。")

    def extract_metadata(self):
        start_time = time.time()
        print(f"提取文档元数据，使用 {self.num_processes} 个进程...")
        documents_to_process = [doc for doc in self.documents if "tags" not in doc.metadata]
        with mp.Pool(processes=self.num_processes) as pool:
            metadata_results = pool.starmap(extract_metadata, [(doc.metadata, doc.text) for doc in documents_to_process])
        updated_documents = []
        metadata_idx = 0
        for doc in self.documents:
            if "tags" not in doc.metadata:
                meta_update = metadata_results[metadata_idx]
                metadata_idx += 1
                new_doc = Document(text=doc.text, metadata={**doc.metadata, **meta_update})
            else:
                new_doc = doc
            updated_documents.append(new_doc)
        self.documents = updated_documents
        end_time = time.time()
        print(f"提取 {len(self.documents)} 个文档的元数据，耗时 {end_time - start_time:.2f} 秒。")

    def generate_embeddings(self):
        start_time = time.time()
        print(f"生成文档嵌入，使用 {self.num_processes} 个进程...")
        if not self.embed_model:
            print("未提供嵌入模型，跳过嵌入生成。")
            return
        documents_to_process = [doc for doc in self.documents if not hasattr(doc, 'embedding') or doc.embedding is None]
        with mp.Pool(processes=self.num_processes) as pool:
            embeddings = pool.starmap(generate_embedding, [(self.embed_model, doc.text) for doc in documents_to_process])
        updated_documents = []
        embedding_idx = 0
        for doc in self.documents:
            if not hasattr(doc, 'embedding') or doc.embedding is None:
                embedding = embeddings[embedding_idx]
                embedding_idx += 1
                new_doc = Document(text=doc.text, metadata=doc.metadata.copy(), embedding=embedding)
            else:
                new_doc = doc
            updated_documents.append(new_doc)
        self.documents = updated_documents
        end_time = time.time()
        print(f"生成 {len(self.documents)} 个文档的嵌入，耗时 {end_time - start_time:.2f} 秒。")

    def store_to_chroma(self):
        start_time = time.time()
        print("将文档存储到 Chroma 数据库...")
        for doc in self.documents:
            self.collection.add(embeddings=[doc.embedding], documents=[doc.text], metadatas=[doc.metadata], ids=[doc.metadata["chunk_id"]])
        end_time = time.time()
        print(f"文档存储到 Chroma 数据库，耗时 {end_time - start_time:.2f} 秒。")

# 自定义提示模板
text_qa_template = PromptTemplate(
    """你是一个高中数学老师，用户会就历年高考题目向你提问。请严格使用中文回答，所有输出内容必须为简体中文，不得包含英文或其他语言。回答需遵循以下要求：

1. 校对提供的上下文信息中的题目原文和解答，验证其基本计算逻辑，并将其转化为 LaTeX 格式。
2. 分析题目考察的知识点（基于上下文中的标签或内容），使用中文描述。
3. 如果上下文包含题目和解答，完成步骤 1 后，完整输出它们(注意将题目设问和答案分开输出)，并生成一道类似的新习题（包括题目、答案和解析），全部用中文。
4. 如果上下文没有题目和解答，返回“未找到相关题目，请检查查询内容或尝试其他关键词。”（仅用中文）。
5. 回答末尾添加一句鼓励的话，需为中文。

### 用户意图 ###
{user_intent}

### 对话历史 ###
{chat_history}

上下文信息如下（以下是匹配到的内容）：
---------------------
{context_str}
---------------------

根据上下文信息回答以下问题，严格使用简体中文并以 LaTeX 格式输出数学内容：
问题：{query_str}
回答：
"""
)

# 问题压缩提示模板
condense_prompt = PromptTemplate(
"""给定以下对话历史和用户的新问题，将新问题改写为一个独立的问题，适合直接检索上下文。

### 对话历史 ###
{chat_history}

### 新问题 ###
{question}

### 改写后的问题 ###
"""
)

def build_knowledge_graph(documents):
    start_time = time.time()
    G = nx.DiGraph()
    file_docs = {}
    for doc in documents:
        file_path = doc.metadata["file_path"]
        if file_path not in file_docs:
            file_docs[file_path] = []
        file_docs[file_path].append(doc)
    for file_path, docs in file_docs.items():
        file_id = str(uuid.uuid4())
        meta = docs[0].metadata
        year = meta["year"]
        category = meta["category"]
        G.add_node(file_id, type="file", file_path=file_path, metadata={"file_name": meta["file_name"], "year": year, "category": category})
        G.add_node(year, type="year")
        G.add_node(category, type="category")
        G.add_edge(file_id, year, relationship="belongs_to_year")
        G.add_edge(file_id, category, relationship="belongs_to_category")
        for doc in docs:
            chunk_id = doc.metadata["chunk_id"]
            question_number = doc.metadata["question_number"]
            tags = doc.metadata["tags"].split(", ")
            G.add_node(chunk_id, type="question", content=doc.text, metadata=doc.metadata)
            question_number_node = f"Q{question_number}"
            G.add_node(question_number_node, type="question_number", value=question_number)
            G.add_edge(chunk_id, file_id, relationship="belongs_to_file")
            G.add_edge(chunk_id, question_number_node, relationship="has_question_number")
            for tag in tags:
                G.add_node(tag, type="tag")
                G.add_edge(chunk_id, tag, relationship="has_topic")
    end_time = time.time()
    graph_build_time = end_time - start_time
    print(f"知识图谱构建完成，包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边，耗时 {graph_build_time:.2f} 秒")
    node_types = set(data.get("type", "unknown") for _, data in G.nodes(data=True))
    print(f"知识图谱中的节点类型: {node_types}")
    return G, graph_build_time

def build_graph_index(G):
    start_time = time.time()
    graph_index = {
        "by_year": {},
        "by_category": {},
        "by_tag": {}
    }
    for node, data in G.nodes(data=True):
        node_type = data.get("type", "unknown")
        if node_type == "question":
            meta = data["metadata"]
            year = meta["year"]
            category = meta["category"]
            tags = meta["tags"].split(", ")
            if year not in graph_index["by_year"]:
                graph_index["by_year"][year] = []
            graph_index["by_year"][year].append(node)
            if category not in graph_index["by_category"]:
                graph_index["by_category"][category] = []
            graph_index["by_category"][category].append(node)
            for tag in tags:
                if tag not in graph_index["by_tag"]:
                    graph_index["by_tag"][tag] = []
                graph_index["by_tag"][tag].append(node)
    end_time = time.time()
    graph_index_time = end_time - start_time
    print(f"知识图谱索引构建完成，耗时 {graph_index_time:.2f} 秒")
    invalid_nodes = []
    for index_type, index_data in graph_index.items():
        for key, nodes in index_data.items():
            for node in nodes:
                if node not in G.nodes or G.nodes[node].get("type") != "question":
                    invalid_nodes.append((index_type, key, node))
    if invalid_nodes:
        print(f"索引中的无效节点: {invalid_nodes}")
    else:
        print("索引中的所有节点均为有效问题节点。")
    return graph_index, graph_index_time

def build_vector_index(documents):
    start_time = time.time()
    try:
        index = VectorStoreIndex.from_documents(documents)
        end_time = time.time()
        vector_index_time = end_time - start_time
        print(f"向量索引构建成功，耗时 {vector_index_time:.2f} 秒")
        return index, vector_index_time
    except Exception as e:
        print(f"构建向量索引失败: {e}")
        exit(1)

def normalize_query_components(year, category, question_number):
    if year:
        year = year.replace("年", "")
        if len(year) == 2:
            year = "20" + year
        year += "年"
    if category:
        category = category.replace("卷", "").strip()
    if question_number:
        question_number = question_number.replace("第", "").replace("题", "")
        if question_number.startswith("T"):
            question_number = question_number.replace("T", "")
    return year, category, question_number

def truncate_context(context_str, max_length=2000):
    if len(context_str) > max_length:
        return context_str[:max_length] + "... [内容已截断以避免超限]"
    return context_str

def synthesize_response(query_str, context_str, nodes: List[Document], user_intent: str, chat_history: str) -> str:
    synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",
        llm=Settings.llm,
        text_qa_template=text_qa_template
    )
    try:
        # 转换为 NodeWithScore
        node_with_scores = []
        for doc in nodes:
            text_node = TextNode(text=doc.text, metadata={})  # 排除元数据
            node_with_score = NodeWithScore(node=text_node, score=1.0)
            node_with_scores.append(node_with_score)
        
        # 记录提示以供调试
        prompt = text_qa_template.format(
            user_intent=user_intent,
            chat_history=chat_history,
            context_str=context_str,
            query_str=query_str
        )
        print(f"发送给 LLM 的提示: {prompt[:500]}...")  # 记录前 500 字符
        
        # 生成响应
        response = synthesizer.synthesize(
            query=query_str,
            nodes=node_with_scores,
            additional_context={"user_intent": user_intent, "context_str": context_str, "chat_history": chat_history}
        )
        response_text = response.response if response.response else context_str

        # 清理响应：移除元数据标记并规范化
        response_text = re.sub(r'\[metadata:.*?\]', '', response_text)  # 移除元数据标签
        response_text = response_text.replace("\uf07b", "{").replace("\uf07d", "}").strip().replace("\r\n", "\n")
        
        # 确保 LaTeX 格式
        response_text = re.sub(r'(\$.*?\$)', r'\1', response_text)  # 保留行内数学
        response_text = re.sub(r'(\\\[\s*.*?\s*\\\])', r'\1', response_text, flags=re.DOTALL)  # 保留显示数学
        
        # 如果没有鼓励的话，添加一句
        if not re.search(r'继续努力|加油|保持信心', response_text):
            response_text += "\n\n继续努力，你一定能掌握这道题！"
        
        print(f"合成响应: {response_text[:200]}...")
        return response_text
    except Exception as e:
        print(f"synthesize_response 出错: {str(e)}")
        error_message = ""
        if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
            error_message = "API 配额不足，请稍后重试或检查 Google Gemini API 配额。"
        elif "network" in str(e).lower():
            error_message = "网络连接失败，请检查网络后重试。"
        else:
            error_message = "处理查询时发生错误，请检查输入或稍后重试。"
        return f"错误：{error_message}"

def query_index(index, knowledge_graph, graph_index, year_in_query, category_in_query, question_number, additional_question, query_str, chat_history, top_k=3):
    start_time = time.time()
    try:
        # 规范化查询组件
        year_in_query, category_in_query, question_number = normalize_query_components(year_in_query, category_in_query, question_number)

        # 如果提供了 question_number，转换为整数
        if question_number:
            try:
                question_number = int(question_number)
            except ValueError:
                question_number = None

        # 从 additional_question 或 query_str 中提取查询标签
        query_tags = set()
        if additional_question:
            for topic, keywords in TOPIC_KEYWORDS.items():
                if topic == additional_question or any(keyword in additional_question.lower() for keyword in keywords):
                    query_tags.add(topic)
        if not query_tags:
            query_lower = query_str.lower()
            for topic, keywords in TOPIC_KEYWORDS.items():
                if any(keyword in query_lower for keyword in keywords):
                    query_tags.add(topic)

        # 知识图谱检索（优先级 1：精确匹配）
        relevant_docs = []
        if year_in_query and category_in_query and question_number is not None:
            candidate_nodes = set()
            if year_in_query in graph_index["by_year"]:
                candidate_nodes.update(graph_index["by_year"][year_in_query])
            if category_in_query in graph_index["by_category"]:
                candidate_nodes &= set(graph_index["by_category"][category_in_query]) if candidate_nodes else set(graph_index["by_category"][category_in_query])
            for node in candidate_nodes:
                data = knowledge_graph.nodes[node]
                meta = data["metadata"]
                doc_question_number = meta.get("question_number")
                try:
                    doc_question_number = int(doc_question_number)
                except (ValueError, TypeError):
                    doc_question_number = None
                if doc_question_number == question_number:
                    relevant_docs.append(Document(text=data["content"], metadata=meta))
            print(f"精确匹配 - 候选节点: {len(candidate_nodes)}, 相关文档: {len(relevant_docs)}")
            if relevant_docs:
                print(f"精确匹配文档: {[doc.metadata['question_number'] for doc in relevant_docs]}")

        # 知识图谱检索（优先级 2：基于主题的匹配）
        if not relevant_docs and (year_in_query or category_in_query or query_tags):
            candidate_nodes = set()
            if year_in_query and year_in_query in graph_index["by_year"]:
                candidate_nodes.update(graph_index["by_year"][year_in_query])
            if category_in_query and category_in_query in graph_index["by_category"]:
                candidate_nodes &= set(graph_index["by_category"][category_in_query]) if candidate_nodes else set(graph_index["by_category"][category_in_query])
            for tag in query_tags:
                if tag in graph_index["by_tag"]:
                    candidate_nodes &= set(graph_index["by_tag"][tag]) if candidate_nodes else set(graph_index["by_tag"][tag])
            for node in candidate_nodes:
                data = knowledge_graph.nodes[node]
                meta = data["metadata"]
                relevant_docs.append(Document(text=data["content"], metadata=meta))
            print(f"主题匹配 - 候选节点: {len(candidate_nodes)}, 相关文档: {len(relevant_docs)}")
            if relevant_docs:
                print(f"主题匹配文档: {[doc.metadata['question_number'] for doc in relevant_docs]}")
                for doc in relevant_docs:
                    print(f"文档元数据: {doc.metadata}")

        # 向量索引检索（回退）
        if not relevant_docs:
            retriever = index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(query_str)
            relevant_docs = [Document(text=node.get_text(), metadata=node.metadata) for node in nodes]
            print(f"向量索引回退 - 检索到 {len(nodes)} 个节点")
            if relevant_docs:
                print(f"向量匹配文档: {[doc.metadata.get('question_number', 'N/A') for doc in relevant_docs]}")

        # 记录状态
        print(f"主题匹配后: {len(relevant_docs)} 个相关文档")

        # 按问题编号排序文档
        def get_question_number(doc):
            try:
                return int(doc.metadata.get("question_number", float('inf')))
            except (ValueError, TypeError):
                return float('inf')
        relevant_docs.sort(key=get_question_number)

        # 处理空的 question_number
        if not question_number and relevant_docs:
            print("未提供问题编号，返回所有主题匹配的文档")

        # 构建用户意图和上下文
        user_intent = f"用户询问的是{year_in_query}{category_in_query}第{question_number}题" if year_in_query and category_in_query and question_number else f"用户询问的是{query_str}"
        if additional_question:
            user_intent += f"，题目类别：{additional_question}"
        context_str = "\n\n".join(f"**第{doc.metadata['question_number']}题**:\n{doc.text}" for doc in relevant_docs if "question_number" in doc.metadata) if relevant_docs else "未找到相关题目内容。"
        context_str = truncate_context(context_str)

        # 转换为 synthesizer 的节点
        nodes = [Document(text=doc.text, metadata=doc.metadata) for doc in relevant_docs]

        # 生成响应
        response_text = synthesize_response(query_str, context_str, nodes, user_intent, chat_history)

        # 移除上下文中的任何剩余元数据
        
        
        end_time = time.time()
        print(f"查询处理完成，耗时 {end_time - start_time:.2f} 秒")
        return {
            "response": response_text,
            "status": "success"
        }
    except ValueError as ve:
        print(f"query_index 中 ValueError: {ve}")
        return {
            "response": "输入格式错误，请检查年份、类别或题号格式。",
            "status": "error",
            "error_type": "validation_error"
        }
    except Exception as e:
        print(f"查询索引失败: {e}")
        error_type = "general_error"
        error_message = "查询失败，请稍后重试。"
        if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
            error_type = "api_quota_error"
            error_message = "API 配额不足，请稍后重试。"
        elif "network" in str(e).lower():
            error_type = "network_error"
            error_message = "网络连接失败，请检查网络后重试。"
        return {
            "response": f"错误：{error_message}",
            "status": "error",
            "error_type": error_type
        }

# 自定义查询引擎类
class CustomQueryEngine(BaseQueryEngine):
    def __init__(self, index, knowledge_graph, graph_index):
        super().__init__(callback_manager=CallbackManager([]))
        self.index = index
        self.knowledge_graph = knowledge_graph
        self.graph_index = graph_index

    def _query(self, query_bundle: QueryBundle) -> StreamingResponse:
        try:
            # 日志记录输入
            print(f"QueryBundle.query_str: {query_bundle.query_str}")
            
            # 确保 query_bundle.query_str 是字典
            if isinstance(query_bundle.query_str, str):
                try:
                    query_dict = json.loads(query_bundle.query_str)
                except json.JSONDecodeError as e:
                    print(f"JSON 解析错误: {e}")
                    return StreamingResponse(
                        response="错误：无效的查询格式。",
                        sources=[],
                    )
            else:
                query_dict = query_bundle.query_str

            # 验证 query_dict 是一个字典
            if not isinstance(query_dict, dict):
                print(f"无效的 query_dict 类型: {type(query_dict)}")
                return StreamingResponse(
                    response="错误：查询数据必须是字典格式。",
                    sources=[],
                )

            chat_history = query_dict.get("chat_history", "")
            result = query_index(
                index=self.index,
                knowledge_graph=self.knowledge_graph,
                graph_index=self.graph_index,
                year_in_query=query_dict.get("year_in_query", ""),
                category_in_query=query_dict.get("category_in_query", ""),
                question_number=query_dict.get("question_number", ""),
                additional_question=query_dict.get("additional_question", ""),
                query_str=query_dict.get("query", ""),
                chat_history=chat_history,
                top_k=3
            )
            return StreamingResponse(response=result["response"], sources=[])
        except Exception as e:
            print(f"CustomQueryEngine._query 出错: {e}")
            return StreamingResponse(
                response=f"错误：查询处理失败 - {str(e)}",
                sources=[],
            )

    async def _aquery(self, query_bundle: QueryBundle) -> StreamingResponse:
        return self._query(query_bundle)

    def _get_prompt_modules(self) -> Dict[str, Any]:
        return {}

    def query(self, query_bundle: Dict) -> Dict:
        try:
            # 日志记录输入
            print(f"query 输入: {query_bundle}")
            
            # 确保 query_bundle 是字典
            if not isinstance(query_bundle, dict):
                print(f"无效的 query_bundle 类型: {type(query_bundle)}")
                return {
                    "response": "错误：查询数据必须是字典格式。",
                    "status": "error",
                    "error_type": "invalid_query_type"
                }

            chat_history = query_bundle.get("chat_history", "")
            return query_index(
                index=self.index,
                knowledge_graph=self.knowledge_graph,
                graph_index=self.graph_index,
                year_in_query=query_bundle.get("year_in_query", ""),
                category_in_query=query_bundle.get("category_in_query", ""),
                question_number=query_bundle.get("question_number", ""),
                additional_question=query_bundle.get("additional_question", ""),
                query_str=query_bundle.get("query", ""),
                chat_history=chat_history,
                top_k=3
            )
        except Exception as e:
            print(f"CustomQueryEngine.query 出错: {e}")
            return {
                "response": f"错误：查询处理失败 - {str(e)}",
                "status": "error",
                "error_type": "query_error"
            }

    async def aquery(self, query_bundle: Dict) -> Dict:
        return self.query(query_bundle)

# 全局字典存储查询引擎
query_engines = {}

# API 端点
@app.route('/api/categories', methods=['GET'])
def get_categories():
    try:
        categories = set()
        for node, data in knowledge_graph.nodes(data=True):
            if data.get("type") == "category":
                categories.add(data["metadata"]["category"] if "metadata" in data else node)
        return jsonify({"categories": sorted(list(categories))})
    except Exception as e:
        print(f"/api/categories 出错: {e}")
        return jsonify({
            "response": "错误：无法获取类别列表。",
            "status": "error",
            "error_type": "server_error"
        }), 500

@app.route('/api/query', methods=['POST'])
def query():
    try:
        print(f"原始请求数据: {request.get_data(as_text=True)}")
        data = request.get_json()
        print(f"解析的 JSON 数据: {data}")
        if not data:
            return jsonify({
                "response": "错误：无效的 JSON 数据。",
                "status": "error",
                "error_type": "invalid_json"
            }), 400

        year_in_query = data.get('year_in_query', '')
        category_in_query = data.get('category_in_query', '')
        question_number = data.get('question_number', '')
        additional_question = data.get('additional_question', '')
        query_str = data.get('query', '')
        session_id = data.get('session_id', str(uuid.uuid4()))

        # 输入验证
        if not any([year_in_query, category_in_query, question_number]):
            return jsonify({
                "response": "错误：请至少提供年份、类别或题号之一。",
                "status": "error",
                "error_type": "missing_parameters"
            }), 400

        # 验证 question_number 格式
        if question_number:
            if not (question_number.isdigit() or question_number == ''):
                return jsonify({
                    "response": "错误：题号必须是数字或空。",
                    "status": "error",
                    "error_type": "invalid_question_number"
                }), 400

        # 如果 query_str 为空，设置默认查询
        if not query_str:
            query_str = "用中文回答，不仅要给出题目原题，还要给出思考过程，解题过程，说明为什么答案是这样的。"
            print(f"未提供查询，使用默认值: {query_str}")

        print(f"收到查询: {query_str} (会话 ID: {session_id})")

        # 初始化或检索查询引擎
        if session_id not in query_engines:
            memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
            custom_query_engine = CustomQueryEngine(index, knowledge_graph, graph_index)
            query_engines[session_id] = custom_query_engine
        else:
            custom_query_engine = query_engines[session_id]

        # 构造自然语言问题供对话历史记录
        question = query_str
        if year_in_query or category_in_query or question_number:
            question = f"{year_in_query}{category_in_query}第{question_number}题：{query_str}"
        if additional_question:
            question += f"（类别：{additional_question}）"

        # 从会话历史中获取对话历史
        chat_history = "\n".join([f"用户: {msg['query']}\n回答: {msg['response']}" for msg in conversation_history.get(session_id, [])])[:1000]  # 限制长度
        print(f"对话历史: {chat_history[:200]}...")

        # 构造查询输入
        query_input = {
            "year_in_query": year_in_query,
            "category_in_query": category_in_query,
            "question_number": question_number,
            "additional_question": additional_question,
            "query": query_str,
            "chat_history": chat_history
        }
        print(f"查询输入: {query_input}")

        # 直接调用 CustomQueryEngine.query
        query_result = custom_query_engine.query(query_input)

        # 存储对话历史
        if session_id not in conversation_history:
            conversation_history[session_id] = []
        conversation_history[session_id].append({"query": question, "response": query_result["response"]})

        print(f"返回结果: {query_result['response'][:200]}...")
        return jsonify({
            "response": query_result["response"],
            "status": query_result["status"],
            "session_id": session_id
        })
    except Exception as e:
        print(f"/api/query 出错: {e}")
        error_type = "server_error"
        error_message = "服务器内部错误，请稍后重试。"
        if "json" in str(e).lower():
            error_type = "invalid_json"
            error_message = "无效的 JSON 数据。"
        return jsonify({
            "response": f"错误：{error_message}",
            "status": "error",
            "error_type": error_type
        }), 500

def main():
    data_directory = "G:\data_text\presentation"
    num_processes = 2
    print(f"\n使用 {num_processes} 个进程运行流水线...")
    pipeline = DataIngestionPipeline(directory_path=data_directory, separator="【分割线】", embed_model=embed_model, num_processes=num_processes)
    documents = pipeline.run()
    if not documents:
        print("未加载任何文档，退出。")
        exit(1)
    global knowledge_graph, graph_index, index
    knowledge_graph, _ = build_knowledge_graph(documents)
    graph_index, _ = build_graph_index(knowledge_graph)
    index, _ = build_vector_index(documents)
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    ensure_model_download(MODEL_NAME, CACHE_DIR)
    global embed_model
    embed_model = ModelScopeEmbedding(model_name=MODEL_NAME, cache_folder=CACHE_DIR)
    Settings.embed_model = embed_model
    try:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        Settings.llm = GoogleGenAI(model="gemini-1.5-flash", api_key=os.environ["GOOGLE_API_KEY"])
        print("LLM 加载成功")
    except Exception as e:
        print(f"设置 LLM 失败: {e}。API 不可用。")
        exit(1)
    main()