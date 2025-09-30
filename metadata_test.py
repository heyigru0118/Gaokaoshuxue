import os
import re
import uuid
from pathlib import Path
from typing import Dict, List

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

# 提取元数据的函数
def extract_metadata(metadata: Dict, text: str) -> Dict:
    filename = metadata["file_name"]
    name = filename.replace(".md", "")
    year_match = re.search(r"(20\d{2})年", name)
    year = year_match.group(0) if year_match else "unknown"
    category = name.replace(year, "").strip() if year_match else "unknown"
    question_number = metadata["chunk_index"]
    text_lower = text.lower()
    tags = []
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

# 数据摄取管道的简化版本，仅用于加载和提取元数据
class DataIngestionPipeline:
    def __init__(self, directory_path: str, separator: str = "【分割符】"):
        self.directory_path = directory_path
        self.separator = separator
        self.documents = []

    def load_and_split_files(self):
        file_paths = [os.path.join(root, file) for root, _, files in os.walk(self.directory_path) for file in files if file.endswith(".md")]
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            chunks = content.split(self.separator)
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    doc = {"text": chunk.strip(), "metadata": {"file_path": file_path, "file_name": Path(file_path).name, "chunk_index": i}}
                    self.documents.append(doc)

    def extract_metadata_for_documents(self):
        for doc in self.documents:
            doc["metadata"].update(extract_metadata(doc["metadata"], doc["text"]))

# 主函数：打印前两个节点的元数据
def print_metadata_from_files(directory_path):
    pipeline = DataIngestionPipeline(directory_path)
    pipeline.load_and_split_files()
    pipeline.extract_metadata_for_documents()
    
    # 打印前两个节点的元数据
    for i in range(min(40,len(pipeline.documents))):
        meta = pipeline.documents[i]["metadata"]
        print(f"\nNode {i+1} Metadata:")
        for key, value in meta.items():
            print(f"{key}: {value}")

# 运行脚本
if __name__ == "__main__":
    # 请将 'my_data' 替换为您的实际数据目录路径
    directory_path = "G:\data_text\data_pre\md"
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist. Please create it and add your .md files.")
    else:
        print_metadata_from_files(directory_path)