import re

# 主题关键词字典，用于提取 tags
TOPIC_KEYWORDS = {
    "圆锥曲线": ["圆锥曲线", "椭圆", "双曲线", "抛物线"],
    "解析几何": ["解析几何", "圆", "轨迹", "距离", "斜率"],
    "概率统计": ["概率", "统计", "期望", "方差"],
    "三角函数": ["三角函数", "正弦", "余弦", "正切"],
    "数列": ["数列", "等差", "等比", "递推"],
    # 可根据需要扩展
}

# 查询转化函数
def convert_query_to_conditions(query: str) -> dict:
    # 初始化查询条件
    conditions = {
        "year": None,
        "category": None,
        "question_number": None,
        "tags": None
    }
    
    # 提取年份（例如 "2011年" 或 "21年"）
    year_match = re.search(r"(\d{2,4})年", query)
    if year_match:
        year = year_match.group(1)
        if len(year) == 2:  # 处理 "21年" -> "2021"
            year = "20" + year
        conditions["year"] = year + "年"
    
    # 提取题目编号（例如 "第20题"）
    question_match = re.search(r"第(\d+)题", query)
    if question_match:
        conditions["question_number"] = int(question_match.group(1))
    
    # 提取类别（例如 "天津卷理科卷" 或 "全国新高考I卷"）
    # 假设类别在年份和题目编号之间，或在年份之后
    query_clean = query
    if conditions["year"]:
        query_clean = query_clean.replace(conditions["year"], "")
    if question_match:
        query_clean = query_clean.replace(f"第{conditions['question_number']}题", "")
    # 移除常见无关词语
    query_clean = query_clean.replace("考了什么", "").strip()
    # 类别通常是剩下的主要部分
    conditions["category"] = query_clean
    
    # 提取标签（基于 TOPIC_KEYWORDS）
    query_lower = query.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in query_lower for keyword in keywords):
            conditions["tags"] = topic
            break
    
    return conditions

# 格式化输出
def format_conditions(conditions: dict) -> str:
    year = conditions["year"] if conditions["year"] else "None"
    category = conditions["category"] if conditions["category"] else "None"
    question_number = conditions["question_number"] if conditions["question_number"] is not None else "None"
    tags = conditions["tags"] if conditions["tags"] else "None"
    return f"查询year={year}, category={category}, question_number={question_number}, tags={tags}的Node。"

# 测试函数
def test_query_conversion():
    # 测试用例
    queries = [
        "2018年山东卷理科卷中圆锥曲线考了什么",
        "21年全国新高考I卷第20题考了什么",
        "2020年北京卷三角函数关题目是什么"
    ]
    
    for query in queries:
        print(f"\n原始查询: {query}")
        conditions = convert_query_to_conditions(query)
        formatted = format_conditions(conditions)
        print(f"转化结果: {formatted}")

# 运行测试
if __name__ == "__main__":
    test_query_conversion()