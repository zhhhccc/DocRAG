from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# 1. 配置
# ============================================================

PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "docx_knowledge"
OLLAMA_BASE_URL = "http://localhost:11434"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3.5:9b"

VECTOR_TOP_K = 10
BM25_TOP_K = 10
FINAL_TOP_K = 6
RRF_K = 60


# ============================================================
# 2. 初始化 Ollama 和 Chroma
# ============================================================

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL,
)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
)

collection = vector_store._collection

count = collection.count()
print(f"Chroma collection: {COLLECTION_NAME}")
print(f"文档块数量: {count}")

if count == 0:
    raise RuntimeError(
        "Chroma collection 为空。请检查 PERSIST_DIR 和 COLLECTION_NAME，"
        "确保连接的是原有向量数据库。"
    )


# ============================================================
# 3. 加载 BM25 所需的语料
# ============================================================

def load_all_documents() -> List[Document]:
    """
    从现有 Chroma collection 读取已经存储的文本和元数据，
    供 BM25 建立关键词索引。

    注意：这里不重新向量化，也不写入新的向量。
    """
    result = collection.get(
        include=["documents", "metadatas"]
    )

    texts = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    ids = result.get("ids") or []

    documents = []

    for i, text in enumerate(texts):
        if not text or not text.strip():
            continue

        metadata = metadatas[i] if i < len(metadatas) else {}
        metadata = dict(metadata or {})

        metadata["_chroma_id"] = ids[i] if i < len(ids) else str(i)

        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    return documents


corpus = load_all_documents()

if not corpus:
    raise RuntimeError("没有从 Chroma 读取到可用于 BM25 的文本。")

print(f"BM25 索引文档块数量: {len(corpus)}")


# ============================================================
# 4. 中文 + 代码友好的分词
# ============================================================

def tokenize(text: str) -> List[str]:
    """
    保留中文词组、英文标识符、命令、路径、版本号等。

    例如：
      rosrun turtlesim turtlesim_node
      /cmd_vel
      geometry_msgs/Twist
      catkin_make
      CMakeLists.txt
      qwen3.5:9b
    """
    text = text.lower()

    tokens = re.findall(
        r"[a-z_][a-z0-9_./:+#-]*"
        r"|\d+(?:\.\d+)*"
        r"|[\u4e00-\u9fff]+",
        text,
    )

    result = []

    for token in tokens:
        result.append(token)

        if re.fullmatch(r"[\u4e00-\u9fff]+", token) and len(token) > 2:
            result.extend(
                token[i:i + 2]
                for i in range(len(token) - 1)
            )

    return result


bm25_retriever = BM25Retriever.from_documents(
    corpus,
    preprocess_func=tokenize,
)

bm25_retriever.k = BM25_TOP_K


# ============================================================
# 5. 混合检索：向量召回 + BM25 + RRF
# ============================================================

def vector_search_with_ids(query: str, k: int) -> List[Document]:
    """
    用底层 collection.query() 做向量检索，并注入 _chroma_id。

    关键点：
    1. 不再用 vector_store.similarity_search()，因为它返回的 Document
       不带原始 Chroma ID，导致去重和相关性判定失效。
    2. 手动把 query 编码成向量，调用 collection.query() 拿 ids/documents/
       metadatas/distances。
    3. 保证返回的每个 Document 都带 _chroma_id。
    """
    query_emb = embeddings.embed_query(query)

    result = collection.query(
        query_embeddings=[query_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    ids = (result.get("ids") or [[]])[0]
    texts = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    docs: List[Document] = []

    for i, cid in enumerate(ids):
        text = texts[i] if i < len(texts) else ""
        if not text or not text.strip():
            continue

        metadata = metadatas[i] if i < len(metadatas) else {}
        metadata = dict(metadata or {})
        metadata["_chroma_id"] = cid

        if i < len(distances):
            metadata["_vector_distance"] = distances[i]

        docs.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    return docs


def document_key(doc: Document) -> str:
    """
    优先使用原始 Chroma ID 去重；
    如果没有 ID，则用内容和元数据构造稳定键。
    """
    chroma_id = doc.metadata.get("_chroma_id")

    if chroma_id:
        return str(chroma_id)

    return (
        str(doc.metadata.get("source", ""))
        + "|"
        + str(doc.metadata.get("start_seq", ""))
        + "|"
        + doc.page_content
    )


def hybrid_search(query: str) -> List[Document]:
    """
    RRF 分数：
        score(d) = Σ 1 / (RRF_K + rank(d))

    不直接相加 BM25 分数和向量距离，
    避免两种不同尺度的分数互相影响。
    """
    # 向量召回：改为带 ID 的版本
    vector_docs = vector_search_with_ids(query, VECTOR_TOP_K)

    # BM25 召回：语料本身带 _chroma_id
    bm25_docs = bm25_retriever.invoke(query)

    scores = defaultdict(float)
    documents_by_key = {}

    # 向量检索排名
    for rank, doc in enumerate(vector_docs, start=1):
        key = document_key(doc)
        scores[key] += 1.0 / (RRF_K + rank)
        documents_by_key[key] = doc

    # BM25 检索排名
    for rank, doc in enumerate(bm25_docs, start=1):
        key = document_key(doc)
        scores[key] += 1.0 / (RRF_K + rank)

        # 如果同一个块在两路结果中出现，保留已有对象即可；
        # 但若已有对象缺 _chroma_id，而当前对象有，则用当前的补全。
        existing = documents_by_key.get(key)
        if existing is None:
            documents_by_key[key] = doc
        else:
            if not existing.metadata.get("_chroma_id") and doc.metadata.get("_chroma_id"):
                documents_by_key[key] = doc

    ranked_keys = sorted(
        scores.keys(),
        key=lambda key: scores[key],
        reverse=True,
    )

    results = []

    for key in ranked_keys[:FINAL_TOP_K]:
        doc = documents_by_key[key]

        metadata = dict(doc.metadata)
        metadata["rrf_score"] = round(scores[key], 6)

        results.append(
            Document(
                page_content=doc.page_content,
                metadata=metadata,
            )
        )

    return results


# ============================================================
# 6. 构建问答 Prompt
# ============================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
你是一个严谨的技术文档问答助手。

请严格遵守以下要求：
1. 优先依据提供的检索上下文回答，不要把猜测当作文档事实。
2. 如果上下文没有足够信息，请明确说“当前知识库检索到的内容不足以确定”，
   并指出还缺少什么信息。
3. 对终端命令、代码、配置项、ROS 话题名、服务名、文件路径等，
   尽量保留原始拼写，不要随意改写。
4. 解释代码时，区分文档中明确写出的内容与推断。
5. 如果多个上下文存在差异，不要自行编造统一结论，请说明差异。
6. 回答末尾列出使用的来源。优先使用上下文中已有的 source、
   heading_path、type 等元数据。
7. 不要声称已经执行了命令或验证了代码，除非上下文明确提供了验证结果。
"""
        ),
        (
            "human",
            """
检索上下文：
{context}

用户问题：
{question}

请使用中文回答。必要时使用 Markdown 代码块展示命令或代码。
"""
        ),
    ]
)

llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
)

qa_chain = prompt | llm | StrOutputParser()


# ============================================================
# 7. 格式化上下文和来源
# ============================================================

def format_context(docs: List[Document]) -> str:
    sections = []

    for i, doc in enumerate(docs, start=1):
        metadata = doc.metadata

        source = metadata.get("source", "未知来源")
        heading_path = metadata.get("heading_path", "")
        chunk_type = metadata.get("type", "text")

        if isinstance(heading_path, list):
            heading_path = " > ".join(map(str, heading_path))

        sections.append(
            f"[资料 {i}]\n"
            f"来源：{source}\n"
            f"章节：{heading_path or '未记录'}\n"
            f"类型：{chunk_type}\n"
            f"内容：\n{doc.page_content}"
        )

    return "\n\n".join(sections)


def print_sources(docs: List[Document]) -> None:
    '''print("\n========== 检索来源 ==========")

    for i, doc in enumerate(docs, start=1):
        metadata = doc.metadata

        print(f"\n[{i}]")
        print("source:", metadata.get("source", "未知来源"))
        print("heading_path:", metadata.get("heading_path", "未记录"))
        print("type:", metadata.get("type", "text"))
        print("_chroma_id:", metadata.get("_chroma_id", "未记录"))
        print("rrf_score:", metadata.get("rrf_score", "未记录"))

        preview = doc.page_content[:300].replace("\n", " ")
        print("内容预览:", preview)'''


# ============================================================
# 8. 对外问答接口
# ============================================================

def ask(question: str) -> str:
    if not question.strip():
        return "请输入问题。"

    docs = hybrid_search(question)

    if not docs:
        return "没有检索到相关内容，暂时无法依据知识库回答。"

    context = format_context(docs)

    answer = qa_chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    print_sources(docs)

    return answer


# ============================================================
# 9. 命令行交互
# ============================================================

if __name__ == "__main__":
    print("\nDOCX 混合检索问答已启动。输入 exit 或 quit 退出。\n")

    while True:
        question = input("请输入问题：").strip()

        if question.lower() in {"exit", "quit"}:
            break

        try:
            answer = ask(question)
            print("\n========== 回答 ==========")
            print(answer)
            print()
        except Exception as exc:
            print(f"\n运行失败：{exc}\n")