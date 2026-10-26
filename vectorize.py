import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


# 固定项目目录：与本脚本放在同一目录
BASE_DIR = Path(__file__).resolve().parent
CHUNKS_FILE = BASE_DIR / "output" /"chunks.jsonl"
PERSIST_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "docx_knowledge"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
BATCH_SIZE = 32


def load_chunks():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"找不到分块文件：{CHUNKS_FILE}"
        )

    documents = []
    ids = []
    seen_ids = set()

    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            content = record.get("content", "").strip()

            if not content:
                continue

            metadata = record.get("metadata") or {}

            # Chroma metadata 仅接受基本标量类型
            safe_metadata = {}
            for key, value in metadata.items():
                if value is None:
                    continue
                if isinstance(value, (str, int, float, bool)):
                    safe_metadata[key] = value
                else:
                    safe_metadata[key] = json.dumps(
                        value, ensure_ascii=False
                    )

            chunk_id = str(
                record.get("id") or f"chunk-{line_no}"
            )

            if chunk_id in seen_ids:
                raise ValueError(
                    f"发现重复 chunk ID：{chunk_id}，"
                    f"请检查第 {line_no} 行"
                )
            seen_ids.add(chunk_id)

            safe_metadata["chunk_id"] = chunk_id

            documents.append(
                Document(
                    page_content=content,
                    metadata=safe_metadata,
                )
            )
            ids.append(chunk_id)

    if not documents:
        raise ValueError("chunks.jsonl 中没有有效内容")

    return documents, ids


def main():
    documents, ids = load_chunks()

    print(f"读取到 {len(documents)} 个 chunk")
    print(f"向量模型：{EMBEDDING_MODEL}")
    print(f"数据库目录：{PERSIST_DIR}")

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    # 如果数据库已经存在，则更新相同 ID 的文档，
    # 并保留不在本次文件中的旧记录。
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
    )

    for start in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[start:start + BATCH_SIZE]
        batch_ids = ids[start:start + BATCH_SIZE]

        vectorstore.add_documents(
            documents=batch_docs,
            ids=batch_ids,
        )

        print(
            f"处理进度："
            f"{min(start + BATCH_SIZE, len(documents))}"
            f"/{len(documents)}"
        )

    print("向量化完成，数据库已保存。")


if __name__ == "__main__":
    main()
