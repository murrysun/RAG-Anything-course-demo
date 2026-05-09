import numpy as np
import argparse
import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig

load_dotenv(dotenv_path=".env", override=False)

DEEPSEEK_API_KEY = os.getenv("LLM_BINDING_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
DEEPSEEK_THINKING = os.getenv("DEEPSEEK_THINKING", "disabled").lower()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")

TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
TIMEOUT = float(os.getenv("TIMEOUT", "240"))

deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    timeout=TIMEOUT,
)


async def deepseek_llm_func(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> str:
    """使用 DeepSeek 作为 RAG-Anything 的文本 LLM。"""

    messages: List[Dict[str, Any]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if history_messages:
        messages.extend(history_messages)

    messages.append({"role": "user", "content": prompt})

    request_kwargs: Dict[str, Any] = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": kwargs.get("temperature", TEMPERATURE),
        "max_tokens": kwargs.get("max_tokens", MAX_TOKENS),
    }

    # DeepSeek V4 支持 thinking enabled / disabled。
    # 做课程 demo 时建议 disabled，速度和费用都更友好。
    if DEEPSEEK_THINKING in {"enabled", "disabled"}:
        request_kwargs["extra_body"] = {
            "thinking": {
                "type": DEEPSEEK_THINKING,
            }
        }

    # 透传少量常见参数，避免 LightRAG 调用时丢配置。
    for key in ["top_p", "stop", "response_format"]:
        if key in kwargs and kwargs[key] is not None:
            request_kwargs[key] = kwargs[key]

    response = await deepseek_client.chat.completions.create(**request_kwargs)
    content = response.choices[0].message.content
    return content or ""


async def ollama_embedding_func(texts: List[str]) -> np.ndarray:
    """使用 Ollama 本地 embedding 模型，并返回 LightRAG 需要的 numpy.ndarray。"""
    import ollama

    # 限制 Ollama embedding 并发，避免 8 个 worker 同时冲 Ollama
    if not hasattr(ollama_embedding_func, "_sem"):
        ollama_embedding_func._sem = asyncio.Semaphore(
            int(os.getenv("OLLAMA_EMBED_CONCURRENCY", "1"))
        )

    async with ollama_embedding_func._sem:
        last_error = None

        for attempt in range(5):
            try:
                client = ollama.AsyncClient(host=OLLAMA_HOST)
                response = await client.embed(
                    model=OLLAMA_EMBEDDING_MODEL,
                    input=texts,
                    keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "60m"),
                )

                if hasattr(response, "embeddings"):
                    embeddings = response.embeddings
                elif isinstance(response, dict) and "embeddings" in response:
                    embeddings = response["embeddings"]
                else:
                    raise RuntimeError(f"无法解析 Ollama embedding 返回结果: {response}")

                arr = np.asarray(embeddings, dtype=np.float32)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                return arr

            except Exception as e:
                last_error = e
                await asyncio.sleep(2 * (attempt + 1))

        raise RuntimeError(f"Ollama embedding 连续失败，最后错误: {last_error}")


async def test_deepseek() -> None:
    """简单测试 DeepSeek API 是否可用。"""
    print("正在测试 DeepSeek API...")
    result = await deepseek_llm_func("只回复 OK。")
    print(f"DeepSeek 返回：{result.strip()[:100]}")


async def test_ollama_embedding() -> int:
    """测试 Ollama embedding，并返回向量维度。"""
    print("正在测试 Ollama embedding...")
    vectors = await ollama_embedding_func(["hello world"])
    embedding_dim = len(vectors[0])
    print(f"Ollama embedding OK，维度 = {embedding_dim}")
    return embedding_dim


async def build_rag(embedding_dim: int) -> RAGAnything:
    """初始化 RAG-Anything。"""
    embedding = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        func=ollama_embedding_func,
    )

    config = RAGAnythingConfig(
        working_dir=os.getenv("WORKING_DIR", "./rag_storage_demo"),
        parser=os.getenv("PARSER", "mineru"),
        parse_method=os.getenv("PARSE_METHOD", "auto"),
        parser_output_dir=os.getenv("OUTPUT_DIR", "./output"),
        enable_image_processing=os.getenv("ENABLE_IMAGE_PROCESSING", "false").lower() == "true",
        enable_table_processing=os.getenv("ENABLE_TABLE_PROCESSING", "true").lower() == "true",
        enable_equation_processing=os.getenv("ENABLE_EQUATION_PROCESSING", "true").lower() == "true",
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=deepseek_llm_func,
        embedding_func=embedding,
    )

    return rag


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="要处理的 PDF / 文档路径")
    parser.add_argument(
        "--query",
        default="请总结这篇文档的核心贡献，并说明它和传统文本 RAG 的区别。",
        help="查询问题",
    )
    parser.add_argument(
        "--skip-process",
        action="store_true",
        help="跳过文档解析，直接使用已有 rag_storage_demo 索引查询",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=None,
        help="只解析起始页，0-based；用于快速测试",
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=None,
        help="只解析结束页，0-based；用于快速测试",
    )
    args = parser.parse_args()

    if not DEEPSEEK_API_KEY:
        raise RuntimeError("没有读取到 DeepSeek API Key，请检查 .env 里的 LLM_BINDING_API_KEY。")

    file_path = Path(args.file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"找不到文件: {file_path}")

    print("=" * 80)
    print("RAG-Anything Demo: DeepSeek + Ollama embedding")
    print("=" * 80)
    print(f"文档路径: {file_path}")
    print(f"DeepSeek 模型: {DEEPSEEK_MODEL}")
    print(f"DeepSeek thinking: {DEEPSEEK_THINKING}")
    print(f"Ollama host: {OLLAMA_HOST}")
    print(f"Ollama embedding: {OLLAMA_EMBEDDING_MODEL}")
    print(f"Working dir: {os.getenv('WORKING_DIR', './rag_storage_demo')}")
    print(f"Output dir: {os.getenv('OUTPUT_DIR', './output')}")
    print("=" * 80)

    await test_deepseek()
    embedding_dim = await test_ollama_embedding()
    rag = await build_rag(embedding_dim)

    if not args.skip_process:
        print("\n开始解析文档并构建知识库。第一次运行会比较慢，请耐心等它跑完。")

        parser_kwargs: Dict[str, Any] = {}
        if args.start_page is not None:
            parser_kwargs["start_page"] = args.start_page
        if args.end_page is not None:
            parser_kwargs["end_page"] = args.end_page

        await rag.process_document_complete(
            file_path=str(file_path),
            output_dir=os.getenv("OUTPUT_DIR", "./output"),
            parse_method=os.getenv("PARSE_METHOD", "auto"),
            display_stats=True,
            **parser_kwargs,
        )
    else:
        print("\n跳过文档解析，尝试加载已有知识库。")
        init_result = await rag._ensure_lightrag_initialized()
        if not init_result or not init_result.get("success"):
            raise RuntimeError(f"加载已有知识库失败: {init_result}")

    print("\n开始查询...")
    result = await rag.aquery(args.query, mode="hybrid", vlm_enhanced=False)

    print("\n" + "=" * 80)
    print("问题：")
    print(args.query)
    print("\n回答：")
    print(result)
    print("=" * 80)

    await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())