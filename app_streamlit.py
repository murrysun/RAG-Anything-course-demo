# app_streamlit.py
# RAG-Anything 小组作业演示前端
# 放在 RAG-Anything-main 根目录，与 demo_deepseek_ollama.py 同级
#
# 启动方式：
#   conda activate raganything-demo
#   pip install streamlit
#   streamlit run app_streamlit.py

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
DEMO_SCRIPT = PROJECT_DIR / "demo_deepseek_ollama.py"

UPLOAD_DIR = PROJECT_DIR / "demo_uploads"
DEFAULT_WORKING_DIR = PROJECT_DIR / "rag_storage_demo"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output"

UPLOAD_DIR.mkdir(exist_ok=True)


st.set_page_config(
    page_title="RAG-Anything Demo",
    page_icon="📄",
    layout="wide",
)

st.title("📄 RAG-Anything 小组作业 Demo")
st.caption(
    "基于 RAG-Anything + MinerU + DeepSeek API + Ollama Embedding 的多模态文档问答演示系统"
)


def init_state() -> None:
    defaults = {
        "last_uploaded_path": "",
        "last_logs": "",
        "last_answer": "",
        "last_return_code": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def save_uploaded_file(uploaded_file) -> Optional[Path]:
    if uploaded_file is None:
        return None

    safe_name = Path(uploaded_file.name).name
    target = UPLOAD_DIR / safe_name

    with open(target, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.session_state.last_uploaded_path = str(target)
    return target


def extract_answer_from_logs(log_text: str) -> str:
    if not log_text:
        return ""

    pattern = r"回答：\s*\n(?P<answer>.*?)(?:\n={20,}|\Z)"
    matches = list(re.finditer(pattern, log_text, flags=re.DOTALL))
    if not matches:
        return ""

    return matches[-1].group("answer").strip()


def copy_parse_cache_if_exists(src_working_dir: Path, backup_file: Path) -> bool:
    candidates = [
        src_working_dir / "kv_store_parse_cache.json",
        src_working_dir / "parse_cache.json",
    ]

    for p in candidates:
        if p.exists():
            shutil.copy2(p, backup_file)
            return True

    return False


def restore_parse_cache_if_exists(dst_working_dir: Path, backup_file: Path) -> bool:
    if not backup_file.exists():
        return False

    dst_working_dir.mkdir(exist_ok=True)
    shutil.copy2(backup_file, dst_working_dir / "kv_store_parse_cache.json")
    return True


def clear_index_keep_parse_cache(working_dir: Path) -> Tuple[bool, str]:
    backup_file = PROJECT_DIR / "_parse_cache_backup.json"
    if backup_file.exists():
        backup_file.unlink()

    had_cache = False
    if working_dir.exists():
        had_cache = copy_parse_cache_if_exists(working_dir, backup_file)
        shutil.rmtree(working_dir, ignore_errors=True)

    working_dir.mkdir(exist_ok=True)

    restored = False
    if had_cache:
        restored = restore_parse_cache_if_exists(working_dir, backup_file)

    if backup_file.exists():
        backup_file.unlink()

    if restored:
        return True, "已清理索引状态，并保留 / 恢复 parse cache。"
    if had_cache and not restored:
        return False, "尝试保留 parse cache，但恢复失败。"
    return True, "已清理索引状态；未发现可保留的 parse cache。"


def full_cleanup(working_dir: Path, output_dir: Path) -> str:
    if working_dir.exists():
        shutil.rmtree(working_dir, ignore_errors=True)
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    return "已彻底清理 working_dir 和 output。"


def build_child_env(
    api_key: str,
    llm_model: str,
    embedding_model: str,
    ollama_host: str,
    working_dir: Path,
    output_dir: Path,
    mineru_model_source: str,
    low_vram_mode: bool,
    table_processing: bool,
    equation_processing: bool,
    image_processing: bool,
    embed_concurrency: int,
    ollama_keep_alive: str,
) -> Dict[str, str]:
    env = os.environ.copy()

    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    env["LLM_BINDING_API_KEY"] = api_key
    env["LLM_BINDING_HOST"] = "https://api.deepseek.com"
    env["LLM_MODEL"] = llm_model
    env["DEEPSEEK_THINKING"] = "disabled"

    env["OLLAMA_HOST"] = ollama_host
    env["OLLAMA_EMBEDDING_MODEL"] = embedding_model
    env["OLLAMA_EMBED_CONCURRENCY"] = str(embed_concurrency)
    env["OLLAMA_KEEP_ALIVE"] = ollama_keep_alive

    env["PARSER"] = "mineru"
    env["PARSE_METHOD"] = "auto"
    env["MINERU_MODEL_SOURCE"] = mineru_model_source

    env["WORKING_DIR"] = str(working_dir)
    env["OUTPUT_DIR"] = str(output_dir)

    env["ENABLE_IMAGE_PROCESSING"] = str(image_processing).lower()
    env["ENABLE_TABLE_PROCESSING"] = str(table_processing).lower()
    env["ENABLE_EQUATION_PROCESSING"] = str(equation_processing).lower()

    env["TEMPERATURE"] = "0"
    env["MAX_TOKENS"] = "4096"
    env["TIMEOUT"] = "240"

    if low_vram_mode:
        env["MINERU_PROCESSING_WINDOW_SIZE"] = "4"
        env["MINERU_HYBRID_BATCH_RATIO"] = "1"
        env["MINERU_API_MAX_CONCURRENT_REQUESTS"] = "1"
        env["MINERU_PDF_RENDER_THREADS"] = "1"
    else:
        env.setdefault("MINERU_PROCESSING_WINDOW_SIZE", "8")
        env.setdefault("MINERU_HYBRID_BATCH_RATIO", "2")
        env.setdefault("MINERU_API_MAX_CONCURRENT_REQUESTS", "1")
        env.setdefault("MINERU_PDF_RENDER_THREADS", "2")

    return env


def run_command_live(cmd: List[str], env: Dict[str, str]) -> Tuple[int, str, str]:
    st.session_state.last_logs = ""
    st.session_state.last_answer = ""
    st.session_state.last_return_code = None

    log_box = st.empty()
    answer_box = st.empty()

    logs = ""

    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert process.stdout is not None

    for line in process.stdout:
        logs += line

        display_logs = logs[-30000:]
        log_box.code(display_logs, language="text")

        current_answer = extract_answer_from_logs(logs)
        if current_answer:
            answer_box.code(current_answer, language="text")

    return_code = process.wait()
    answer = extract_answer_from_logs(logs)

    st.session_state.last_logs = logs
    st.session_state.last_answer = answer
    st.session_state.last_return_code = return_code

    return return_code, logs, answer


def check_prerequisites() -> List[str]:
    problems = []
    if not DEMO_SCRIPT.exists():
        problems.append(f"没有找到 {DEMO_SCRIPT.name}。请把本前端文件放到 RAG-Anything-main 根目录。")
    return problems


init_state()

problems = check_prerequisites()
if problems:
    for p in problems:
        st.error(p)
    st.stop()


with st.sidebar:
    st.header("⚙️ 配置")

    api_key = st.text_input(
        "DeepSeek API Key",
        type="password",
        placeholder="sk-xxxxxxxx",
        help="只在本次运行中通过环境变量传给子进程，不会写入源码文件。",
    )

    llm_model = st.selectbox(
        "DeepSeek 模型",
        options=["deepseek-v4-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner"],
        index=0,
        help="做课程 demo 推荐 deepseek-v4-flash。",
    )

    st.divider()

    ollama_host = st.text_input("Ollama 地址", value="http://localhost:11434")

    embedding_model = st.selectbox(
        "Ollama Embedding 模型",
        options=["nomic-embed-text", "bge-m3"],
        index=0,
        help="推荐 nomic-embed-text。bge-m3 在部分环境可能出现 NaN 或服务端 500。",
    )

    embed_concurrency = st.slider(
        "Embedding 并发",
        min_value=1,
        max_value=8,
        value=1,
        help="完整文档建议 1，避免多个 worker 同时打爆 Ollama。",
    )

    ollama_keep_alive = st.text_input(
        "Ollama keep_alive",
        value="60m",
        help="建议 60m，避免构建索引时反复加载 / 卸载 embedding 模型。",
    )

    st.divider()

    mineru_model_source = st.selectbox(
        "MinerU 模型来源",
        options=["local", "huggingface", "modelscope"],
        index=0,
        help="已经手动下载模型时选 local。",
    )

    low_vram_mode = st.checkbox(
        "低显存 / 稳定模式",
        value=True,
        help="限制 MinerU window size 和 batch，适合 16GB 显存或 Windows 环境。",
    )

    table_processing = st.checkbox("启用表格处理", value=True)
    equation_processing = st.checkbox("启用公式处理", value=True)
    image_processing = st.checkbox(
        "启用图片处理",
        value=False,
        help="没有 VLM API 时建议关闭；MinerU 仍然会解析图片，但 RAG-Anything 不额外做 image processor。",
    )

    st.divider()

    working_dir_text = st.text_input("Working Dir", value=str(DEFAULT_WORKING_DIR))
    output_dir_text = st.text_input("Output Dir", value=str(DEFAULT_OUTPUT_DIR))

    working_dir = Path(working_dir_text)
    output_dir = Path(output_dir_text)

    st.divider()

    if st.button("🧹 清理索引，保留解析缓存"):
        ok, msg = clear_index_keep_parse_cache(working_dir)
        if ok:
            st.success(msg)
        else:
            st.error(msg)

    if st.button("🧨 彻底清理索引和解析输出"):
        msg = full_cleanup(working_dir, output_dir)
        st.warning(msg)

    st.caption("注意：彻底清理后，下一次会重新跑 MinerU 解析，可能很慢。")


tab_run, tab_help = st.tabs(["🚀 运行 Demo", "📌 使用说明"])


with tab_run:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("1️⃣ 选择 / 上传文档")

        uploaded_file = st.file_uploader(
            "拖拽或选择文档",
            type=["pdf", "docx", "pptx", "xlsx", "png", "jpg", "jpeg"],
        )

        saved_file_path = save_uploaded_file(uploaded_file)

        manual_path = st.text_input(
            "或输入已有文件路径",
            value=st.session_state.last_uploaded_path,
            help="例如 RAG.pdf 或 D:\\homework\\RAG-Anything-main\\RAG.pdf",
        )

        selected_file = Path(manual_path) if manual_path else saved_file_path

        if selected_file:
            st.info(f"当前文件：{selected_file}")

        st.subheader("2️⃣ 解析选项")

        use_page_range = st.checkbox(
            "只解析部分页面",
            value=False,
            help="用于快速演示。例如 start=0, end=2 表示前 3 页。",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            start_page = st.number_input("start-page，0-based", min_value=0, value=0, step=1, disabled=not use_page_range)
        with col_b:
            end_page = st.number_input("end-page，0-based", min_value=0, value=2, step=1, disabled=not use_page_range)

        clean_mode = st.selectbox(
            "运行前清理方式",
            options=[
                "不清理，直接运行",
                "清理索引但保留解析缓存",
                "彻底清理索引和解析输出",
            ],
            index=0,
        )

        parse_query = st.text_area(
            "解析完成后顺便问一个问题",
            value="请用三句话概括这篇文档想解决什么问题。",
            height=100,
        )

        parse_btn = st.button("🚀 解析文档并建立索引", type="primary")

    with right:
        st.subheader("3️⃣ 基于已有索引提问")

        query = st.text_area(
            "问题",
            value="RAG-Anything 的核心贡献是什么？",
            height=140,
        )

        query_btn = st.button("💬 直接查询已有索引")

        st.subheader("4️⃣ 最近一次回答")
        if st.session_state.last_answer:
            st.text_area("回答结果", value=st.session_state.last_answer, height=260)
        else:
            st.info("还没有回答。运行解析或查询后会显示在这里。")

    st.divider()

    env = build_child_env(
        api_key=api_key,
        llm_model=llm_model,
        embedding_model=embedding_model,
        ollama_host=ollama_host,
        working_dir=working_dir,
        output_dir=output_dir,
        mineru_model_source=mineru_model_source,
        low_vram_mode=low_vram_mode,
        table_processing=table_processing,
        equation_processing=equation_processing,
        image_processing=image_processing,
        embed_concurrency=embed_concurrency,
        ollama_keep_alive=ollama_keep_alive,
    )

    if parse_btn:
        if not api_key:
            st.error("请先输入 DeepSeek API Key。")
        elif not selected_file:
            st.error("请先上传文件或输入文件路径。")
        elif not Path(selected_file).exists():
            st.error(f"文件不存在：{selected_file}")
        else:
            if clean_mode == "清理索引但保留解析缓存":
                ok, msg = clear_index_keep_parse_cache(working_dir)
                st.info(msg)
            elif clean_mode == "彻底清理索引和解析输出":
                msg = full_cleanup(working_dir, output_dir)
                st.warning(msg)

            cmd = [
                sys.executable,
                "-u",
                str(DEMO_SCRIPT),
                str(selected_file),
                "--query",
                parse_query,
            ]

            if use_page_range:
                cmd.extend(["--start-page", str(int(start_page)), "--end-page", str(int(end_page))])

            st.code(" ".join(cmd), language="bash")
            code, logs, answer = run_command_live(cmd, env)

            if code == 0:
                if answer and "[no-context]" not in answer:
                    st.success("解析 / 建库 / 查询完成。")
                else:
                    st.warning("命令执行结束，但回答为空或没有检索到上下文，请检查索引是否真的构建成功。")
            else:
                st.error(f"命令执行失败，退出码：{code}")

    if query_btn:
        if not api_key:
            st.error("请先输入 DeepSeek API Key。")
        elif not selected_file:
            st.error("请先上传文件或输入文件路径。")
        elif not Path(selected_file).exists():
            st.error(f"文件不存在：{selected_file}")
        elif not working_dir.exists():
            st.error("没有发现 working_dir。请先解析并建立索引。")
        else:
            cmd = [
                sys.executable,
                "-u",
                str(DEMO_SCRIPT),
                str(selected_file),
                "--skip-process",
                "--query",
                query,
            ]

            st.code(" ".join(cmd), language="bash")
            code, logs, answer = run_command_live(cmd, env)

            if code == 0:
                if answer and "[no-context]" not in answer:
                    st.success("查询完成。")
                else:
                    st.warning("查询结束，但没有找到上下文。可能是索引为空或上一次失败运行污染了状态。")
            else:
                st.error(f"命令执行失败，退出码：{code}")


with tab_help:
    st.subheader("使用流程")

    st.markdown(
        """
1. 确认 `demo_deepseek_ollama.py` 已经能在命令行跑通。
2. 在左侧输入 DeepSeek API Key。
3. 确认 Ollama 正在运行，并且已经拉取 `nomic-embed-text`。
4. 上传 PDF，或输入本地文件路径。
5. 第一次点击“解析文档并建立索引”。
6. 后续演示时点击“直接查询已有索引”，不要重复解析。
        """
    )

    st.subheader("推荐命令行准备")

    st.code(
        "conda activate raganything-demo\n"
        "pip install streamlit\n"
        "ollama pull nomic-embed-text\n"
        "streamlit run app_streamlit.py",
        language="bash",
    )

    st.subheader("常见问题")

    st.markdown(
        """
- 如果回答是 `[no-context]`，通常说明索引是空的，或者上一次失败运行污染了 `doc_status`。请点击“清理索引但保留解析缓存”，再重新建立索引。
- 如果 MinerU 解析完整 PDF 很慢，建议勾选“低显存 / 稳定模式”，或先用部分页面演示。
- 如果提示连不上 Ollama，请确认 `ollama serve` 正在运行，或重新打开 Ollama。
- 如果显存被残留 Python 进程占用，可以用 `nvidia-smi` 找 PID，再用 `taskkill /F /T /PID 进程号` 清理。
- 交作业时不要提交 `.env` 或任何真实 API Key。
        """
    )
