# fix_streamlit_duplicate_id.py
# 用法：把这个文件放到 RAG-Anything-main 根目录，与 app_streamlit.py 同级，然后运行：
#   python fix_streamlit_duplicate_id.py

from pathlib import Path

p = Path("app_streamlit.py")
if not p.exists():
    raise FileNotFoundError("没有找到 app_streamlit.py，请把本脚本放到 RAG-Anything-main 根目录运行。")

s = p.read_text(encoding="utf-8")

# 修复 1：日志窗口不能在循环里反复创建 text_area，改成非交互式 code。
s = s.replace(
    'log_box.text_area("运行日志", value=display_logs, height=460)',
    'log_box.code(display_logs, language="text")',
)

# 修复 2：回答预览也不能在循环里反复创建 text_area，改成非交互式 code。
s = s.replace(
    'answer_box.text_area("回答结果", value=current_answer, height=220)',
    'answer_box.code(current_answer, language="text")',
)

# 保险：如果还有完全相同的动态 text_area 写法，强制报出来。
if 'answer_box.text_area("回答结果", value=current_answer, height=220)' in s:
    raise RuntimeError("仍然发现 answer_box.text_area，请手动检查 app_streamlit.py。")
if 'log_box.text_area("运行日志", value=display_logs, height=460)' in s:
    raise RuntimeError("仍然发现 log_box.text_area，请手动检查 app_streamlit.py。")

p.write_text(s, encoding="utf-8")
print("patched app_streamlit.py successfully")
