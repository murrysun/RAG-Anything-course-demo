\# RAG-Anything 小组作业 Demo 部署说明



本项目基于 \[HKUDS/RAG-Anything](https://github.com/HKUDS/RAG-Anything) 进行课程作业演示适配，实现了一个多模态文档 RAG 问答系统 Demo。



本项目主要适配内容包括：



\- 使用 MinerU 解析 PDF / 文档内容

\- 使用 DeepSeek API 作为文本生成模型

\- 使用 Ollama + nomic-embed-text 作为本地 embedding 模型

\- 使用 RAG-Anything / LightRAG 完成图谱构建、向量检索和混合检索

\- 提供命令行 Demo：`demo\_deepseek\_ollama.py`

\- 提供 Streamlit 前端界面：`app\_streamlit.py`



\---



\## 1. 环境要求



推荐环境：



\- Windows 10 / Windows 11

\- Python 3.10

\- Conda / Miniconda

\- Git

\- Ollama

\- DeepSeek API Key

\- NVIDIA GPU 可选，但建议有；如果没有 GPU，MinerU 解析会比较慢



\---



\## 2. 创建 conda 环境



在项目根目录外或任意位置打开 CMD / Anaconda Prompt，执行：



```cmd

conda create -n raganything-demo python=3.10 -y

```



激活环境：



```cmd

conda activate raganything-demo

```



升级基础工具：



```cmd

python -m pip install -U pip setuptools wheel

```



\---



\## 3. 安装 Python 依赖



进入项目目录：



```cmd

cd /d D:\\homework\\RAG-Anything-main

```



安装依赖：



```cmd

pip install -e ".\[all]" openai python-dotenv ollama streamlit numpy

```



如果网络较慢，可以使用清华源：



```cmd

pip install -e ".\[all]" openai python-dotenv ollama streamlit numpy -i https://pypi.tuna.tsinghua.edu.cn/simple

```



\---



\## 4. 安装 Ollama embedding 模型



请先安装 Ollama。



安装完成后，在powershell里拉取 embedding 模型：



```cmd

ollama pull nomic-embed-text

```



测试 Ollama 是否可用：



```cmd

python -c "import ollama; print(ollama.embed(model='nomic-embed-text', input='hello world')\['embeddings']\[0]\[:3])"

```



如果能输出几个数字，说明 Ollama embedding 正常。



注意：不推荐默认使用 `bge-m3`，因为部分环境下它可能出现 NaN 或服务端 500 问题。本 Demo 默认使用 `nomic-embed-text`。



\---



\## 5. 准备 MinerU 模型



MinerU 负责解析 PDF 中的文本、表格、公式、图表、代码块等内容。



推荐使用 MinerU 官方模型下载工具：



```cmd

mineru-models-download -s modelscope -m all

```



如果上面的 `all` 参数不可用，请先查看帮助：



```cmd

mineru-models-download --help

```



然后根据帮助分别下载 pipeline 和 vlm 模型。



下载完成后，检查用户目录下是否存在 MinerU 配置文件：



```text

C:\\Users\\你的用户名\\mineru.json

```



其中应该包含类似内容：



```json

"models-dir": {

&#x20; "pipeline": "...",

&#x20; "vlm": "..."

}

```



如果本地已经手动下载好模型，需要确保 `.env` 里有：



```env

MINERU\_MODEL\_SOURCE=local

```



\---



\## 6. 配置 DeepSeek API Key



复制 `.env.example` 为 `.env`：



```cmd

copy .env.example .env

```



然后编辑 `.env`，填写自己的 DeepSeek API Key。



`.env.example` 示例：



```env

LLM\_BINDING\_API\_KEY=请填写自己的DeepSeek\_API\_Key

LLM\_BINDING\_HOST=https://api.deepseek.com

LLM\_MODEL=deepseek-v4-flash

DEEPSEEK\_THINKING=disabled



OLLAMA\_HOST=http://localhost:11434

OLLAMA\_EMBEDDING\_MODEL=nomic-embed-text

OLLAMA\_EMBED\_CONCURRENCY=1

OLLAMA\_KEEP\_ALIVE=60m



PARSER=mineru

PARSE\_METHOD=auto

MINERU\_MODEL\_SOURCE=local



OUTPUT\_DIR=./output

WORKING\_DIR=./rag\_storage\_demo



ENABLE\_IMAGE\_PROCESSING=false

ENABLE\_TABLE\_PROCESSING=true

ENABLE\_EQUATION\_PROCESSING=true



MINERU\_PROCESSING\_WINDOW\_SIZE=4

MINERU\_HYBRID\_BATCH\_RATIO=1

MINERU\_API\_MAX\_CONCURRENT\_REQUESTS=1

MINERU\_PDF\_RENDER\_THREADS=1



TEMPERATURE=0

MAX\_TOKENS=4096

TIMEOUT=240

```



重要提醒：



\- `.env` 里包含个人 API Key，不能上传 GitHub

\- `.env.example` 只有占位符，可以上传

\- 不要把自己的 DeepSeek API Key 发给别人



\---



\## 7. 命令行运行 Demo



\### 7.1 快速测试前 3 页



建议第一次先只解析前 3 页，确认系统能跑通：



```cmd

python demo\_deepseek\_ollama.py RAG.pdf --start-page 0 --end-page 2 --query "请用三句话概括这篇论文想解决什么问题。"

```



如果成功，终端最后会输出回答。



\### 7.2 完整解析 PDF



完整解析会比较慢，尤其 MinerU 解析阶段可能占用 GPU / CPU。



```cmd

python demo\_deepseek\_ollama.py RAG.pdf --query "请总结 RAG-Anything 的核心贡献。"

```



\### 7.3 基于已有索引直接查询



如果已经完成过解析和建库，后续不要重复解析，可以直接查询：



```cmd

python demo\_deepseek\_ollama.py RAG.pdf --skip-process --query "RAG-Anything 的 dual-graph construction 是什么？"

```



课堂演示时建议优先使用 `--skip-process`，避免现场重新解析 PDF。



\---



\## 8. 启动 Streamlit 前端



本项目提供了一个简单前端界面，可以上传文件、输入 API Key、解析文档、提问并查看运行日志。



启动前端：



```cmd

streamlit run app\_streamlit.py

```



浏览器一般会自动打开：



```text

http://localhost:8501

```



如果没有自动打开，可以手动复制终端中的 Local URL 到浏览器。



前端支持：



\- 上传 PDF / 文档

\- 输入 DeepSeek API Key

\- 解析文档并建立索引

\- 基于已有索引提问

\- 查看实时运行日志

\- 低显存 / 稳定模式

\- 部分页面解析

\- 清理索引但保留解析缓存

\- 彻底清理索引和解析输出



\---



\## 9. 常见问题



\### 9.1 提示找不到 Ollama



如果出现：



```text

Failed to connect to Ollama

```



请确认 Ollama 正在运行。



可以执行：



```cmd

ollama list

```



如果 Ollama 没启动，可以尝试：



```cmd

ollama serve

```



然后重新运行 Demo。



\---



\### 9.2 回答是 no-context



如果出现：



```text

Sorry, I'm not able to provide an answer to that question.\[no-context]

```



通常说明索引为空，或者上一次失败运行污染了 `doc\_status`。



解决方法：



1\. 前端点击“清理索引但保留解析缓存”

2\. 或命令行手动清理 `rag\_storage\_demo`

3\. 然后重新建立索引



注意：如果已经完整解析过 PDF，不建议直接删除 `output`，否则 MinerU 需要重新解析，会很慢。



\---



\### 9.3 MinerU 完整解析很慢



完整 PDF 解析时，MinerU 可能会使用 VLM、OCR、Layout、MFR 等多个模块，耗时较长。



建议在 `.env` 中使用低显存稳定配置：



```env

MINERU\_PROCESSING\_WINDOW\_SIZE=4

MINERU\_HYBRID\_BATCH\_RATIO=1

MINERU\_API\_MAX\_CONCURRENT\_REQUESTS=1

MINERU\_PDF\_RENDER\_THREADS=1

```



如果只做课堂演示，可以先解析前几页：



```cmd

python demo\_deepseek\_ollama.py RAG.pdf --start-page 0 --end-page 5 --query "请总结这篇文档的核心内容。"

```



\---



\### 9.4 显存被残留进程占用



如果关闭终端后 GPU 显存没有释放，可以执行：



```cmd

nvidia-smi

```



找到占用显存的 Python 进程 PID，然后执行：



```cmd

taskkill /F /T /PID 进程号

```



例如：



```cmd

taskkill /F /T /PID 18828

```



\---



\### 9.5 Streamlit 报 DuplicateElementId



如果前端报：



```text

StreamlitDuplicateElementId

```



请确认 `app\_streamlit.py` 中日志窗口和回答窗口没有在循环中反复创建同名 `text\_area`。



如果有修复脚本：



```cmd

python fix\_streamlit\_duplicate\_id.py

```



然后重新启动：



```cmd

streamlit run app\_streamlit.py

```



\---



\## 10. GitHub 提交注意事项



不要上传以下内容：



```text

.env

rag\_storage\_demo/

output/

demo\_uploads/

MinerU 模型文件

\*.safetensors

\*.pth

\*.pt

\*.onnx

\*.bin

\_\_pycache\_\_/

\*.pyc

\*.log

```



推荐 `.gitignore` 中包含：



```gitignore

.env

.env.\*

!.env.example



rag\_storage\_demo/

rag\_storage/

output/

demo\_uploads/

mineru\_test\_output/

mineru\_full\_test/

mineru\_pipeline\_test/



last\_answer.txt

\_parse\_cache\_backup.json



\_\_pycache\_\_/

\*.pyc

\*.log



.venv/

.ipynb\_checkpoints/



tiktoken\_cache/



\*.safetensors

\*.pth

\*.pt

\*.onnx

\*.bin

```



\---



\## 11. 小组汇报建议



本 Demo 可以展示以下流程：



```text

上传 PDF

↓

MinerU 解析文档

↓

RAG-Anything 构建多模态知识单元

↓

DeepSeek 抽取实体关系

↓

Ollama 生成 embedding

↓

构建图谱和向量索引

↓

用户提问

↓

Hybrid retrieval 检索上下文

↓

DeepSeek 生成最终回答

```



汇报时可以强调：



\- RAG-Anything 不是训练新模型，而是构建多模态 RAG 系统

\- 它解决传统 RAG 难以处理图、表、公式、代码块的问题

\- 核心方法包括多模态知识统一、双图构建和跨模态混合检索

\- 我们的 Demo 使用 DeepSeek + Ollama + MinerU 完成了本地部署适配



\---



\## 12. 文件说明



```text

demo\_deepseek\_ollama.py

```



命令行 Demo，负责调用 RAG-Anything、DeepSeek、Ollama 和 MinerU。



```text

app\_streamlit.py

```



Streamlit 前端界面，适合课堂演示。



```text

.env.example

```



环境变量模板，不包含真实 API Key。



```text

README\_RUN.md

```



部署说明文档。



```text

requirements-demo.txt

```



Demo 依赖说明。



```text

environment.yml

```



Conda 环境说明，可选。



\---



\## 13. 免责声明



本项目仅用于课程作业演示。原始项目来自 HKUDS/RAG-Anything，相关版权和许可证请参考原仓库说明。



本 Demo 中使用的 DeepSeek API Key 需要用户自行配置，禁止将个人 API Key 上传到 GitHub 或发送给他人。

