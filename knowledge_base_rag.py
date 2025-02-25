import os
from pathlib import Path

import pytesseract
import torch
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from pdf2image import convert_from_path
from PIL import Image


class DeepSeekKnowledgeBase:
    def __init__(
        self,
        model_name="BAAI/bge-large-zh",
        llm_path="deepseek-ai/deepseek-llm-7b-chat",
        device="cuda" if torch.cuda.is_available() else "cpu",
        chunk_size=1000,
        chunk_overlap=100,
    ):
        self.device = device
        self.embedding_model_path = model_name
        self.llm_path = llm_path
        self.documents = []
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # 初始化Embedding模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_path, model_kwargs={"device": self.device}
        )

        # 初始化DeepSeek LLM
        self.llm = HuggingFaceEndpoint(
            repo_id=self.llm_path, task="text-generation", model_kwargs={"temperature": 0.7, "max_length": 2048}
        )

        # OCR设置
        self.ocr_tool = pytesseract

    def process_text_file(self, file_path):
        """处理文本文件"""
        loader = TextLoader(file_path)
        documents = loader.load()
        return documents

    def process_pdf_file(self, file_path):
        """处理PDF文件，包括文本和图像"""
        pdf_documents = []

        # 提取PDF文本
        loader = PyPDFLoader(file_path)
        text_documents = loader.load()
        pdf_documents.extend(text_documents)

        # 提取PDF中的图像并进行OCR
        images = convert_from_path(file_path)
        for i, img in enumerate(images):
            # 进行OCR识别
            text = self.ocr_tool.image_to_string(img)
            if text.strip():  # 如果提取到了文本
                pdf_documents.append({"page_content": text, "metadata": {"source": f"{file_path}:image_{i}"}})

        return pdf_documents

    def process_image_file(self, file_path):
        """处理图像文件"""
        try:
            img = Image.open(file_path)
            # 图像预处理：调整大小、二值化、去噪
            img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)  # 提高分辨率
            img = img.convert("L")  # 转换为灰度图像
            img = img.point(lambda x: 0 if x < 128 else 255, "1")  # 二值化

            # 基本OCR处理
            text = self.ocr_tool.image_to_string(img)

            # 对于复杂图像，可以调用DeepSeek-VL2进行分析
            # 这里是一个示例调用，实际实现需要根据DeepSeek-VL2的API调整
            # vl2_description = self.analyze_image_with_deepseek_vl2(file_path)

            # 合并OCR结果和图像分析结果
            # combined_text = text + "\n" + vl2_description

            return [{"page_content": text, "metadata": {"source": file_path}}]
        except Exception as e:
            print(f"处理图像出错: {e}")
            return []

    # 以下是DeepSeek-VL2的模拟实现，实际使用需要根据其API调整
    def analyze_image_with_deepseek_vl2(self, image_path):
        """使用DeepSeek-VL2分析图像"""
        # 这里是一个模拟实现，实际使用需要根据DeepSeek-VL2的实际API调整

        # 加载DeepSeek-VL2模型
        # 注意：以下代码仅为示例，需要根据实际API调整
        """
        from transformers import AutoProcessor, AutoModelForVision2Text

        processor = AutoProcessor.from_pretrained("deepseek-ai/deepseek-vl2")
        model = AutoModelForVision2Text.from_pretrained("deepseek-ai/deepseek-vl2").to(self.device)

        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt").to(self.device)

        outputs = model.generate(
            **inputs,
            max_length=50,
        )

        description = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return description
        """

        # 由于模型可能不直接可用，这里返回一个占位符
        return "This is a placeholder for DeepSeek-VL2 image analysis results."

    def add_document(self, file_path):
        """添加文档到知识库"""
        file_ext = Path(file_path).suffix.lower()

        if file_ext in [".txt", ".md", ".csv"]:
            docs = self.process_text_file(file_path)
        elif file_ext == ".pdf":
            docs = self.process_pdf_file(file_path)
        elif file_ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            docs = self.process_image_file(file_path)
        else:
            print(f"不支持的文件类型: {file_ext}")
            return

        # 分割文档
        if docs:
            split_docs = self.text_splitter.split_documents(docs)
            self.documents.extend(split_docs)
            print(f"成功添加文档: {file_path}，共{len(split_docs)}个片段")

    def build_knowledge_base(self, save_path="./knowledge_base"):
        """构建向量数据库"""
        if not self.documents:
            print("没有文档可以构建知识库")
            return None

        print(f"正在为{len(self.documents)}个文档片段创建向量索引...")
        vectorstore = FAISS.from_documents(self.documents, self.embeddings)

        # 保存向量库到磁盘
        os.makedirs(save_path, exist_ok=True)
        vectorstore.save_local(save_path)

        print(f"知识库已保存到: {save_path}")
        return vectorstore

    def load_knowledge_base(self, load_path="./knowledge_base"):
        """加载已有知识库"""
        if not os.path.exists(load_path):
            print(f"知识库路径不存在: {load_path}")
            return None

        vectorstore = FAISS.load_local(load_path, self.embeddings)
        print(f"已加载知识库，包含{vectorstore.index.ntotal}个向量")
        return vectorstore

    def query_knowledge_base(self, query, vectorstore=None, k=3):
        """查询知识库"""
        if vectorstore is None:
            vectorstore = self.load_knowledge_base()
            if vectorstore is None:
                return "知识库未找到，请先构建知识库"

        # 创建检索QA链
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff", retriever=retriever, return_source_documents=True
        )

        # 执行查询
        result = qa_chain({"query": query})

        return {"answer": result["result"], "sources": [doc.metadata["source"] for doc in result["source_documents"]]}


class DeepSeekAdvancedRAG:
    def __init__(self, model_path="deepseek-ai/deepseek-llm-7b-chat", embedding_model="BAAI/bge-large-zh"):
        # 初始化基础组件
        self.kb = DeepSeekKnowledgeBase(llm_path=model_path, embedding_model=embedding_model)

        # 初始化各种增强组件
        self.query_expander = QueryExpander(self.kb.llm)
        self.context_compressor = ContextCompressor(self.kb.llm)
        self.adaptive_rag = AdaptiveRAG(self.kb)

        # 初始化可选组件 (需要时才加载)
        self.reranker = None
        self.hierarchical_retriever = None
        self.multimodal_processor = None

    def add_document(self, file_path):
        """添加文档到知识库"""
        self.kb.add_document(file_path)

    def build_knowledge_base(self, save_path="./advanced_kb"):
        """构建并保存知识库"""
        vectorstore = self.kb.build_knowledge_base(save_path)

        # 如果需要层次化索引，可以在这里构建
        if len(self.kb.documents) > 100:  # 当文档数量较大时使用层次化索引
            print("构建层次化索引...")
            self.hierarchical_retriever = HierarchicalRetrieval(self.kb.embeddings)
            self.hierarchical_retriever.index_documents(self.kb.documents)

        return vectorstore

    def load_knowledge_base(self, load_path="./advanced_kb"):
        """加载知识库"""
        return self.kb.load_knowledge_base(load_path)

    def query(self, query_text, vectorstore=None, use_advanced=True, image_path=None):
        """高级查询接口"""
        if vectorstore is None:
            vectorstore = self.load_knowledge_base()
            if vectorstore is None:
                return "知识库未找到，请先构建知识库"

        # 根据查询复杂度和知识库大小选择策略
        if not use_advanced:
            # 使用基本RAG
            return self.kb.query_knowledge_base(query_text, vectorstore)

        # 使用图像内容增强查询
        if image_path:
            if self.multimodal_processor is None:
                self.multimodal_processor = MultimodalRAG(self.kb.embeddings, self.kb)
            return self.multimodal_processor.query_with_image(query_text, image_path)

        # 使用查询扩展
        expanded_queries = self.query_expander.expand_query(query_text)
        print(f"扩展查询: {expanded_queries}")

        # 文档检索 - 选择合适的检索方法
        if self.hierarchical_retriever and len(self.kb.documents) > 100:
            # 使用层次化检索
            retrieved_docs = self.hierarchical_retriever.hierarchical_search(query_text, top_k_docs=5, top_k_passages=3)
        else:
            # 使用基本检索 + 查询扩展
            all_docs = []
            for q in expanded_queries:
                docs = vectorstore.similarity_search(q, k=3)
                all_docs.extend(docs)

            # 去重
            retrieved_docs = []
            seen_contents = set()
            for doc in all_docs:
                if doc.page_content not in seen_contents:
                    retrieved_docs.append(doc)
                    seen_contents.add(doc.page_content)

            # 限制使用的文档数量
            retrieved_docs = retrieved_docs[:7]

        # 使用上下文压缩
        compressed_context = self.context_compressor.compress_context(query_text, retrieved_docs, max_len=4000)

        # 修复：使用压缩后的上下文构建自定义查询
        custom_prompt = f"""
        <context>
        {compressed_context}
        </context>

        基于以上信息，请回答以下问题: {query_text}

        如果上下文中没有足够信息，请明确指出。
        回答:
        """

        # 直接使用压缩后的上下文生成答案
        response = self.kb.llm.invoke(custom_prompt)

        return response.content

    def enable_reranker(self, reranker_model="BAAI/bge-reranker-large"):
        """启用重排序器以提高检索质量"""
        if self.reranker is None:
            print(f"加载重排序模型: {reranker_model}")
            self.reranker = DocumentReranker(reranker_model)
        return self.reranker

    def query_with_reranking(self, query_text, vectorstore=None, top_k=10, rerank_top_k=5):
        """使用重排序增强检索质量的查询接口"""
        if self.reranker is None:
            self.enable_reranker()

        if vectorstore is None:
            vectorstore = self.load_knowledge_base()
            if vectorstore is None:
                return "知识库未找到，请先构建知识库"

        # 初始检索
        initial_docs = vectorstore.similarity_search(query_text, k=top_k)

        # 重排序
        reranked_docs = self.reranker.rerank_documents(query_text, initial_docs, top_k=rerank_top_k)

        # 生成答案
        compressed_context = self.context_compressor.compress_context(query_text, reranked_docs, max_len=4000)

        custom_prompt = f"""
        <context>
        {compressed_context}
        </context>

        基于以上信息，请回答以下问题: {query_text}

        如果上下文中没有足够信息，请明确指出。
        回答:
        """

        response = self.kb.llm.invoke(custom_prompt)
        return response.content

    def adaptive_query(self, query_text, vectorstore=None):
        """自适应查询 - 根据查询类型和知识库情况自动选择最佳策略"""
        return self.adaptive_rag.query(query_text, vectorstore)

    def batch_add_documents(self, file_paths):
        """批量添加文档到知识库"""
        for file_path in file_paths:
            self.add_document(file_path)
        print(f"已添加 {len(file_paths)} 个文档到知识库")

    def evaluate_retrieval(self, test_queries, ground_truth, vectorstore=None):
        """评估检索系统性能"""
        if vectorstore is None:
            vectorstore = self.load_knowledge_base()
            if vectorstore is None:
                return "知识库未找到，请先构建知识库"

        metrics = {
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "mrr": 0,  # Mean Reciprocal Rank
        }

        total_queries = len(test_queries)
        for i, query in enumerate(test_queries):
            print(f"评估查询 {i + 1}/{total_queries}: {query}")
            retrieved_docs = vectorstore.similarity_search(query, k=5)

            # 计算检索指标
            # 获取检索到的文档ID或内容，取决于ground_truth的格式
            if isinstance(ground_truth[0], list) and all(isinstance(item, str) for item in ground_truth[0]):
                # 如果ground_truth是文档ID列表
                retrieved_items = [getattr(doc, "metadata", {}).get("source", "") for doc in retrieved_docs]
            else:
                # 如果ground_truth是文档内容列表或其他格式
                retrieved_items = [doc.page_content for doc in retrieved_docs]

            # 获取当前查询的真实相关文档
            current_ground_truth = ground_truth[i] if i < len(ground_truth) else []

            # 计算精确率 (Precision) - 检索结果中相关文档的比例
            relevant_retrieved = sum(1 for item in retrieved_items if item in current_ground_truth)
            precision = relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0

            # 计算召回率 (Recall) - 相关文档中被检索到的比例
            recall = relevant_retrieved / len(current_ground_truth) if current_ground_truth else 0

            # 计算F1分数
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            # 计算MRR (Mean Reciprocal Rank)
            mrr = 0
            for rank, item in enumerate(retrieved_items, 1):
                if item in current_ground_truth:
                    mrr = 1 / rank
                    break

            # 累加指标
            metrics["precision"] += precision
            metrics["recall"] += recall
            metrics["f1_score"] += f1
            metrics["mrr"] += mrr

        # 计算平均值
        for key in metrics:
            metrics[key] /= total_queries if total_queries > 0 else 1

        return metrics

    def save(self, save_dir="./advanced_rag_system"):
        """保存整个RAG系统状态"""
        import json
        import os
        import pickle

        os.makedirs(save_dir, exist_ok=True)

        # 保存配置信息
        config = {
            "model_path": self.kb.llm_path,
            "embedding_model": self.kb.embedding_model,
            "documents_count": len(self.kb.documents) if hasattr(self.kb, "documents") else 0,
            "has_hierarchical_index": self.hierarchical_retriever is not None,
            "has_reranker": self.reranker is not None,
            "has_multimodal": self.multimodal_processor is not None,
        }

        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # 保存知识库
        self.kb.build_knowledge_base(os.path.join(save_dir, "kb"))

        # 保存层次化索引(如果有)
        if self.hierarchical_retriever:
            with open(os.path.join(save_dir, "hierarchical_index.pkl"), "wb") as f:
                pickle.dump(self.hierarchical_retriever, f)

        print(f"RAG系统已保存到: {save_dir}")

    @classmethod
    def load(cls, load_dir="./advanced_rag_system"):
        """加载RAG系统"""
        import json
        import os
        import pickle

        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"保存目录不存在: {load_dir}")

        # 加载配置
        with open(os.path.join(load_dir, "config.json"), "r") as f:
            config = json.load(f)

        # 创建实例
        rag_system = cls(model_path=config["model_path"], embedding_model=config["embedding_model"])

        # 加载知识库
        rag_system.load_knowledge_base(os.path.join(load_dir, "kb"))

        # 加载层次化索引(如果有)
        if config["has_hierarchical_index"]:
            hier_index_path = os.path.join(load_dir, "hierarchical_index.pkl")
            if os.path.exists(hier_index_path):
                with open(hier_index_path, "rb") as f:
                    rag_system.hierarchical_retriever = pickle.load(f)

        # 加载重排序器(如果需要)
        if config["has_reranker"]:
            rag_system.enable_reranker()

        print(f"成功加载RAG系统，包含 {config['documents_count']} 个文档")
        return rag_system


# 辅助组件类


class QueryExpander:
    """查询扩展器 - 将单一查询扩展为多个相关查询以提高召回率"""

    def __init__(self, llm):
        self.llm = llm

    def expand_query(self, query, num_expansions=3):
        """扩展查询为多个相关查询"""
        prompt = f"""
        原始查询: "{query}"

        请帮我生成 {num_expansions} 个不同的但相关的查询，以便更全面地检索相关信息。
        这些查询应该涵盖不同的角度、使用不同的关键词，但都应与原始查询的意图相关。

        生成的查询:
        """

        response = self.llm.invoke(prompt)
        # 解析响应，提取扩展查询
        expanded_queries = []
        for line in response.content.strip().split("\n"):
            if line and not line.isspace():
                # 移除数字编号或其他前缀
                clean_line = line
                for prefix in ["-", "•", "*", "1.", "2.", "3.", "4.", "5."]:
                    if clean_line.startswith(prefix):
                        clean_line = clean_line[len(prefix) :].strip()
                if clean_line and not clean_line.isspace():
                    expanded_queries.append(clean_line)

        # 始终在列表首位包含原始查询
        if query not in expanded_queries:
            expanded_queries.insert(0, query)

        # 限制扩展查询数量
        return expanded_queries[: num_expansions + 1]


class ContextCompressor:
    """上下文压缩器 - 压缩检索到的文档以适应模型上下文窗口"""

    def __init__(self, llm):
        self.llm = llm

    def compress_context(self, query, documents, max_len=4000):
        """压缩文档以适应上下文窗口"""
        if not documents:
            return ""

        # 先计算当前上下文总长度
        current_context = "\n\n".join([doc.page_content for doc in documents])
        if len(current_context) <= max_len:
            return current_context

        # 需要压缩
        prompt = f"""
        查询: "{query}"

        我需要你帮我将以下检索到的文档内容压缩，使其总长度不超过 {max_len} 字符，同时保留与查询最相关的信息。

        检索到的文档:
        {current_context}

        请提供压缩后的内容，保留对回答查询最重要的信息:
        """

        response = self.llm.invoke(prompt)
        return response.content


class AdaptiveRAG:
    """自适应RAG - 根据查询类型自动选择最佳检索和生成策略"""

    def __init__(self, kb):
        self.kb = kb
        self.llm = kb.llm

    def analyze_query(self, query):
        """分析查询类型和复杂度"""
        prompt = f"""
        请分析以下查询的类型和复杂度:
        "{query}"

        回答以下问题:
        1. 查询类型是什么? (事实性、概念性、推理性、创意性、观点性)
        2. 查询复杂度如何? (简单、中等、复杂)
        3. 是否需要多个知识领域的信息? (是/否)
        4. 是否需要时间序列或最新信息? (是/否)

        以JSON格式回答:
        """

        response = self.llm.invoke(prompt)
        # 解析JSON响应
        import json

        try:
            analysis = json.loads(response.content)
        except Exception:
            # 简单解析作为后备
            analysis = {
                "type": "factual" if "什么" in query or "如何" in query else "conceptual",
                "complexity": "medium",
                "multi_domain": "是" in response.content,
                "temporal": "是" in response.content,
            }

        return analysis

    def query(self, query_text, vectorstore=None):
        """自适应查询处理"""
        if vectorstore is None:
            vectorstore = self.kb.load_knowledge_base()
            if vectorstore is None:
                return "知识库未找到，请先构建知识库"

        # 分析查询
        analysis = self.analyze_query(query_text)
        print(f"查询分析: {analysis}")

        # 根据分析结果选择策略
        if analysis.get("complexity") == "simple" and analysis.get("type") == "factual":
            # 简单事实性查询 - 使用基本检索
            docs = vectorstore.similarity_search(query_text, k=3)

        elif analysis.get("multi_domain", False):
            # 跨领域查询 - 使用查询扩展
            expander = QueryExpander(self.llm)
            expanded_queries = expander.expand_query(query_text)

            all_docs = []
            for q in expanded_queries:
                docs = vectorstore.similarity_search(q, k=2)
                all_docs.extend(docs)

            # 去重
            docs = []
            seen_contents = set()
            for doc in all_docs:
                if doc.page_content not in seen_contents:
                    docs.append(doc)
                    seen_contents.add(doc.page_content)

            # 限制数量
            docs = docs[:5]

        elif analysis.get("complexity") == "complex":
            # 复杂查询 - 增加检索数量
            docs = vectorstore.similarity_search(query_text, k=7)

        else:
            # 默认策略
            docs = vectorstore.similarity_search(query_text, k=4)

        # 压缩上下文
        compressor = ContextCompressor(self.llm)
        compressed_context = compressor.compress_context(query_text, docs)

        # 生成答案
        custom_prompt = f"""
        <context>
        {compressed_context}
        </context>

        基于以上信息，请回答以下问题: {query_text}

        如果上下文中没有足够信息，请明确指出。
        回答:
        """

        response = self.llm.invoke(custom_prompt)
        return response.content


class HierarchicalRetrieval:
    """层次化检索 - 适用于大规模文档集合"""

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.document_index = None  # 文档级别索引
        self.passage_indices = {}  # 每个文档的段落级别索引

    def index_documents(self, documents):
        """构建层次化索引"""
        from langchain.vectorstores import FAISS

        # 构建文档级索引
        document_texts = []
        self.document_mapping = {}

        for i, doc in enumerate(documents):
            # 提取文档标题或使用文档ID作为键值映射
            self.document_mapping[i] = doc
            # 使用文档开头作为表示
            doc_text = doc.page_content[:200]
            document_texts.append(doc_text)

        # 创建文档级向量存储
        self.document_index = FAISS.from_texts(document_texts, self.embedding_model)

        # 为每个文档创建段落级索引
        for i, doc in enumerate(documents):
            # 划分文档为段落
            paragraphs = self._split_into_paragraphs(doc.page_content)

            if paragraphs:
                # 创建段落级向量存储
                self.passage_indices[i] = FAISS.from_texts(paragraphs, self.embedding_model)

        print(
            f"已构建层次化索引: {len(documents)} 文档, {
                sum(len(self.passage_indices.get(i, [])) for i in self.passage_indices)
            } 段落"
        )

    def _split_into_paragraphs(self, text, min_length=50):
        """将文本分割为段落"""
        import re

        # 使用换行符或多个换行符分割
        splits = re.split(r"\n+", text)

        # 过滤太短的段落
        paragraphs = [s.strip() for s in splits if len(s.strip()) >= min_length]

        # 如果没有足够长的段落，就返回原文本
        if not paragraphs:
            return [text]

        return paragraphs

    def hierarchical_search(self, query, top_k_docs=3, top_k_passages=2):
        """层次化搜索"""
        if not self.document_index:
            raise ValueError("索引尚未构建，请先调用index_documents")

        # 第一步：检索最相关的文档
        doc_results = self.document_index.similarity_search_with_score(query, k=top_k_docs)

        # 第二步：从每个相关文档中检索最相关的段落
        all_passages = []

        for doc_idx, (doc, score) in enumerate(doc_results):
            doc_id = doc_idx

            # 如果该文档有段落索引
            if doc_id in self.passage_indices:
                # 从该文档中检索段落
                passage_results = self.passage_indices[doc_id].similarity_search(query, k=top_k_passages)

                # 将原始文档元数据添加到段落中
                for passage in passage_results:
                    if hasattr(doc, "metadata"):
                        if not hasattr(passage, "metadata"):
                            passage.metadata = {}
                        passage.metadata.update(doc.metadata)

                all_passages.extend(passage_results)
            else:
                # 如果没有段落索引，使用整个文档
                all_passages.append(self.document_mapping[doc_id])

        return all_passages


class DocumentReranker:
    """文档重排序器 - 使用语义相似度重新排序检索到的文档"""

    def __init__(self, model_name="BAAI/bge-reranker-large"):
        from sentence_transformers import CrossEncoder

        print(f"加载重排序模型: {model_name}")
        self.reranker = CrossEncoder(model_name)

    def rerank_documents(self, query, documents, top_k=None):
        """重新排序文档列表"""
        if not documents:
            return []

        # 准备重排序的文档对
        document_pairs = [[query, doc.page_content] for doc in documents]

        # 计算分数
        scores = self.reranker.predict(document_pairs, show_progress_bar=False)
        # 检查预测结果是否包含空值
        if any(score is None for score in scores):
            scores = [0 if s is None else s for s in scores]

        # 结合分数和文档，按分数排序
        scored_documents = list(zip(documents, scores))
        scored_documents.sort(key=lambda x: x[1], reverse=True)

        # 提取排序后的文档
        if top_k:
            reranked_docs = [doc for doc, score in scored_documents[:top_k]]
        else:
            reranked_docs = [doc for doc, score in scored_documents]

        return reranked_docs


class MultimodalRAG:
    """多模态RAG - 支持图像和文本混合查询"""

    def __init__(self, text_embeddings, kb, vision_model="openai/clip-vit-base-patch32"):
        self.text_embeddings = text_embeddings
        self.kb = kb

        # 延迟加载视觉模型
        self.vision_model = None
        self.vision_model_name = vision_model

    def _load_vision_model(self):
        """按需加载视觉模型"""
        if self.vision_model is None:
            from transformers import CLIPModel, CLIPProcessor

            print(f"加载视觉模型: {self.vision_model_name}")
            self.clip_model = CLIPModel.from_pretrained(self.vision_model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(self.vision_model_name)

    def _extract_image_features(self, image_path):
        """提取图像特征"""
        import torch
        from PIL import Image

        self._load_vision_model()

        # 加载图像
        image = Image.open(image_path).convert("RGB")

        # 处理图像
        inputs = self.clip_processor(text=[""], images=image, return_tensors="pt", padding=True)

        # 提取特征
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            image_features = outputs.image_embeds

        return image_features

    def _describe_image(self, image_path):
        """使用LLM描述图像内容"""
        # 注意：这里需要实现对接实际的多模态LLM来描述图像
        # 为简化实现，这里使用一个占位符

        # 实际实现应该调用多模态LLM
        image_description = "这里是图像的描述，实际实现需要调用多模态模型"

        return image_description

    def query_with_image(self, query_text, image_path):
        """使用图像和文本进行混合查询"""
        # 获取图像描述
        image_description = self._describe_image(image_path)

        # 结合查询文本和图像描述
        enhanced_query = f"{query_text} {image_description}"

        # 使用增强查询进行检索
        vectorstore = self.kb.load_knowledge_base()
        if vectorstore is None:
            return "知识库未找到，请先构建知识库"

        # 检索相关文档
        docs = vectorstore.similarity_search(enhanced_query, k=5)

        # 构建多模态上下文
        prompt = f"""
        我需要你回答一个涉及图像的问题。

        图像描述: {image_description}

        用户问题: {query_text}

        以下是从知识库中检索到的相关信息:

        {self._format_documents(docs)}

        请基于图像描述和检索到的信息回答用户问题。如果无法从提供的信息中找到答案，请明确说明。
        回答:
        """

        response = self.kb.llm.invoke(prompt)
        return response.content

    def _format_documents(self, docs):
        """格式化文档列表"""
        formatted = []
        for i, doc in enumerate(docs):
            formatted.append(f"文档 {i + 1}:\n{doc.page_content}\n")
        return "\n".join(formatted)


# 使用示例
def main():
    # 初始化高级RAG系统
    rag_system = DeepSeekAdvancedRAG()

    # 添加文档
    rag_system.add_document("path/to/document.pdf")
    rag_system.add_document("path/to/image.jpg")

    # 构建知识库
    vectorstore = rag_system.build_knowledge_base()
    if vectorstore is None:
        print("知识库构建失败，请检查文档路径或模型配置")
        return

    # 查询示例
    query = "DeepSeek模型的特点是什么?"
    result = rag_system.query(query, vectorstore)
    print(f"回答: {result.content}")

    # 带图像的查询示例
    image_path = "path/to/chart.png"
    try:
        image_result = rag_system.query("图表中的趋势是什么?", image_path=image_path)
        print(f"图像分析回答: {image_result.content}")
    except Exception as e:
        print(f"图像查询失败: {e}")


if __name__ == "__main__":
    main()
