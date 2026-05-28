import sys
from typing import List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END

from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from logger import GLOBAL_LOGGER as log


class RAGState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    rewrite_count: int


class AgenticRAG:
    """
    LangGraph-based agentic RAG.

    Flow: retrieve → grade_docs → (rewrite → retrieve)* → generate

    If retrieved docs are irrelevant, the agent rewrites the query and retries
    up to MAX_REWRITES times before falling back to whatever it has.
    """

    MAX_REWRITES = 2

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self.retriever = None
        self.graph = None
        self.llm = ModelLoader().load_llm()
        log.info("AgenticRAG initialized", session_id=self.session_id)

    # ---------- Public API ----------

    def load_retriever_from_faiss(self, index_path: str, k: int = 10, index_name: str = "index"):
        try:
            embeddings = ModelLoader().load_embeddings()
            vectorstore = FAISS.load_local(
                index_path, embeddings, index_name=index_name, allow_dangerous_deserialization=True
            )
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            self._build_graph()
            log.info("AgenticRAG: FAISS retriever loaded", index_path=index_path, k=k, session_id=self.session_id)
        except Exception as e:
            log.error("AgenticRAG: failed to load retriever", error=str(e))
            raise DocumentPortalException("AgenticRAG retriever loading failed", sys)

    def invoke(self, question: str) -> str:
        try:
            if self.graph is None:
                raise DocumentPortalException(
                    "Graph not initialized. Call load_retriever_from_faiss() first.", sys
                )
            initial: RAGState = {
                "question": question,
                "documents": [],
                "answer": "",
                "rewrite_count": 0,
            }
            result = self.graph.invoke(initial)
            return result["answer"]
        except Exception as e:
            log.error("AgenticRAG: invocation failed", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("AgenticRAG invocation failed", sys)

    # ---------- Graph nodes ----------

    def _retrieve(self, state: RAGState) -> dict:
        docs = self.retriever.invoke(state["question"])
        log.info("AgenticRAG: retrieved", count=len(docs), session_id=self.session_id)
        return {"documents": docs}

    def _grade_documents(self, state: RAGState) -> dict:
        grade_prompt = ChatPromptTemplate.from_template(
            "Is the document relevant to the question?\n"
            "Answer only 'yes' or 'no'.\n"
            "Document: {document}\nQuestion: {question}"
        )
        grader = grade_prompt | self.llm | StrOutputParser()
        relevant = [
            doc for doc in state["documents"]
            if "yes" in grader.invoke(
                {"document": doc.page_content[:400], "question": state["question"]}
            ).lower()
        ]
        log.info(
            "AgenticRAG: graded docs",
            relevant=len(relevant),
            total=len(state["documents"]),
            session_id=self.session_id,
        )
        return {"documents": relevant}

    def _rewrite_query(self, state: RAGState) -> dict:
        rewrite_prompt = ChatPromptTemplate.from_template(
            "Rewrite this question to retrieve better documents from a vector store.\n"
            "Original: {question}\nRewritten:"
        )
        new_q = (rewrite_prompt | self.llm | StrOutputParser()).invoke({"question": state["question"]})
        log.info(
            "AgenticRAG: rewrote query",
            original=state["question"],
            rewritten=new_q,
            session_id=self.session_id,
        )
        return {"question": new_q, "rewrite_count": state["rewrite_count"] + 1}

    def _generate(self, state: RAGState) -> dict:
        if not state["documents"]:
            return {"answer": "I couldn't find relevant information in the documents to answer your question."}
        context = "\n\n".join(d.page_content for d in state["documents"])
        gen_prompt = ChatPromptTemplate.from_template(
            "Answer the question based only on the context below. "
            "If the answer is not in the context, say 'I don't know.'\n\n"
            "Context: {context}\nQuestion: {question}\nAnswer:"
        )
        answer = (gen_prompt | self.llm | StrOutputParser()).invoke(
            {"context": context, "question": state["question"]}
        )
        log.info("AgenticRAG: generated answer", preview=answer[:100], session_id=self.session_id)
        return {"answer": answer}

    # ---------- Graph routing ----------

    def _decide_next(self, state: RAGState) -> str:
        if not state["documents"] and state["rewrite_count"] < self.MAX_REWRITES:
            return "rewrite"
        return "generate"

    # ---------- Graph construction ----------

    def _build_graph(self):
        g = StateGraph(RAGState)
        g.add_node("retrieve", self._retrieve)
        g.add_node("grade_docs", self._grade_documents)
        g.add_node("rewrite", self._rewrite_query)
        g.add_node("generate", self._generate)

        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "grade_docs")
        g.add_conditional_edges(
            "grade_docs",
            self._decide_next,
            {"rewrite": "rewrite", "generate": "generate"},
        )
        g.add_edge("rewrite", "retrieve")
        g.add_edge("generate", END)

        self.graph = g.compile()
        log.info("AgenticRAG: graph compiled", session_id=self.session_id)