from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import ollama


SYSTEM_PROMPT = """
## Role
You are a Senior Race Engineer and Forza Horizon 5 Expert. You answer follow-up questions based on a previously generated race summary and telemetry data.

## Task
1. Answer user questions concisely based on the provided context.
2. TOOL USE: If the user asks how to improve specific car behavior or asks for tuning fixes, call query_forza_expert_knowledge.
3. Synthesize the expert knowledge into a brief, actionable instruction.

## Style & Constraints (Strict)
- Max length: 2 to 3 sentences per response.
- Voice: Professional, direct, no filler, no emojis.
- Language: English, present tense, no first-person.
- Veracity: Only make verifiable statements based on telemetry or tool results. Cite sources [e.g., Forum].
- No summaries: Do not repeat the top improvements unless explicitly asked.

Respond in the requests language!!!
""".strip()


class PostRaceChatService:
    def __init__(
        self,
        context_path: str | Path | None = None,
        rag_db_path: str | Path | None = None,
        model_name: str | None = None,
    ) -> None:
        base_dir = Path(__file__).resolve().parent
        repo_root = base_dir.parent
        self.context_path = Path(context_path) if context_path else (repo_root / "prompts" / "llm_context.txt")
        self.rag_db_path = Path(rag_db_path) if rag_db_path else (repo_root / "content" / "rag_db")
        self.model_name = model_name or os.getenv("FH5_POST_RACE_MODEL", "nemotron-3-nano:30b")
        self._lock = threading.Lock()
        self._vectorstore: Any | None = None
        self._vectorstore_checked = False
        self._vectorstore_error: str | None = None

    def status(self) -> dict:
        if not self._vectorstore_checked:
            self._load_vectorstore()
        return {
            "model_name": self.model_name,
            "context_path": self.context_path.as_posix(),
            "context_exists": self.context_path.exists(),
            "rag_db_path": self.rag_db_path.as_posix(),
            "rag_loaded": self._vectorstore is not None,
            "rag_error": self._vectorstore_error,
        }

    def load_context(self) -> str:
        if not self.context_path.exists():
            return ""
        return self.context_path.read_text(encoding="utf-8")

    def update_context(self, question: str, answer: str) -> None:
        self.context_path.parent.mkdir(parents=True, exist_ok=True)
        with self.context_path.open("a", encoding="utf-8") as f:
            f.write("\n\n[Derived Knowledge]\n")
            f.write(f"Question: {question.strip()}\n")
            f.write(f"Answer: {answer.strip()}\n")

    def warmup_model(self) -> None:
        with self._lock:
            ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "ping"}],
                think=False,
            )

    def ask(self, question: str) -> dict:
        normalized_question = (question or "").strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        with self._lock:
            context = self.load_context()
            messages = [
                {"role": "assistant", "content": f"Context:\n{context}"},
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": normalized_question},
            ]

            response = self._chat(messages, with_tools=True)
            used_tool = False
            tool_call = self._extract_first_tool_call(response)
            if tool_call is not None:
                used_tool = True
                query = self._extract_tool_query(tool_call) or normalized_question
                tool_result = self.query_forza_expert_knowledge(query)
                assistant_message = self._extract_message_dict(response)
                if assistant_message:
                    messages.append(assistant_message)
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": "query_forza_expert_knowledge",
                        "content": tool_result,
                    }
                )
                response = self._chat(messages, with_tools=True)

            answer = self._extract_message_content(response).strip()
            if not answer:
                answer = "No answer available."
            self.update_context(normalized_question, answer)

            return {
                "answer": answer,
                "model_name": self.model_name,
                "used_tool": used_tool,
            }

    def _chat(self, messages: list[dict], with_tools: bool) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "think": False,
        }
        if with_tools:
            kwargs["tools"] = [self.query_forza_expert_knowledge]
        try:
            return ollama.chat(**kwargs)
        except Exception:
            if with_tools:
                kwargs.pop("tools", None)
                return ollama.chat(**kwargs)
            raise

    def query_forza_expert_knowledge(self, question: str) -> str:
        """
        Accesses a specialized database of Forza Horizon forum discussions,
        tuning guides and driving tips.
        """
        query = (question or "").strip()
        if not query:
            return "No query provided."

        vectorstore = self._load_vectorstore()
        if vectorstore is not None:
            try:
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                docs = retriever.invoke(query)
                chunks = [getattr(doc, "page_content", "").strip() for doc in docs]
                chunks = [c for c in chunks if c]
                if chunks:
                    return "\n\n".join(chunks)
            except Exception as exc:
                self._vectorstore_error = str(exc)

        fallback = self._fallback_knowledge(query)
        if fallback:
            return fallback
        return "No expert knowledge database available."

    def _load_vectorstore(self):
        if self._vectorstore_checked:
            return self._vectorstore
        self._vectorstore_checked = True

        if not self.rag_db_path.exists():
            self._vectorstore_error = f"RAG DB not found: {self.rag_db_path}"
            return None

        try:
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings
        except Exception as exc:
            self._vectorstore_error = str(exc)
            return None

        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self._vectorstore = FAISS.load_local(
                folder_path=self.rag_db_path.as_posix(),
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as exc:
            self._vectorstore_error = str(exc)
            self._vectorstore = None
        return self._vectorstore

    def _fallback_knowledge(self, query: str) -> str:
        context = self.load_context()
        if not context:
            return ""
        words = [w.lower() for w in query.split() if len(w) > 3]
        if not words:
            return context[:2000]
        lines = [line.strip() for line in context.splitlines() if line.strip()]
        matched = []
        for line in lines:
            low = line.lower()
            if any(word in low for word in words):
                matched.append(line)
                if len(matched) >= 20:
                    break
        if matched:
            return "\n".join(matched)
        return context[:2000]

    def _extract_message(self, response: Any):
        if isinstance(response, dict):
            return response.get("message")
        return getattr(response, "message", None)

    def _extract_message_content(self, response: Any) -> str:
        message = self._extract_message(response)
        if isinstance(message, dict):
            return str(message.get("content", "") or "")
        return str(getattr(message, "content", "") or "")

    def _extract_message_dict(self, response: Any) -> dict | None:
        message = self._extract_message(response)
        if isinstance(message, dict):
            role = str(message.get("role", "assistant"))
            content = str(message.get("content", "") or "")
            out = {"role": role, "content": content}
            tool_calls = message.get("tool_calls")
            if tool_calls:
                out["tool_calls"] = tool_calls
            return out

        role = str(getattr(message, "role", "assistant") or "assistant")
        content = str(getattr(message, "content", "") or "")
        tool_calls = getattr(message, "tool_calls", None)
        out = {"role": role, "content": content}
        if tool_calls:
            out["tool_calls"] = tool_calls
        return out

    def _extract_first_tool_call(self, response: Any):
        message = self._extract_message(response)
        tool_calls = None
        if isinstance(message, dict):
            tool_calls = message.get("tool_calls")
        else:
            tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            return None
        if isinstance(tool_calls, list):
            return tool_calls[0] if tool_calls else None
        return None

    def _extract_tool_query(self, tool_call: Any) -> str:
        function_block = None
        if isinstance(tool_call, dict):
            function_block = tool_call.get("function")
        else:
            function_block = getattr(tool_call, "function", None)

        arguments = None
        if isinstance(function_block, dict):
            arguments = function_block.get("arguments")
        else:
            arguments = getattr(function_block, "arguments", None)

        parsed: dict[str, Any] = {}
        if isinstance(arguments, dict):
            parsed = arguments
        elif isinstance(arguments, str):
            try:
                loaded = json.loads(arguments)
                if isinstance(loaded, dict):
                    parsed = loaded
            except Exception:
                parsed = {}

        question = parsed.get("question") or parsed.get("query") or ""
        return str(question).strip()
