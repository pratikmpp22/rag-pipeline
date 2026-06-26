class HybridMemory:
    """Two-tier conversation memory: recent turns verbatim + LLM-compressed summary."""

    def __init__(self, token_budget=2000, summary_llm=None):
        self._summary: str = ""
        self._recent_turns: list[dict] = []  # [{"role": "user"|"assistant", "content": str}]
        self._token_budget = token_budget
        self._summary_llm = summary_llm

    def add_turn(self, role: str, content: str):
        """Append a turn. Triggers compression if budget exceeded."""
        self._recent_turns.append({"role": role, "content": content})
        self._maybe_compress()

    def _estimate_tokens(self, text: str) -> int:
        """Approximate token count. ~4 chars/token for English."""
        return len(text) // 4

    def _total_tokens(self) -> int:
        summary_tokens = self._estimate_tokens(self._summary)
        turns_tokens = sum(self._estimate_tokens(t["content"]) for t in self._recent_turns)
        return summary_tokens + turns_tokens

    def _maybe_compress(self):
        """If over budget and we have enough turns, fold oldest pair into summary."""
        while self._total_tokens() > self._token_budget and len(self._recent_turns) > 2:
            # Pop the oldest user+assistant pair
            oldest_pair = self._recent_turns[:2]
            self._recent_turns = self._recent_turns[2:]

            if self._summary_llm:
                self._summary = self._compress_into_summary(self._summary, oldest_pair)
            # If no LLM available, just drop oldest (graceful degradation)

    def _compress_into_summary(self, existing_summary: str, turns: list[dict]) -> str:
        """LLM call: fold turns into the running summary."""
        exchange = "\n".join(
            f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content']}"
            for t in turns
        )
        prompt = (
            "You are a conversation summarizer. Update the existing summary "
            "to incorporate the new exchange. Be concise but preserve key facts, "
            "decisions, and any specific data points mentioned.\n\n"
            f"Existing summary:\n{existing_summary or '(empty — this is the start)'}\n\n"
            f"New exchange:\n{exchange}\n\n"
            "Updated summary:"
        )
        try:
            response = self._summary_llm.invoke(prompt)
            content = response.content
            if isinstance(content, list):
                content = "".join(
                    str(c) if not isinstance(c, dict) else c.get("text", "")
                    for c in content
                )
            return content.strip()
        except Exception:
            # Fallback: keep a simple text concatenation
            return f"{existing_summary}\n{exchange}".strip()

    def format_for_prompt(self) -> str:
        """Format memory for injection into the system prompt."""
        if not self._summary and not self._recent_turns:
            return ""

        parts = []
        if self._summary:
            parts.append(f"Summary of earlier conversation:\n{self._summary}")
        if self._recent_turns:
            parts.append("Recent conversation:")
            for turn in self._recent_turns:
                label = "User" if turn["role"] == "user" else "Assistant"
                parts.append(f"{label}: {turn['content']}")
        return "\n\n".join(parts)

    def get_history(self) -> list[dict]:
        """Return copy of recent turns (for inspection/testing)."""
        return list(self._recent_turns)

    def get_summary(self) -> str:
        """Return current summary (for inspection/testing)."""
        return self._summary

    def clear(self):
        """Reset all memory state."""
        self._summary = ""
        self._recent_turns = []
