class ConversationMemory:
    """Sliding window conversation memory for prompt injection."""

    def __init__(self, max_turns=10):
        self._turns = []
        self._max_turns = max_turns

    def add_turn(self, role, content):
        """Append a turn, evict oldest pair if over max."""
        self._turns.append({"role": role, "content": content})
        if len(self._turns) > self._max_turns * 2:
            self._turns = self._turns[2:]

    def get_history(self):
        """Return copy of turns list."""
        return list(self._turns)

    def clear(self):
        """Reset conversation history."""
        self._turns = []

    def format_for_prompt(self):
        """Format history as string for system prompt injection."""
        if not self._turns:
            return ""
        lines = ["Previous conversation:"]
        for turn in self._turns:
            label = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{label}: {turn['content']}")
        return "\n".join(lines)
