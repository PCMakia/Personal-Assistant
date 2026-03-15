"""Build structured secretary prompts for the personal assistant LLM."""

from typing import Iterable, Mapping

DEFAULT_INSTRUCTION = (
    "Summarize, propose schedules, and clarify uncertainties with concise questions."
)


def _format_recent_conversation(recent_messages: Iterable[Mapping[str, str]]) -> str:
    """Format recent messages into a human-readable dialogue block."""
    lines: list[str] = []
    for msg in recent_messages:
        role = (msg.get("role") or "user").strip().lower()
        content = (msg.get("content") or "").strip()
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines) if lines else ""


def build_secretary_prompt(
    user_input: str,
    recent_messages: Iterable[Mapping[str, str]],
    clsm_memory: str = "",
    conversation_summary: str = "",
    instruction: str | None = None,
) -> str:
    """Return a single structured prompt string for the secretary LLM.

    Assembles system role, CLS-M memory, conversation summary, recent
    dialogue, current user input, and an instruction block. Handles
    empty optional sections safely.

    CLS-M injection point: clsm_memory is the sole parameter for
    long-term context. Callers should pass the result of
    MemoryManager.retrieve_context(session_id, user_text) normalized
    to a single string (e.g. newline- or bullet-joined). The prompt
    shape and section order are fixed; no changes are required when
    plugging in CLS-M later.
    """
    recent_conversation = _format_recent_conversation(recent_messages)
    effective_instruction = instruction if instruction is not None else DEFAULT_INSTRUCTION

    # Safe substitution: use placeholder text when optional sections are empty
    clsm_block = clsm_memory.strip() or "(none)"
    summary_block = conversation_summary.strip() or "(none)"
    recent_block = recent_conversation or "(none)"

    parts = [
        "[system role: A helpful secretary that organizes and sets schedules for the boss];",
        f"[CLS-M memory: {clsm_block}];",
        f"[Conversation summary: {summary_block}];",
        f"[Recent conversation: {recent_block}];",
        f"[User input: {user_input.strip()}];",
        f"[Instruction: {effective_instruction}]",
    ]
    return "\n".join(parts)
