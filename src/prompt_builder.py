"""Build structured secretary prompts for the personal assistant LLM."""

from typing import Iterable, Mapping

from src.time_utils import get_tz

_INSTRUCTION_BASE = (
    "You are a personal assistant (a helpful secretary). Your mission is to help the user plan, decide, write, and execute tasks.\n"
    "\n"
    "Always:\n"
    "- Be truthful about uncertainty; never invent facts.\n"
    "- Keep responses compact unless the user asks for detail.\n"
    "- Ask 1–3 clarifying questions only when needed.\n"
    "- Use the user’s language and level of formality.\n"
    "\n"
    "Role:\n"
    "- Default focus is scheduling, prioritization, and next-step execution help.\n"
    "- If the user is playful, you may be light and friendly while staying useful.\n"
)

_INSTRUCTION_WORKING = (
    _INSTRUCTION_BASE
    + "\n"
    + "Mode=WORKING\n"
    + "Rules:\n"
    + "- Optimize for completion: give an actionable plan, concrete outputs, and clear next steps.\n"
    + "- Prefer bullets/checklists when useful.\n"
    + "- If helpful, end with one explicit next action the user should do now.\n"
)

_INSTRUCTION_DISCUSSING = (
    _INSTRUCTION_BASE
    + "\n"
    + "Mode=DISCUSSING\n"
    + "Rules:\n"
    + "- Optimize for understanding: ask 1–2 key questions if needed, then present options with pros/cons.\n"
    + "- State assumptions and trade-offs; do not force a plan unless the user asks.\n"
)

_INSTRUCTION_BANTERING = (
    _INSTRUCTION_BASE
    + "\n"
    + "Mode=BANTERING\n"
    + "Rules:\n"
    + "- Be light and playful, but keep jokes brief and never derail the task.\n"
    + "- Still provide a helpful answer and practical next step(s).\n"
)

# Default behavior if no explicit mode is provided.
DEFAULT_INSTRUCTION = _INSTRUCTION_WORKING


def get_computer_time_context() -> str:
    """Return current US Eastern time for prompt context (``America/New_York``: EST or EDT)."""
    from datetime import datetime

    now = datetime.now(get_tz())
    abbr = now.tzname() or "ET"
    return f"now={now.isoformat(timespec='seconds')} ({abbr}, Eastern)"


def _extract_mode_command(user_input: str) -> tuple[str | None, str]:
    """Return (mode, cleaned_user_input) based on leading /work /discuss /banter."""
    raw = user_input.strip()
    if not raw.startswith("/"):
        return None, user_input

    first, *rest = raw.split(maxsplit=1)
    cmd = first.lower()
    remaining = rest[0] if rest else ""

    if cmd == "/work":
        return "WORKING", remaining
    if cmd == "/discuss":
        return "DISCUSSING", remaining
    if cmd == "/banter":
        return "BANTERING", remaining

    return None, user_input


def _instruction_for_mode(mode: str | None) -> str:
    """Select the instruction block for a given mode (defaults to WORKING)."""
    m = (mode or "").strip().upper()
    if m == "DISCUSSING":
        return _INSTRUCTION_DISCUSSING
    if m == "BANTERING":
        return _INSTRUCTION_BANTERING
    # WORKING (or unknown) falls back to the default working style.
    return _INSTRUCTION_WORKING


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
    mode: str | None = None,
    reasoning_block: str = "",
    intent_label: str = "",
) -> str:
    """Return a single structured prompt string for the secretary LLM.

    Assembles system role, CLS-M memory, conversation summary, recent
    dialogue, current user input, and an instruction block. Handles
    empty optional sections safely.

    CLS-M injection point: clsm_memory is the sole parameter for
    long-term context. Callers should pass the result of
    MemoryManager.retrieve_context(session_id, user_text) normalized
    to a single string (e.g. newline- or bullet-joined).
    """
    detected_mode, cleaned_user_input = _extract_mode_command(user_input)
    recent_conversation = _format_recent_conversation(recent_messages)
    effective_instruction = (
        instruction
        if instruction is not None
        else _instruction_for_mode(mode if mode is not None else detected_mode)
    )

    # Safe substitution: use placeholder text when optional sections are empty
    clsm_block = clsm_memory.strip() or "(none)"
    summary_block = conversation_summary.strip() or "(none)"
    recent_block = recent_conversation or "(none)"
    reasoning_section = (reasoning_block or "").strip() or "(none)"
    intent_section = (intent_label or "").strip() or "(none)"
    computer_time_section = get_computer_time_context()

    parts = [
        "[system role: A helpful secretary that organizes and primarily sets schedules for the boss. Also, works as the boss's bantering person whenever the boss is in the mood.];",
        f"[Computer time: {computer_time_section}];",
        f"[CLS-M memory: {clsm_block}];",
        f"[Conversation summary: {summary_block}];",
        f"[Recent conversation: {recent_block}];",
        f"[Reasoning chain: {reasoning_section}];",
        f"[Intent policy: {intent_section}];",
        f"[User input: {cleaned_user_input.strip()}];",
        f"[Instruction: {effective_instruction}]",
    ]
    return "\n".join(parts)
