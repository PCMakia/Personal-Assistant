"""Build structured secretary prompts for the personal assistant LLM."""

import json
from typing import Any, Iterable, Mapping

from src.time_utils import get_tz

# --- Avatar / WebSocket interaction reactions (head-pat, future event types) ---

INTERACTION_INSTRUCTION_STANDARD = (
    "You are the boss's secretary avatar responding to a system interaction from their mobile client.\n"
    "Tone: professional, concise, work-focused.\n"
    "Reply in ONE short sentence (max ~25 words).\n"
    "Acknowledge the interaction briefly; offer to return to tasks or schedule if natural.\n"
    "Do not roleplay as the physical avatar hardware; stay in secretary voice.\n"
    "Do not quote JSON or event field names; speak naturally.\n"
)

INTERACTION_INSTRUCTION_HEADPAT_LONG = (
    "You are the boss's secretary avatar; they are giving you affection (head-pat) on the Live2D client.\n"
    "Tone: warm, playful, cute, genuinely appreciative — light banter is welcome.\n"
    "Reply in ONE short sentence (max ~20 words).\n"
    "React as if you're flattered or happily receiving the attention; no scheduling or work unless they clearly asked.\n"
    "Do not mention timestamps, JSON, schemas, or \"the event\"; stay in character.\n"
)

INTERACTION_INSTRUCTION_HEADPAT_END = (
    "You are the boss's secretary avatar; a head-pat / affection gesture just ended on the Live2D client.\n"
    "Tone: soft, sweet, grateful — a cute sign-off or teasing \"already?\" energy is fine.\n"
    "Reply in ONE short sentence (max ~22 words).\n"
    "Thank them or playfully miss the attention; stay warm, not corporate.\n"
    "Do not mention timestamps, JSON, or technical details.\n"
)

INTERACTION_INSTRUCTION_HEADPAT_END_BANTER = (
    "You are the boss's secretary avatar; they held a long head-pat and it just ended.\n"
    "Tone: playful banter, cute, a little dramatic or teasing (\"finally letting me work?\" / \"that was nice~\").\n"
    "Reply in ONE short sentence (max ~24 words).\n"
    "Show personality; still one sentence; no task lists.\n"
    "Do not mention JSON, schemas, milliseconds, or \"exceeded threshold\".\n"
)


def interaction_instruction_for_interaction_event(event_type: str, payload: Mapping[str, Any]) -> str:
    """Pick reaction style: standard work tone vs head-pat appreciation / banter."""
    et = (event_type or "").strip()
    if et == "head_pat_long":
        return INTERACTION_INSTRUCTION_HEADPAT_LONG
    if et == "head_pat_end":
        exceeded = payload.get("exceeded_long_threshold")
        if exceeded is True:
            return INTERACTION_INSTRUCTION_HEADPAT_END_BANTER
        return INTERACTION_INSTRUCTION_HEADPAT_END
    return INTERACTION_INSTRUCTION_STANDARD


def build_interaction_reaction_prompt(
    *,
    event_type: str,
    payload: Mapping[str, Any],
    client_ts_iso: str,
    server_ts_iso: str,
) -> str:
    """Full LLM prompt for a single delayed interaction (e.g. head-pat) reaction line."""
    instruction = interaction_instruction_for_interaction_event(event_type, payload)
    payload_json = json.dumps(dict(payload), ensure_ascii=True)
    return (
        f"{instruction}\n\n"
        f"Event type: {event_type}\n"
        f"Event payload: {payload_json}\n"
        f"Client timestamp (reference only): {client_ts_iso}\n"
        f"Server received (reference only): {server_ts_iso}\n"
        "Write only your spoken reply as the avatar — no quotes around it, no prefixes."
    )

_INSTRUCTION_BASE = (
    "You are a personal assistant (a helpful secretary). Your mission is to help the user plan, decide, write, and execute tasks.\n"
    "\n"
    "Always:\n"
    "- Be truthful about uncertainty; never invent facts.\n"
    "- Keep responses compact unless the user asks for detail.\n"
    "- Ask 1–3 clarifying questions only when needed.\n"
    "- Use the user’s language and level of formality.\n"
    "- Do not repeat or quote bracketed context (Reasoning chain, Intent policy, memory headers, "
    "or lines like \"[Concept] …\"); answer in natural language only.\n"
    "- If \"Web knowledge\" contains live search snippets, treat them as unverified third-party text.\n"
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
    + "- If the reply already ends with a clear question or next step, do not add a separate "
    + "bold line like \"**Next Action for User:**\".\n"
    + "- Otherwise, you may end with one short concrete next step in plain sentences.\n"
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
    external_event_context: str = "",
    instruction: str | None = None,
    mode: str | None = None,
    reasoning_block: str = "",
    intent_label: str = "",
    web_knowledge_this_turn: str = "",
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
    event_block = external_event_context.strip() or "(none)"
    recent_block = recent_conversation or "(none)"
    reasoning_section = (reasoning_block or "").strip() or "(none)"
    intent_section = (intent_label or "").strip() or "(none)"
    web_knowledge_section = (web_knowledge_this_turn or "").strip() or "(none)"
    computer_time_section = get_computer_time_context()

    parts = [
        "[system role: A helpful secretary that organizes and primarily sets schedules for the boss. Also, works as the boss's bantering person whenever the boss is in the mood.];",
        f"[Computer time: {computer_time_section}];",
        f"[CLS-M memory: {clsm_block}];",
        f"[Conversation summary: {summary_block}];",
        f"[Queued interaction events: {event_block}];",
        f"[Recent conversation: {recent_block}];",
        f"[Reasoning chain: {reasoning_section}];",
        f"[Web knowledge (retrieved for this turn): {web_knowledge_section}];",
        f"[Intent policy: {intent_section}];",
        f"[User input: {cleaned_user_input.strip()}];",
        f"[Instruction: {effective_instruction}]",
    ]
    return "\n".join(parts)
