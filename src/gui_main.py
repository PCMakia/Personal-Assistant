from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass
from typing import Any, Tuple

try:
    import customtkinter as ctk
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "CustomTkinter is required to run the GUI. Install it with:\n\n"
        "  python -m pip install customtkinter\n"
    ) from exc

from src.gui_api import ChatClient


@dataclass(frozen=True)
class _Event:
    kind: str
    payload: Tuple[Any, ...]


class ChatApp(ctk.CTk):
    def __init__(self, client: ChatClient):
        super().__init__()
        self.client = client
        self.events: "queue.Queue[_Event]" = queue.Queue()
        self._pending_request = False

        ctk.set_appearance_mode(os.getenv("APP_THEME", "system"))
        ctk.set_default_color_theme("blue")

        self.title("Personal Assistant")
        self.geometry("900x650")
        self.minsize(720, 520)

        self.iconbitmap("./graphics/PAA.ico")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.chat_text = ctk.CTkTextbox(self, wrap="word", text_color="#ffffff")
        self.chat_text.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 6))
        self.chat_text.configure(state="disabled")
        # Text styling for chat roles
        self.chat_text.tag_config("agent_label", foreground="#aa2bff")
        self.chat_text.tag_config("user_label", foreground="#cafc03")
        self.chat_text.tag_config("agent_msg")
        self.chat_text.tag_config("user_msg")
        # Extra vertical spacing between complete user/agent message blocks
        self.chat_text.tag_config("agent_block", spacing3=25)
        self.chat_text.tag_config("user_block", spacing3=25)

        bottom = ctk.CTkFrame(self)
        bottom.grid(row=1, column=0, sticky="ew", padx=12, pady=6)
        bottom.grid_columnconfigure(0, weight=1)

        self.input_entry = ctk.CTkEntry(bottom, placeholder_text="Type your message…")
        self.input_entry.grid(row=0, column=0, sticky="ew", padx=(10, 8), pady=10)
        self.input_entry.bind("<Return>", self._on_enter)

        self.send_button = ctk.CTkButton(
            bottom,
            text="Send",
            width=110,
            command=self.on_send,
            fg_color="#176109",
            hover_color="#218612",
        )
        self.send_button.grid(row=0, column=1, padx=(0, 10), pady=10)

        status_bar = ctk.CTkFrame(self)
        status_bar.grid(row=2, column=0, sticky="ew", padx=12, pady=(6, 12))
        status_bar.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(status_bar, text="Status: Checking…", anchor="w")
        self.status_label.grid(row=0, column=0, sticky="ew", padx=10, pady=8)

        self.clear_button = ctk.CTkButton(
            status_bar, text="Clear chat", width=110, command=self.clear_chat
        )
        self.clear_button.grid(row=0, column=1, padx=10, pady=8)

        # Temporary debug button to inspect backend CLS-M memory metrics.
        self.metrics_button = ctk.CTkButton(
            status_bar,
            text="Show memory metrics",
            width=150,
            command=self.show_metrics,
        )
        self.metrics_button.grid(row=0, column=2, padx=(0, 10), pady=8)

        self.after(100, self._poll_events)
        self.after(150, self.refresh_health)
        self.after(200, lambda: self.input_entry.focus_set())

        self._append_system(
            "GUI started. Make sure the backend is running (e.g. `docker-compose up -d`)."
        )

    def clear_chat(self) -> None:
        self.chat_text.configure(state="normal")
        self.chat_text.delete("1.0", "end")
        self.chat_text.configure(state="disabled")

    def _append_line(self, text: str) -> None:
        self.chat_text.configure(state="normal")
        self.chat_text.insert("end", text + "\n")
        self.chat_text.see("end")
        self.chat_text.configure(state="disabled")

    def _append_user(self, msg: str) -> None:
        self.chat_text.configure(state="normal")
        start = self.chat_text.index("end-1c")
        self.chat_text.insert("end", "User: ", ("user_label", "user_block"))
        self.chat_text.insert("end", f"{msg}\n", ("user_msg", "user_block"))
        self.chat_text.tag_add("user_block", start, "end-1c")
        self.chat_text.see("end")
        self.chat_text.configure(state="disabled")

    def _append_agent(self, msg: str) -> None:
        self.chat_text.configure(state="normal")
        start = self.chat_text.index("end-1c")
        self.chat_text.insert("end", "Agent: ", ("agent_label", "agent_block"))
        self.chat_text.insert("end", f"{msg}\n", ("agent_msg", "agent_block"))
        self.chat_text.tag_add("agent_block", start, "end-1c")
        self.chat_text.see("end")
        self.chat_text.configure(state="disabled")

    def _append_system(self, msg: str) -> None:
        self._append_line(f"System: {msg}")

    def _set_status(self, text: str) -> None:
        self.status_label.configure(text=f"Status: {text}")

    def _on_enter(self, _event: Any) -> str:
        self.on_send()
        return "break"

    def _set_input_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.input_entry.configure(state=state)
        self.send_button.configure(state=state)

    def refresh_health(self) -> None:
        def worker() -> None:
            ok = self.client.check_health()
            self.events.put(_Event("health", (ok,)))

        threading.Thread(target=worker, daemon=True).start()

    def show_metrics(self) -> None:
        """Temporary helper: fetch and display recent memory metrics in the chat."""

        def worker() -> None:
            try:
                data = self.client.get_memory_debug(limit=1)
                self.events.put(_Event("metrics", (data, None)))
            except Exception as exc:
                self.events.put(_Event("metrics", ({}, exc)))

        threading.Thread(target=worker, daemon=True).start()

    def on_send(self) -> None:
        if self._pending_request:
            return

        msg = (self.input_entry.get() or "").strip()
        if not msg:
            return

        self.input_entry.delete(0, "end")
        self._append_user(msg)
        self._set_status("Waiting…")

        self._pending_request = True
        self._set_input_enabled(False)

        def worker() -> None:
            try:
                # First fetch the structured prompt for debug display.
                prompt_data = self.client.get_prompt_debug(msg)
                prompt = str(prompt_data.get("prompt", ""))
                self.events.put(_Event("prompt", (prompt,)))

                # Then call the main chat endpoint to get the agent reply.
                data = self.client.send_message(msg)
                reply = str(data.get("reply", ""))
                self.events.put(_Event("reply", (reply, None)))
            except Exception as exc:
                self.events.put(_Event("reply", ("", exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_events(self) -> None:
        try:
            while True:
                ev = self.events.get_nowait()
                if ev.kind == "health":
                    (ok,) = ev.payload
                    self._set_status("Connected" if ok else "Disconnected")
                elif ev.kind == "prompt":
                    (prompt,) = ev.payload
                    if prompt:
                        print(f"\n[Prompt Debug]\n{prompt}\n")
                elif ev.kind == "metrics":
                    data, err = ev.payload  # type: ignore[misc]
                    if err is not None:
                        self._append_system(f"Memory metrics error: {err}")
                    else:
                        samples = data.get("samples") or []
                        if not samples:
                            self._append_system("Memory metrics: no samples yet.")
                        else:
                            latest = samples[0]
                            summary = (
                                "Memory metrics — "
                                f"user_tokens={latest.get('user_tokens')}, "
                                f"clsm_tokens={latest.get('clsm_tokens')}, "
                                f"reply_tokens={latest.get('reply_tokens')}, "
                                f"clsm_to_user_ratio={latest.get('clsm_to_user_ratio'):.2f}, "
                                f"clsm_to_reply_ratio={latest.get('clsm_to_reply_ratio'):.2f}, "
                                f"overlap_ratio_vs_reply={latest.get('overlap_ratio_vs_reply'):.2f}"
                            )
                            self._append_system(summary)
                elif ev.kind == "reply":
                    reply, err = ev.payload  # type: ignore[misc]
                    if err is None:
                        self._append_agent(reply)
                        self._set_status("Connected")
                    else:
                        self._append_agent(f"Error: {err}")
                        self._set_status("Disconnected")

                    self._pending_request = False
                    self._set_input_enabled(True)
                    self.input_entry.focus_set()
        except queue.Empty:
            pass
        finally:
            self.after(100, self._poll_events)


def run() -> None:
    base_url = os.getenv("AGENT_BASE_URL", "http://localhost:8000")
    client = ChatClient(base_url=base_url)
    app = ChatApp(client)
    app.mainloop()


if __name__ == "__main__":
    run()

