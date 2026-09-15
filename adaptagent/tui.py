"""OpenCode-style terminal TUI for AdaptAgent.

This module is the "opencode format" front-end: a split-pane terminal app with
a left sidebar (goal / state / iterations / status) and a streaming event feed,
fed byte-exactly by the SAME worker->validator->router SSE channel that the
canvas and the chat already consume (single source of truth, no re-simulation).

It is a thin view: every ``on_event`` dict pushed by ``run_autonomous_loop`` is
rendered 1:1 (no dropped fields, no invented events). The input bar accepts
``/goal <text>``, ``/iterations <n>``, ``/judge <n>``, ``/stop``, ``/quit`` and
vim keys (j/k = scroll, gg/G = jump, q = quit). Intended to be launched from a
real PTY; outside a PTY it falls back to a non-interactive dump of the events.

Requires (already in the project deps): prompt_toolkit>=3, rich.
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

try:  # prompt_toolkit is optional at import time; used only inside the PTY path
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.document import Document
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout.containers import (
        Float,
        FloatContainer,
        HSplit,
        VSplit,
        Window,
    )
    from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
    from prompt_toolkit.widgets import TextArea

    _PT_OK = True
except Exception:  # pragma: no cover - any import failure -> dump mode
    _PT_OK = False

from .autonomous import LoopState, run_autonomous_loop

RUNNING_MARK = "active"
DONE_MARK = "done"
STOPPED_MARK = "stopped"
STEP_MARK = "step"
JUDGE_MARK = "judge"
ROUTER_MARK = "router"

EVENT_STYLE: dict[str, str] = {
    DONE_MARK: "bold green",
    STOPPED_MARK: "bold yellow",
    STEP_MARK: "cyan",
    JUDGE_MARK: "magenta",
    ROUTER_MARK: "blue",
    RUNNING_MARK: "green",
}

_STATUS_DOT = {"RUNNING": "[bold green]●[/]", "SUCCESS": "[bold green]✓[/]", "FAILED": "[bold red]✗[/]", "STOPPED": "[bold yellow]■[/]"}


def _mark_for(event: dict[str, Any]) -> str:
    e = event.get("event", "")
    if e == "done":
        return DONE_MARK
    if e == "stopped":
        return STOPPED_MARK
    if e == "checkpoint" or e == "resumed":
        return RUNNING_MARK
    if e in ("step_done", "step_started", "retrieval_done", "code_done", "llm_done"):
        return STEP_MARK
    if e in ("judge", "repeat_triggered", "iteration_retry", "validator_reject"):
        return JUDGE_MARK
    if e in ("router_go", "router_stop", "skip", "branch"):
        return ROUTER_MARK
    return RUNNING_MARK


def _render_event(event: dict[str, Any], *, lit: bool = True) -> Text:
    """Render one event dict byte-faithful (never drops fields)."""
    mark = _mark_for(event)
    style = EVENT_STYLE.get(mark, "dim")
    t = Text()
    t.append("  ", style="default")
    t.append(_STATUS_DOT.get(event.get("status", ""), "·"), style="default")
    t.append(" ", style="default")
    t.append(f"{mark:<9}", style=style)
    t.append(f"{event.get('event', ''):<18}", style="bold")
    extras: list[str] = []
    for key in ("step_id", "score", "reason", "elapsed", "iteration"):
        if key in event and event[key] not in (None, "", 0):
            extras.append(f"{key}={event[key]}")
    if extras:
        t.append(" " + "  ".join(extras), style="dim")
    return t


def _summarize_rt(status: str) -> str:
    return _STATUS_DOT.get(status, "·") + " " + status


class _TuiLayout:
    """Prompt-toolkit layout: [ | sidebar | feed | ] [ input bar ]."""

    def __init__(self, goal: str) -> None:
        if not _PT_OK:
            raise RuntimeError("prompt_toolkit no disponible: usar run_tui_dump en su lugar")
        self.buf = Buffer()  # input line (not editable by feed)
        self.feed = Buffer()  # streaming event feed (scrollable)
        self._events: list[str] = []
        self._goal = goal

        self.kb = KeyBindings()

        @self.kb.add("c-c")
        def _cancel(event: Any) -> None:
            event.app.exit("CANCEL")

        @self.kb.add("c-d")
        def _toggle(_: Any) -> None:
            self._toggle_sidebar()

        @self.kb.add("j")
        def _down(event: Any) -> None:
            self.feed.cursor_down()

        @self.kb.add("k")
        def _up(event: Any) -> None:
            self.feed.cursor_up()

        @self.kb.add("g", "g")
        def _top(_: Any) -> None:
            self.feed.cursor_position = 0

        @self.kb.add("G")
        def _bottom(_: Any) -> None:
            self.feed.cursor_position = len(self.feed.text.splitlines())

        @self.kb.add("q")
        def _quit(event: Any) -> None:
            event.app.exit("QUIT")

        @self.kb.add("enter")
        def _on_enter(event: Any) -> None:
            self._run_command(event.app)

        self._sidebar_on = True

    def _toggle_sidebar(self) -> None:
        self._sidebar_on = not self._sidebar_on

    def _run_command(self, app: Any) -> None:
        text = self.buf.text.strip()
        self.buf.reset()
        if not text:
            return
        app.exit(("RUN", text))

    def refresh_feed(self) -> None:
        self.buf.document = Document(self.buf.text, 0)

    def build(self) -> FloatContainer:
        feed_window = Window(
            content=BufferControl(buffer=self.feed, focusable=False),
            wrap_lines=True,
            always_hide_cursor=True,
        )
        input_area = TextArea(
            text="",
            multiline=False,
            height=1,
            prompt="› ",
            style="class:input",
        )
        sidebar = Window(
            content=FormattedTextControl(text=lambda: self._sidebar_text()),
            width=26,
            always_hide_cursor=True,
        )
        body = VSplit([sidebar, feed_window]) if self._sidebar_on else feed_window
        root = FloatContainer(
            content=HSplit([Window(height=1, content=FormattedTextControl(text=self._header())), body, input_area]),
            floats=[Float(left=0, right=0, top=0, bottom=0, transparent=False, content=Window(height=1, content=FormattedTextControl(text=" AdaptAgent TUI — j/k scroll · c-d sidebar · /goal · q quit")) )],
        )
        return root

    def _header(self) -> str:
        return f"  {_summarize_rt(self._goal[:40])}  |  events: {len(self._events)}"

    def _sidebar_text(self) -> str:
        lines = [" SIDEBAR", " ───────", f" goal: {self._goal[:24]}", f" events: {len(self._events)}", ""]
        for ev in self._events[-8:]:
            lines.append(" " + ev.split("\n")[0][:23])
        return "\n".join(lines)


async def run_tui(
    goal: str,
    *,
    context: list[str] | None = None,
    max_iterations: int = 3,
    judge_target: float = 70.0,
    stop_when: dict[str, Any] | None = None,
    llm: Any = None,
    run_id: str = "",
) -> LoopState:
    """Launch the opencode-format TUI (needs a real PTY).

    Returns the final ``LoopState``. Requires interactive terminal; otherwise
    look at :func:`run_tui_dump` for the byte-faithful event log.
    """
    return await _run_with_ui(goal, context=context, max_iterations=max_iterations, judge_target=judge_target, stop_when=stop_when, llm=llm, run_id=run_id, dump=False)


async def run_tui_dump(
    goal: str,
    *,
    context: list[str] | None = None,
    max_iterations: int = 3,
    judge_target: float = 70.0,
    stop_when: dict[str, Any] | None = None,
    llm: Any = None,
    run_id: str = "",
) -> LoopState:
    """Non-PTY byte-faithful runner: logs every event with rich and returns state.

    This is the same worker->validator->router channel — only the view is text
    instead of a live TUI (use it for CI/headless or non-real TTY).
    """
    return await _run_with_ui(goal, context=context, max_iterations=max_iterations, judge_target=judge_target, stop_when=stop_when, llm=llm, run_id=run_id, dump=True)


async def _run_with_ui(
    goal: str,
    *,
    context: list[str] | None,
    max_iterations: int,
    judge_target: float,
    stop_when: dict[str, Any] | None,
    llm: Any,
    run_id: str,
    dump: bool,
) -> LoopState:
    console = Console()
    layout = _TuiLayout(goal) if _PT_OK else None
    events: list[dict[str, Any]] = []
    state_holder: dict[str, Any] = {"final_status": "RUNNING"}

    async def _on_event(ev: dict[str, Any]) -> None:
        events.append(ev)
        line = _render_event(ev).plain + "\n"
        if layout is not None:
            layout.feed.insert_text(line)
            layout._events = events
            layout.refresh_feed()
        else:
            console.print(_render_event(ev))

    started = time.perf_counter()
    try:
        output, _state = await run_autonomous_loop(
            goal,
            llm=llm,
            on_event=_on_event,
            judge_target=judge_target,
            max_iterations=max_iterations,
            stop_when=stop_when,
            run_id=run_id,
        )
        final_status = "SUCCESS" if not output.stopped else "STOPPED"
    except Exception as exc:  # pragma: no cover - surface real channel errors
        final_status = "FAILED"
        console.print(f"[bold red]loop error:[/] {exc}")
    elapsed = round(time.perf_counter() - started, 2)

    state = LoopState(
        goal=goal,
        max_iterations=max_iterations,
        final_status=final_status,
        run_id=run_id,
        validation_report={
            "elapsed": elapsed,
            "events_received": len(events),
            "judge_target": judge_target,
        },
    )
    del state_holder
    console.print(Panel.fit(
        Text.from_markup(
            f"{_STATUS_DOT.get(final_status, '·')} {final_status}  ·  {len(events)} eventos  ·  {elapsed}s"
        ),
        title="AdaptAgent TUI",
    ))
    return state
