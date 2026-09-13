"""Personal skills: reusable instruction snippets attached to agents by role/name.

A skill is a named prompt fragment the user defines in the playground settings.
When a workflow runs, each skill with a matching `apply` list (roles or agent
names) augments the agent instruction.
"""

from __future__ import annotations

from typing import Any


def apply_skills(manager: Any, skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Augment agent instructions with the personal skills.

    skill: {"name", "text", "apply": [role-or-agent-name, ...]}
    An empty/missing `apply` list attaches the skill to every agent.
    Returns a list of {"name", "applied_to": count} for reporting.
    """
    applied: list[dict[str, Any]] = []
    for skill in skills:
        if not isinstance(skill, dict):
            continue
        text = (skill.get("text") or "").strip()
        if not text:
            continue
        skill_name = (skill.get("name") or "skill").strip() or "skill"
        targets = [str(r).strip() for r in (skill.get("apply") or []) if str(r).strip()]
        count = 0
        for agent in manager.agents.values():
            if not targets or agent.role in targets or agent.name in targets:
                agent.instruction = f"{agent.instruction}\n\n[Skill: {skill_name}] {text}".strip()
                count += 1
        applied.append({"name": skill_name, "applied_to": count})
    return applied
