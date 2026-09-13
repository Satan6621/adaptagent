"""Tests for personal skills (apply_skills)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import make_graph  # noqa: E402

from adaptagent.agents import AgentManager  # noqa: E402
from adaptagent.skills import apply_skills  # noqa: E402


def _manager():
    manager = AgentManager()
    manager.build_agents_from_workflow(make_graph())
    return manager


def test_apply_skills_targets_roles():
    manager = _manager()
    summary = apply_skills(manager, [{"name": "Revisor", "text": "Verifica los casos límite.", "apply": ["writer"]}])
    agents = {a.role: a for a in manager.agents.values()}
    assert "Verifica los casos límite." in agents["writer"].instruction
    assert "Verifica los casos límite." not in agents["researcher"].instruction
    assert summary == [{"name": "Revisor", "applied_to": 1}]


def test_apply_skills_all_when_no_targets():
    manager = _manager()
    apply_skills(manager, [{"name": "S", "text": "Sé conciso."}])
    assert all("Sé conciso." in a.instruction for a in manager.agents.values())


def test_apply_skills_ignores_empty_text():
    manager = _manager()
    summary = apply_skills(manager, [{"name": "Vacía", "text": "   "}, {"name": "Bien", "text": "ok"}])
    assert summary == [{"name": "Bien", "applied_to": len(manager.agents)}]
    assert not any("Vacía" in a.instruction for a in manager.agents.values())
