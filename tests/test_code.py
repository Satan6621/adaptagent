# -*- coding: utf-8 -*-
"""Fase 2 chunk 2 - sandboxed code-step as a first-class workflow node.

Espejo byte-exacto de test_retrieval.py: CodeSpec roundtrip + campos
defaults para no-LLM/sin-coste + dispatch del runner. El sandbox real
(adaptagent.tools.python_repl) ya esta probado por Fase 1; aqui se verifica
la integracion declarativa graph.py<->executor.py.
"""
from __future__ import annotations

from adaptagent.workflow.graph import CodeSpec, WorkflowStep


def test_code_spec_roundtrip() -> None:
    spec = CodeSpec(source="x = 1", timeout_s=10, language="python")
    data = spec.to_dict()
    again = CodeSpec.from_dict(data)
    assert again.to_dict() == data
    assert again.source == "x = 1"
    assert again.timeout_s == 10


def test_code_spec_defaults_are_no_llm() -> None:
    spec = CodeSpec()
    assert spec.source == ""
    assert spec.timeout_s == 20
    assert spec.language == "python"


def test_workflow_step_code_field_roundtrip() -> None:
    step = WorkflowStep(
        id="s1",
        name="run code",
        instruction="",
        code=CodeSpec(source="y = 2", timeout_s=5, language="python"),
    )
    data = step.to_dict()
    assert "code" in data
    again = WorkflowStep.from_dict(data)
    assert again.code is not None
    assert again.code.source == "y = 2"


def test_docs_to_code_default_none_is_grounded() -> None:
    step = WorkflowStep(id="s0", name="plain", instruction="")
    data = step.to_dict()
    assert "code" not in data
    again = WorkflowStep.from_dict(data)
    assert again.code is None
