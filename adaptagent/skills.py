"""Personal skills: reusable instruction snippets attached to agents by role/name.

A skill is a named prompt fragment the user defines in the playground settings.
When a workflow runs, each skill with a matching `apply` list (roles or agent
names) augments the agent instruction.
"""

from __future__ import annotations

from typing import Any

# --- Methodology contracts (adapted from the Superpowers methodology) ---
# Injected into every agent when a workflow opts into a methodology, turning
# process rules (design-before-code, TDD, evidence over claims, root-cause
# debugging) into a mandatory contract for the whole execution.

METHODOLOGY_CONTRACTS: dict[str, str] = {
    "superpowers": """METODOLOGÍA ACTIVA — procesos obligatorios, no sugerencias:

1. Proceso antes que improvisación: antes de CADA tarea, comprueba si existe una habilidad/skill que aplique y síguela.
2. Diseño ANTES de implementar: nunca produzcas el entregable final sin que el diseño esté claramente definido y aprobado.
3. TDD cuando aplique: primero el test que falla, observa el fallo, después el código mínimo que lo haga pasar. No hay código de producción sin un test que haya fallado antes.
4. Evidencia sobre afirmaciones: no declares éxito sin verificarlo; verifica con pruebas ejecutables antes de completar.
5. Debug sistemático: ninguna corrección sin causa raíz investigada primero. Arreglar el síntoma es fracasar.
6. Simplicidad y foco: YAGNI (solo lo necesario) y DRY (sin duplicar). El cambio más pequeño que funcione.
7. Revisiones por severidad: reporta hallazgos como critical/major/minor; los críticos BLOQUEAN el avance.""",
}

# --- Reusable skill pack inspired by Superpowers (for the playground kit) ---
SUPERPOWERS_SKILLS: list[dict[str, Any]] = [
    {
        "name": "TDD (Red-Green-Refactor)",
        "apply": ["coder", "writer"],
        "text": "Escribe primero el test que falle; verifica que falla por la razón esperada; escribe el código mínimo que lo haga pasar; refactoriza solo tras estar en verde. Si no viste fallar el test, no sabes si el test sirve para algo.",
    },
    {
        "name": "Debugging sistemático",
        "apply": ["critic", "reviewer", "analyst"],
        "text": "Ninguna corrección sin investigar la causa raíz: lee el error completo y los stack traces, reproduce de forma consistente, revisa los cambios recientes y formula UNA hipótesis que pruebes con el cambio más pequeño posible. Arreglar el síntoma es fracaso.",
    },
    {
        "name": "Revisión por severidad",
        "apply": ["critic", "reviewer"],
        "text": "Revisa el trabajo contra el plan y los requisitos. Reporta cada hallazgo con severidad critical/major/minor y su localización exacta. Los issues críticos BLOQUEAN el avance hasta resolverse.",
    },
    {
        "name": "Verificación antes de completar",
        "apply": ["reviewer", "critic"],
        "text": "No declares éxito por afirmación: verifica con evidencia ejecutable (tests o ejecución real). Si no puedes verificar, dilo explícitamente y pide el método en lugar de asumir que funciona.",
    },
    {
        "name": "YAGNI + DRY",
        "apply": ["coder", "writer"],
        "text": "Implementa solo lo necesario (YAGNI), elimina duplicación (DRY) y prioriza simplicidad. El cambio más pequeño que funcione correctamente es el correcto.",
    },
    {
        "name": "Brainstorming previo",
        "apply": ["researcher", "analyst"],
        "text": "Antes de proponer soluciones, explora la intención con preguntas claras y plantea 2-3 enfoques alternativos con sus ventajas e inconvenientes; recomienda uno y espera validación antes de profundizar.",
    },
]


def apply_methodology(manager: Any, method: str) -> list[str]:
    """Inject a methodology contract into every agent's instruction.

    Returns the list of applied methods (empty when the method is unknown).
    """
    contract = METHODOLOGY_CONTRACTS.get((method or "").strip().lower())
    if not contract:
        return []
    marker = f"[Methodology: {method.strip().lower()}]"
    for agent in manager.agents.values():
        if marker not in agent.instruction:
            agent.instruction = f"{agent.instruction}\n\n{marker}\n{contract}".strip()
    return [method.strip().lower()]


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
