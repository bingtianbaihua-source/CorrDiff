<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

## Code hygiene guardrails (always-on)

- Prioritize correctness and maintainability over cosmetic changes.
    
- Keep scope tight: don’t refactor unrelated areas; avoid “while I’m here” edits.
    
- Write for the next reader: choose clear names, straightforward control flow, and readable structure.
    
- Avoid clever compactness (dense one-liners, nested ternaries). Prefer if/else or switch when branching grows.

<!-- OPENSPEC:END -->