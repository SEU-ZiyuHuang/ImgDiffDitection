# Domain docs

How engineering skills should consume this repository's domain documentation.

## Before exploring

- Read root `CONTEXT.md` when it exists.
- Read the ADRs in `docs/adr/` that touch the area being changed.

If these files do not exist, proceed silently. Do not suggest creating them up front: `/domain-modeling`, `/grill-with-docs`, and `/improve-codebase-architecture` create them when terms or decisions need to be recorded.

## File structure

This is a single-context repository:

```
/
|- CONTEXT.md
|- docs/adr/
`- src/
```

## Use the glossary's vocabulary

When an issue title, refactor proposal, hypothesis, or test names a domain concept, use the term defined in `CONTEXT.md`. Do not drift to a synonym the glossary explicitly avoids.

If a needed concept is absent from the glossary, reconsider whether a project term already exists; otherwise note the gap for `/domain-modeling`.

## Flag ADR conflicts

If an output contradicts an existing ADR, surface the conflict explicitly instead of silently overriding it.
