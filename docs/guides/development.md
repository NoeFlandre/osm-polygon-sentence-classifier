# Development

Keep modules small and single-purpose. A change should have one clear
responsibility and should preserve the managed-data and remote-operation
boundaries.

The pinned dataset contract, lazy loader, review-only audit, typed training
boundary, and autonomous Grid'5000 operator are explicit surfaces. Remote
publication is limited to the configured final model and Trackio destinations;
model evaluation policy remains deferred until its contract is reviewed. Do
not introduce that deferred surface as an incidental helper in training, audit,
or operator changes.

## RED -> GREEN -> REFACTOR

Use this five-step workflow for behavior changes:

1. **RED:** write one focused unit or contract test describing the required
   behavior.
2. **Confirm RED:** run the smallest relevant test command and record the
   expected failure; fix test errors until the failure is about the missing
   behavior.
3. **GREEN:** implement the smallest change that makes the test pass.
4. **REFACTOR:** simplify names, boundaries, and duplication while keeping the
   tests green and adding no unrequested behavior.
5. **Verify:** run the complete quality, documentation, and repository-hygiene
   checks before handoff.

## Tool roles

- **uv** provides the locked environment and reproducible command entry point.
- **Ruff** formats Python and reports lint and complexity issues.
- **ty** checks the declared Python types.
- The package publishes a PEP 561 `py.typed` marker for downstream type checkers.
- **pytest** runs unit and contract tests.
- **Just** gives the quality and documentation checks stable, short names.

The pre-commit hooks and GitHub Actions use the same locked commands so local
and CI evidence remains comparable.
