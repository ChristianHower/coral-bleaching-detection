# Agent Instructions

This file applies to any agent (Codex, Claude, or otherwise) working in this
repository. It covers writing style and UI/visual design — the things that
make output read or look obviously AI-generated. Follow it for code
comments, docstrings, commit messages, README/doc prose, UI copy, and CSS.

## Writing style

Write like a marine-science engineer documenting real work, not like
marketing copy. Concretely:

- No throat-clearing or hedge openers: "It's important to note that...",
  "In today's world...", "When it comes to...". Just say the thing.
- No rule-of-three padding ("robust, scalable, and reliable") unless all
  three words are actually doing separate work.
- Avoid corporate/AI buzzwords: "leverage", "utilize" (use "use"),
  "seamless", "game-changer", "dive into", "unlock", "empower",
  "cutting-edge", "streamline". Plain verbs over jargon.
- No emoji in code, commit messages, or docs unless the user explicitly
  asks for them.
- Vary sentence length. A paragraph of uniformly medium sentences is a
  tell. Short sentences land points. Longer ones carry detail.
- Don't summarize what you just did at the end of every doc section
  ("In summary, this module handles..."). Say it once, where it belongs.
- Cite real numbers and real limitations (as README.md and
  docs/build-1-status.md already do) instead of vague confidence
  ("this should work well", "highly accurate"). If something is
  unvalidated, say so plainly.
- Comments explain *why*, not *what*. If a comment restates the code in
  English, delete it.

## UI / visual design style

This is a scientific field tool for reef rangers, not a SaaS landing page.
Avoid the default "AI-generated web app" look:

- No purple-to-blue gradient hero sections. No generic centered
  headline + subheadline + 3 icon-cards layout.
- No glassmorphism, no drop-shadow-on-everything, no uniformly rounded
  corners applied by default — pick a value and use it because it fits,
  not because it's the framework default.
- Don't reach for emoji as icons in the UI. Use a real icon set or none.
- Color palette should come from the domain: reef/ocean tones, coral
  bleaching risk should read as an actual risk gradient (not a generic
  brand-blue-to-brand-purple scale). Pick colors deliberately and justify
  them in a comment if the choice isn't obvious.
- Default system font stack is fine; don't add "Inter" or "Poppins" just
  because that's what AI-generated sites default to. Pick a font because
  it reads well at the data density this UI needs.
- Whitespace and padding should vary with content density (a dense data
  table needs different spacing than a single map view) — don't apply one
  uniform "generous padding everywhere" pass.
- Copy in the UI itself (labels, empty states, error messages) should
  sound like it was written by someone who has actually looked at reef
  survey data — specific, terse, no filler.

## Project context

See `docs/superpowers/specs/` for the design spec and
`docs/superpowers/plans/` for implementation plans. `README.md` and
`docs/build-1-status.md` document what Build 1 actually does and its
known limitations — keep new work consistent with the honesty level of
those documents (state what's validated vs. synthetic/unproven).
