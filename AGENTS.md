# Repository maintenance

## Recorded trial provenance

Sam Zaheer has confirmed that the Claude Opus 5.5 (`max`) and GPT-6.1 Sol
(`xhigh`) evaluations used Harbor on a temporary VM, with the Claude Code
(`claude-code`) harness for both models. This includes the adversarial attempts;
the Sol adversarial attempt was refused before any agent action and was not graded.

Keep the overview, task notes, trial READMEs, run summaries, verifier notes and
reproduction launcher consistent with that provenance. Imported event-field names
and unavailable original VM trial directories do not establish a different harness
or execution environment. Do not infer a local CLI run from either of them.

Retain measured results, submissions, timestamps, usage counts and provider
responses when correcting descriptive metadata. Do not fabricate original Harbor
logs, trial identifiers or transcripts. A run summary here is an imported summary,
not a replacement for the original VM job directory.

## Publishing updates

Preserve Sam Zaheer's authorship and the requested single submission commit,
`Prepare house thermal model task for submission`. Before replacing remote history,
use an explicit lease check and preserve any concurrent remote edits.
