# Delegation protocol (to save Claude usage)

Controller: Claude writes task files in tasks/. Workers (GLM-5.3 in zcode, Antigravity, local Qwen via `qwen start`) execute ONE task each.
(`opencode` GLM routes are dead: token expired / no funds / no channel. Use the zcode app.)

How to run a task (owner does this, 20 seconds):
1. Open zcode (or Antigravity) in this repo folder.
2. Paste exactly:  Read delegation/PREAMBLE.md and then do delegation/tasks/<TASK>.md. Obey every rule. Write results to delegation/results/.
3. When it finishes, tell Claude: "T0x done". Claude reads ONLY the first 15 lines of the result.

Claude's own usage rules from now on: no polling loops, no long explanations, one status call per request, short answers.
Zero-token monitoring (owner, any terminal):  watch -n 600 ./status.sh
Claude is needed only for: decisions, submissions logic, code that touches the training/inference pipeline, reviewing worker results.
