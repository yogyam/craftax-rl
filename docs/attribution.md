# Attribution and project provenance

Craftax is by Michael Matthews and collaborators. The game is installed as the
`craftax` dependency; its implementation and assets are not vendored here.
See the [official repository](https://github.com/MichaelTMatthews/Craftax) and
[ICML 2024 paper](https://proceedings.mlr.press/v235/matthews24a.html).

The reference PPO loop, baseline policy and baseline checkpoint were supplied
as starter code. This project extends them with configurable MLPs, warm starts,
checkpoint selection, survival shaping, reconstructed memory and late-state
resets. Existing AI-assistance notes are preserved in the source: Claude Code
assisted the original tooling and trainer extensions, which were reviewed by
Yogya Mehrotra. Codex assisted the standalone packaging and documentation.

The five recorded external benchmark means come from contemporaneous notes.
The local JSON results contain per-episode achievements, seeds and step counts.
No claim of superiority over published Craftax results is made: this project
uses a parameter variant and an arithmetic achievement-count metric.

No new blanket license is assigned to starter code or checkpoints in this
repository. Upstream dependencies retain their respective licenses.
