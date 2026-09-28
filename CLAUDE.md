# CLAUDE.md

Technical notes for this repo: a Gradio GUI for [ChildNet](https://github.com/MartinPernus/ChildNet) kinship face synthesis.

- Run from the repo root: `python -m childnet_gui` (finds `./ChildNet`, or pass `--childnet-dir`). The package uses relative imports.
- `childnet_gui/engine.py` imports ChildNet from a checkout without modifying it. It registers ChildNet's `models` package explicitly (no `__init__.py`, generic name), stubs the e4e ops that JIT-compile CUDA at import, and re-implements `GeneModel.forward` with per-layer `move2parent`. Tests assert bitwise parity with ChildNet's own forward pass; keep it that way.
- Model construction patches `torch.load` (relative `checkpoints/` paths, `weights_only` fallback for files in the checkout) and skips random init and StyleGAN's 100k-sample W statistics, because ChildNet's strict state-dict loads overwrite everything.
- `childnet_gui/app.py`: live updates use `gr.on(..., trigger_mode="always_last")` plus a per-session `gr.State` memo that skips repeated requests (an example click fires two change events). In a multi-output update list, each `gr.update(...)` must be its own dict; Gradio consumes their keys.
- Gradio 5 takes `theme` in `gr.Blocks()`, Gradio 6 in `launch()`; `THEME_IN_LAUNCH` detects which. Tested with Gradio 5.50 and 6.28.
- Tests: `CHILDNET_DIR=ChildNet python -m pytest childnet_gui/tests`. They write ~4 GB of random-weight checkpoints (`childnet_gui/tests/fake_checkpoints.py`) to a temp dir, so no weights download is needed.
- `ChildNet_GUI.ipynb` (Colab) clones this repo at `GUI_BRANCH`, falling back to the default branch.
- `ffhq_align` in `childnet_gui/align.py` is CC BY-NC-SA 4.0 (adapted from NVIDIA's FFHQ scripts); the rest of the GUI is MIT.
