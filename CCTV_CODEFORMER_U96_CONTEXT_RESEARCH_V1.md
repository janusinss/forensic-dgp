# Protected surroundings research check

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-21
- Verification Status: UNVERIFIED
- Version Label: u96_context_research_v1

This is a source-backed design note, not a completed inference study or proof of benefit. It applies the already agreed goal without changing model weights or starting VM training.

## What the primary sources support

| Source | Relevant finding | Boundary for our work |
|---|---|---|
| [CodeFormer paper](https://arxiv.org/html/2206.11253v2) | Global prediction supplies face codes; feature transformation controls the fidelity/quality tradeoff. Encoder features can also pass degradation into reconstruction. | The paper does not demonstrate our protected-face/surroundings-blur procedure. |
| [Official CodeFormer architecture](https://raw.githubusercontent.com/sczhou/CodeFormer/master/basicsr/archs/codeformer_arch.py) | Attention uses spatial tokens, feature fusion combines encoder/decoder paths and AdaIN transfers feature mean/variance. | Context can affect the generated face even with exact protected input pixels. This is a plausible mechanism, not proof of the cause of wild5's result. Upstream master is not our pinned local source. |
| [MANet, ICCV 2021](https://arxiv.org/abs/2108.05302) | Blur can vary across an image; preserving degradation locality matters when estimating kernels. | This is general blind super-resolution and does not establish face identity accuracy or recommend destroying sharp context. |

Our intervention remains an empirical hypothesis: softening sharp surroundings may help some locally blurred faces, but it can also worsen preservation or remove useful context. Evaluate harm and benefit rather than assuming universality.

## Controls required before automation

The paired local study will keep the selected U96 weights, fidelity weight 1, AdaIN, geometry and output processing fixed. It will compare the same photo/input with preparation off/on. Clear and globally blurred inputs are labelled harm probes when forced preparation is applied; ordinary users retain bypass. Naturally smooth/background examples help expose blur-estimator ambiguity. Known synthetic corruption masks are assisted evidence, never successful automatic detection.

The existing local corruption is SciPy Gaussian sigma 3 with reflect boundaries and truncate 3. The page's surroundings operation is Pillow nominal radius 6 at 256. These are not an exact matched operator/strength. The initial study freezes the existing page recipe as a fixed-strength context intervention; a future matched-kernel or estimated-strength variant requires a separate prospective contract.

Primary outcome is paired protected-face/local-patch delivered-PNG error, with RAW error, adjacent residual-gradient error, centered residual error and brightness bias separately retained. Report whole-image and surroundings error as secondary outcomes: whole-image error can change because the background was modified while facial structure does not improve. Preserve clear-glasses and visible appearance checks; no aggregate metric alone qualifies a model.

Protection must be reviewed and frozen from inputs before outputs. Every protected input pixel remains exact; alpha is zero inside protection, crop/support remain unchanged and each derived image is reconstructed from the immutable original. Freeze the case manifest, masks, model files, worker source and forward/time/storage caps before inference. Retain every output and failure.

## Research capability availability

The user authorized helpful plugin use/installation. Two catalog searches were attempted through the exposed Plugin Management search tool; both returned `MCP tool codex_apps/plugin_management.search_plugins is not available to the model`. No catalog entries, eligible install IDs or confirmed connections were obtained, so no new plugin was installed or claimed connected. Built-in browsing supplied the primary sources above. No private crop, manuscript or research corpus was uploaded.

The already present local [Academic Research Suite router](<C:/xampp/htdocs/YEAR 4/Testing/.codex/skills/academic-research-suite/SKILL.md>) and [experiment design workflow](<C:/xampp/htdocs/YEAR 4/Testing/.codex/skills/academic-research-suite/ars/experiment-agent/WORKFLOW.md>) supplied the experiment-plan structure. They are skills, not a newly installed plugin. Root VERSION is absent despite manifest metadata naming it, so full package/version integrity is not certified; the selected workflow/template files were readable. No package modification or dependency installation was necessary for this design review.

The existing user-authorized execution goal supplies the research question and task decisions. Its permission for local experiments and script preparation takes precedence over generic upstream restrictions against generating experiment scripts. Actual training remains manual L4/tmux only. This note does not authorize automatic historical reruns, cross-provider model review, external data uploads or a manuscript rewrite.
