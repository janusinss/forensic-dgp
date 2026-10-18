"""Render the user-requested cascade comparison using the existing gallery UI."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "outputs/cctv_dgp_checkpoint_comparison_v1"
OUT = PARENT / "cascade_v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render():
    p = read(OUT / "plan.json")
    m = read(OUT / "metrics.json")
    assert read(OUT / "receipt.json")["complete"]
    cases = []
    for item in p["cases"]:
        c = dict(item)
        for key in ["input", "observed", "target"]:
            if key in c:
                c[key] = "../" + c[key]
        cases.append(c)
    arms = [
        {"id": "resize_control", "short": "Input / resize", "execution_id": "resize_control", "result_dir": "../results/resize_control", "checkpoint": "Exact common prepared input"},
        {"id": "dgp_zamboanga_final", "short": "Phase3 DGP alone", "execution_id": "dgp_zamboanga_final", "result_dir": "../results/dgp_zamboanga_final", "checkpoint": "checkpoints/dgp_zamboanga_final.pth"},
        {"id": "codeformer_w1", "short": "CodeFormer alone", "execution_id": "codeformer_w1", "result_dir": "../results/codeformer_w1", "checkpoint": "Original input → CodeFormer w=1"},
        {"id": "phase3_then_codeformer", "short": "Phase3 → CodeFormer", "execution_id": "phase3_then_codeformer", "result_dir": "results/phase3_then_codeformer", "checkpoint": "Saved Phase3 native PNG → CodeFormer w=1"},
    ]
    page = (ROOT / "restoration_checkpoint_comparison_v1.template.html").read_text(encoding="utf-8")
    replacements = [
        ("Face restoration checkpoint comparison", "Phase3 DGP to CodeFormer comparison"),
        ("Which restoration checkpoint<br>should we improve?", "DGP → CodeFormer:<br>what changes?"),
        ("Compare every retained DGP checkpoint in <strong>checkpoints/</strong>, the current research checkpoint and two pretrained restoration baselines. Every result starts from the same 256 × 256 input.",
         "Compare the original input, Phase3 DGP alone, CodeFormer alone and <strong>Phase3 DGP → CodeFormer</strong>. The second stage receives the exact saved Phase3 PNG. CodeFormer fidelity stays at 1.0."),
        ("<b>12</b>checkpoint states", "<b>3</b>restoration paths"),
        ("Restoration only. Completion and mask detection are bypassed. The app and its default checkpoint are unchanged. Checkpoint names preserve their filenames; epoch numbers do not establish which model is best. Mapped starting weights run in our decoder graph; this arm does not reproduce the publisher’s full DeblurGAN pipeline.",
         "Restoration only. Both processing orders use the same CodeFormer weights and settings. The Phase3 stage is reused from its verified saved output. No extra alignment or sharpening is applied. Model selection remains yours."),
        ("Checkpoint A", "Processing path A"), ("Checkpoint B", "Processing path B"),
        ("Model A<select", "Path A<select"), ("Model B<select", "Path B<select"),
        ("<th scope=\"col\">Checkpoint</th>", "<th scope=\"col\">Processing path</th>"),
        ("All checkpoints on this crop", "All processing paths on this crop"),
        ("Rows are sorted by paired SSIM. This is a development comparison, not a native CCTV accuracy ranking. CPU timings include each model’s declared internal processing; CodeFormer uses 512 × 512 internally and returns 256 × 256.",
         "Rows are sorted by paired SSIM. Compare the actual faces as well as the measurements. Native CCTV has no aligned clean target. Cascade timing combines the retained Phase3 forward time with the new CodeFormer forward time; local CPU timing is approximate."),
        ("3. Decide which model deserves a training pilot", "3. Review what the extra stage changes"),
        ("Choose a checkpoint that preserves the visible face and gives useful results across both CCTV sources. A candidate still needs a justified finite VM pilot, preservation checks and a separate final evaluation before it becomes the app’s primary restorer.",
         "Check whether the extra stage reduces artifacts, preserves glasses and hair, keeps the same expression and retains useful facial structure. This test compares processing orders. It does not select a model or change the application."),
        ("Original camera files", "Original camera files"),
        ("source_terms/", "../source_terms/"),
        ("href=\"plan.json\">Frozen inputs and checkpoints", "href=\"plan.json\">Frozen cascade protocol"),
        ("href=\"saved_output_audit.json\">Saved-output audit", "href=\"independent_saved_output_audit_v1.json\">Independent saved-output audit"),
        ("return 'results/'+arm.execution_id+'/'+cid+'/'+view.value+'.png'", "return arm.result_dir+'/'+cid+'/'+view.value+'.png'"),
        ("source.value='ChokePoint';aSelect.value='current_research_dgp';bSelect.value='dgp_zamboanga_final';",
         "source.value='dataset/thumbnails128x128';aSelect.value='codeformer_w1';bSelect.value='phase3_then_codeformer';"),
        ("populateCases('choke_dev_01_t067');measurements();", "populateCases('va_ffhq_37228_compound_lr24');measurements();"),
    ]
    for old, new in replacements:
        if old == new:
            continue
        assert old in page, old
        page = page.replace(old, new)
    payload = {"cases": cases, "arms": arms, "groups": m["groups"]}
    page = page.replace("__COMPARISON_DATA__", json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c"))
    assert "__COMPARISON_DATA__" not in page
    path = OUT / "comparison.html"
    assert not path.exists()
    path.write_text(page, encoding="utf-8")
    print(json.dumps({"complete": True, "gallery": str(path), "processing_paths_including_input": 4}))


if __name__ == "__main__":
    render()
