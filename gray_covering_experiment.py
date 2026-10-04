"""Matched RGB/grayscale exposure pilot; no models or optimizers on import."""
import copy
import random

from varied_covering_experiment import FAMILIES, SOURCE_SHA, family, preflight_case, require, validate_source

ARMS = ("rgb133", "gray133")
EPOCHS, STEPS, SEED = 8, 64, 42
UPDATES = EPOCHS * STEPS
MEASUREMENT_IMAGES = 812
MEASUREMENT_CHECKPOINTS = 4  # initial, RGB128 reproduction, RGB512, gray512
FORWARD_BUDGET = 2 * UPDATES * 8 + MEASUREMENT_CHECKPOINTS * MEASUREMENT_IMAGES


def fixed_schedules(real_rows, fixture_rows):
    require(len(real_rows) == 133 and all(r["split"] == "train" for r in real_rows),
            "Require the same133 author-train sources; no held-out inputs")
    old_positive = [i for i, r in enumerate(real_rows[:91]) if r["kind"] == "covered"]
    old_clear = [i for i, r in enumerate(real_rows[:91]) if r["kind"] == "uncovered"]
    new_clear = [i for i in range(91, 133) if real_rows[i]["kind"] == "uncovered"]
    new_positive = {name: [i for i in range(91, 133) if real_rows[i]["kind"] == "covered"
                           and family(real_rows[i]) == name] for name in FAMILIES}
    require((len(old_positive), len(old_clear), len(new_clear)) == (58, 33, 11)
            and [len(new_positive[name]) for name in FAMILIES] == [9, 7, 6, 4, 5], "Reviewed strata differ")
    require(all(r["data_origin"] == "cofw_supported_extension" and r["publisher_split"] == "train"
                for r in real_rows[91:]), "Keep original author training membership")
    require(len(fixture_rows) == 280 and [r["case_id"] for r in fixture_rows] == list(range(280)), "Fixture cohort differs")
    fp = [r["case_id"] for r in fixture_rows if r["style"] != "clear"]
    fc = [r["case_id"] for r in fixture_rows if r["style"] == "clear"]
    require((len(fp), len(fc)) == (224, 56), "Fixture covered/clear strata differ")
    rng = random.Random(SEED)
    for pool in (old_positive, old_clear, new_clear, fp, fc, *(new_positive[name] for name in FAMILIES)):
        rng.shuffle(pool)
    occurrences = {role: {} for role in ("old_positive", "new_positive", "old_clear", "new_clear")}
    result = {arm: [] for arm in ARMS}
    for epoch in range(EPOCHS):
        batches = {arm: [] for arm in ARMS}
        for step in range(STEPS):
            index = epoch * STEPS + step
            name = FAMILIES[index % len(FAMILIES)]
            selected = [old_positive[index % len(old_positive)],
                        new_positive[name][(index // len(FAMILIES)) % len(new_positive[name])],
                        old_clear[index % len(old_clear)], new_clear[index % len(new_clear)]]
            views = []
            for role, source in zip(occurrences, selected):
                visit = occurrences[role].get(source, 0)
                occurrences[role][source] = visit + 1
                views.append({"index": source, "degraded": bool(visit % 2),
                              "grayscale": bool((visit // 2) % 2)})
            fixture = [fp[2 * index % len(fp)], fp[(2 * index + 1) % len(fp)],
                       fc[2 * index % len(fc)], fc[(2 * index + 1) % len(fc)]]
            batches["gray133"].append({"real": views, "fixture": list(fixture)})
            batches["rgb133"].append({"real": [{**v, "grayscale": False} for v in views], "fixture": list(fixture)})
        for arm in ARMS:
            result[arm].append(copy.deepcopy(batches[arm]))
    for arm in ARMS:
        seen = {(r["index"], r["degraded"], r["grayscale"])
                for epoch in result[arm] for batch in epoch for r in batch["real"]}
        gray_conditions = (False, True) if arm == "gray133" else (False,)
        require(seen == {(i, d, g) for i in range(133) for d in (False, True) for g in gray_conditions},
                "Every training source needs all declared input conditions")
        require({i for epoch in result[arm] for batch in epoch for i in batch["fixture"]} == set(range(280)),
                "Every existing replay fixture must occur")
    return result


def grayscale_rgb(tensor):
    """Change input channels only, preserving crop, targets and supervised support."""
    import torch
    require(tensor.ndim == 3 and tensor.shape[0] == 3 and tensor.dtype == torch.float32
            and torch.isfinite(tensor).all() and tensor.min() >= 0 and tensor.max() <= 1,
            "Require finite float RGB in [0,1]")
    coefficients = tensor.new_tensor([.299, .587, .114])[:, None, None]
    return (tensor * coefficients).sum(dim=0, keepdim=True).expand_as(tensor).contiguous()


def prepare_view(view, grayscale=False):
    x, target, valid = view
    require(type(grayscale) is bool, "Grayscale condition must be boolean")
    return (grayscale_rgb(x) if grayscale else x), target, valid


def fit_decisions(final):
    control, treatment = (final[arm]["groups"] for arm in ARMS)

    def retained(a, b):
        return a["iou"] >= b["iou"] and all(a[k] <= b[k] for k in
                   ("missed_fraction", "visible_false_positive", "empty_mask_cases", "negative_false_positive_cases"))

    rgb_groups = ["old_native", "old_degraded", "reflection"] + [
        f"cofw_{name}_{condition}" for name in FAMILIES for condition in ("native", "degraded")]
    checks = {
        "grayscale_hair_native_and_degraded_iou_gain": all(
            treatment[f"cofw_hair_{condition}_grayscale"]["iou"] > control[f"cofw_hair_{condition}_grayscale"]["iou"]
            for condition in ("native", "degraded")),
        "other_new_grayscale_family_iou_retained": all(
            treatment[f"cofw_{name}_{condition}_grayscale"]["iou"] >= control[f"cofw_{name}_{condition}_grayscale"]["iou"]
            for name in FAMILIES if name != "hair" for condition in ("native", "degraded")),
        "rgb_cohorts_and_reflection_fit_retained": all(retained(treatment[group], control[group]) for group in rgb_groups),
        "all_final_supervised_clear_controls_empty": all(group["negative_false_positive_cases"] == 0
            for arm in final.values() for group in arm["groups"].values()),
        "new_covered_predictions_nonempty_all_conditions": all(
            treatment[f"cofw_{name}_{condition}{suffix}"]["empty_mask_cases"] == 0
            for name in FAMILIES for condition in ("native", "degraded") for suffix in ("", "_grayscale")),
    }
    return {"checks": checks, "all_training_fit_checks_pass": all(checks.values()), "training_fit_only": True,
            "original_425_case_gates_evaluated": False, "historical_gate_failure_unchanged": True,
            "selected_checkpoint": None, "promoted": False,
            "next": "Independently audit the return and evaluate unchanged practical sources/guard and face outputs before model selection"}
