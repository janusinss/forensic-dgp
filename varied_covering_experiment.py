"""Finite matched varied-covering experiment; no models or optimizers on import."""
import copy
import random

ARMS = ("existing91", "varied133")
EPOCHS, STEPS, SEED = 2, 64, 42
SOURCE_SHA = "e4b16da0ccb92b2a3a6b29f10b863320ccf3ead1fb9e1464f753a667f76c1702"
SOURCE_PROTOCOL_SHA = "a8adb21910c5aed249f5f2a75e17eef92db35717bd428326c360a1aecf3b99e1"
FAMILIES = ("hand", "hair", "cloth", "object", "eyewear_mask")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def family(row):
    value = row.get("occlusion_stratum")
    mapping = {"hand": "hand", "obstructing_hair": "hair", "cloth_or_scarf": "cloth",
               "other_cloth_or_object": "cloth", "object": "object", "sunglasses": "eyewear_mask",
               "costume_mask": "eyewear_mask", "face_mask_preserve_clear_goggles": "eyewear_mask"}
    require(value in mapping, "Unknown new covering family")
    return mapping[value]


def preflight_case(cases):
    selected = [case for case in cases if case["source_id"] == "cofw_train_0868" and case["condition"] == "native"]
    require(len(selected) == 1 and selected[0]["split"] == "train"
            and selected[0]["family"] == "obstructing_hair", "Unique reviewed native hair preflight required")
    return selected[0]


def validate_source(payload):
    expected = {"format": "dgp-face-occlusion-adapter-v1", "target": "covered_region_is_one",
                "architecture": "resnet18-unet", "smp_version": "0.5.0", "initialization": "pretrained",
                "real_camera_arm": "camera91", "real_camera_protocol_sha256": SOURCE_PROTOCOL_SHA,
                "epoch": 44, "additional_epoch": 34, "optimizer_updates": 994,
                "fresh_optimizer_updates": 784, "experiment_updates": 112}
    require(all(payload.get(k) == v for k, v in expected.items()), "Fixed camera91 model lineage/counters differ")
    require(payload["selection"].get("selected") is False, "Source selection history differs")


def fixed_schedules(real_rows, fixture_rows):
    require(len(real_rows) == 133 and all(r["split"] == "train" for r in real_rows),
            "Require exact133 training rows; never held-out data")
    old_positive = [i for i, r in enumerate(real_rows[:91]) if r["kind"] == "covered"]
    old_clear = [i for i, r in enumerate(real_rows[:91]) if r["kind"] == "uncovered"]
    new_clear = [i for i in range(91, 133) if real_rows[i]["kind"] == "uncovered"]
    new_positive = {name: [i for i in range(91, 133) if real_rows[i]["kind"] == "covered"
                           and family(real_rows[i]) == name] for name in FAMILIES}
    require((len(old_positive), len(old_clear), len(new_clear)) == (58, 33, 11)
            and [len(new_positive[name]) for name in FAMILIES] == [9, 7, 6, 4, 5], "Fixed real strata differ")
    require(all(r["data_origin"] == "cofw_supported_extension" and r["publisher_split"] == "train"
                for r in real_rows[91:]), "New sources must retain author training membership")
    require(len(fixture_rows) == 280 and [r["case_id"] for r in fixture_rows] == list(range(280)), "Fixed fixture cohort differs")
    fp = [r["case_id"] for r in fixture_rows if r["style"] != "clear"]
    fc = [r["case_id"] for r in fixture_rows if r["style"] == "clear"]
    require((len(fp), len(fc)) == (224, 56), "Fixed fixture strata differ")
    rng = random.Random(SEED)
    for pool in (old_positive, old_clear, new_clear, fp, fc, *(new_positive[name] for name in FAMILIES)):
        rng.shuffle(pool)
    occurrences = {role: {} for role in ("old_positive", "old_clear", "new_positive", "new_clear")}

    def condition(role, index):
        count = occurrences[role].get(index, 0)
        occurrences[role][index] = count + 1
        return bool(count % 2)

    result = {arm: [] for arm in ARMS}
    for epoch in range(EPOCHS):
        batches = {arm: [] for arm in ARMS}
        for step in range(STEPS):
            index = epoch * STEPS + step
            name = FAMILIES[index % len(FAMILIES)]
            new_pos = new_positive[name][(index // len(FAMILIES)) % len(new_positive[name])]
            new_neg = new_clear[index % len(new_clear)]
            old_pos, old_neg = old_positive[index % len(old_positive)], old_clear[index % len(old_clear)]
            conditions = [condition("old_positive", old_pos), condition("new_positive", new_pos),
                          condition("old_clear", old_neg), condition("new_clear", new_neg)]
            control = [old_pos, old_positive[(index + 29) % len(old_positive)],
                       old_neg, old_clear[(index + 16) % len(old_clear)]]
            treatment = [old_pos, new_pos, old_neg, new_neg]
            fixture = [fp[2 * index % len(fp)], fp[(2 * index + 1) % len(fp)],
                       fc[2 * index % len(fc)], fc[(2 * index + 1) % len(fc)]]
            for arm, selected in zip(ARMS, (control, treatment)):
                batches[arm].append({"real": [{"index": i, "degraded": d} for i, d in zip(selected, conditions)],
                                     "fixture": list(fixture)})
        for arm in ARMS:
            result[arm].append(copy.deepcopy(batches[arm]))
    for arm, size in zip(ARMS, (91, 133)):
        seen = {(row["index"], row["degraded"]) for epoch in result[arm] for batch in epoch for row in batch["real"]}
        require(seen == {(i, condition) for i in range(size) for condition in (False, True)},
                "Every arm source must appear native and degraded within the fixed budget")
        require({i for epoch in result[arm] for batch in epoch for i in batch["fixture"]} == set(range(280)),
                "All training fixtures must appear")
    return result


def fit_decisions(final):
    control, treatment = (final[arm]["groups"] for arm in ARMS)

    def retained(a, b):
        return a["iou"] >= b["iou"] and all(a[k] <= b[k] for k in
                   ("missed_fraction", "visible_false_positive", "empty_mask_cases", "negative_false_positive_cases"))

    checks = {
        "all_new_families_native_and_degraded_iou_gain": all(treatment[f"cofw_{name}_{condition}"]["iou"] >
            control[f"cofw_{name}_{condition}"]["iou"] for name in FAMILIES for condition in ("native", "degraded")),
        "old_native_and_degraded_fit_retained": all(retained(treatment[group], control[group])
                                                    for group in ("old_native", "old_degraded")),
        "reflection_fit_retained": retained(treatment["reflection"], control["reflection"]),
        "all_final_clear_controls_empty": all(group["negative_false_positive_cases"] == 0
                                               for arm in final.values() for group in arm["groups"].values()),
        "new_covered_masks_nonempty_native_and_degraded": all(treatment[f"cofw_{name}_{condition}"]["empty_mask_cases"] == 0
            for name in FAMILIES for condition in ("native", "degraded")),
    }
    return {"checks": checks, "all_training_fit_checks_pass": all(checks.values()), "training_fit_only": True,
            "original_425_case_gates_evaluated": False, "historical_gate_failure_unchanged": True,
            "selected_checkpoint": None, "promoted": False,
            "next": "Audit returned states/masks and full preview, then decide original-gate and practical-output evaluation; do not promote from training fit"}
