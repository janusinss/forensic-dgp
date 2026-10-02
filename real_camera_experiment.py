"""Static, training-only three-arm schedule. No model or optimizer imports."""
import copy
import random

ARMS = ("native83", "camera83", "camera91")
EPOCHS, STEPS, SEED = 2, 56, 42


def fixed_schedules(real_rows, fixture_rows):
    if len(real_rows) != 91 or any(r["split"] != "train" for r in real_rows):
        raise ValueError("Only the exact91-source training split is accepted")
    old = real_rows[:83]
    positive = [i for i, r in enumerate(old) if r["kind"] == "covered"]
    clear = [i for i, r in enumerate(old) if r["kind"] == "uncovered"]
    new_positive = [i for i in range(83, 91) if real_rows[i]["kind"] == "covered"]
    new_clear = [i for i in range(83, 91) if real_rows[i]["kind"] == "uncovered"]
    if (len(positive), len(clear), len(new_positive), len(new_clear)) != (51, 32, 7, 1):
        raise ValueError("Fixed old/new real membership differs")
    if len(fixture_rows) != 280 or [r["case_id"] for r in fixture_rows] != list(range(280)):
        raise ValueError("Fixed reflection fixture membership differs")
    fp = [r["case_id"] for r in fixture_rows if r["style"] != "clear"]
    fc = [r["case_id"] for r in fixture_rows if r["style"] == "clear"]
    if (len(fp), len(fc)) != (224, 56):
        raise ValueError("Fixed reflection strata differ")
    rng = random.Random(SEED)
    result = {arm: [] for arm in ARMS}
    for epoch in range(EPOCHS):
        a, b, c, d = map(list, (positive, clear, fp, fc))
        for pool in (a, b, c, d):
            rng.shuffle(pool)
        batches = [{"real": [{"index": a[2 * step % len(a)], "degraded": False},
                              {"index": a[(2 * step + 1) % len(a)], "degraded": False},
                              {"index": b[step % len(b)], "degraded": False}],
                    "fixture": c[4 * step:4 * step + 4] + [d[step]]} for step in range(STEPS)]
        result["native83"].append(copy.deepcopy(batches))
        for step, batch in enumerate(batches):
            for slot, record in enumerate(batch["real"]):
                record["degraded"] = bool((epoch + step + slot) % 2)
        result["camera83"].append(copy.deepcopy(batches))
        for step, batch in enumerate(batches):
            if step % 4 == 0:
                batch["real"][0] = {"index": new_positive[(step // 4) % 7],
                                     "degraded": bool((step // 4 + epoch) % 2)}
            if step in (0, 8, 16):
                batch["real"][2] = {"index": new_clear[0], "degraded": bool((step // 8 + epoch) % 2)}
        result["camera91"].append(copy.deepcopy(batches))
        seen = {r["index"] for batch in batches for r in batch["real"]}
        if seen != set(range(91)):
            raise ValueError("Extension sampling lost original/new real-source coverage")
    return result
