from collections import Counter
import unittest

from real_camera_experiment import ARMS, EPOCHS, STEPS, fixed_schedules


class CameraScheduleContract(unittest.TestCase):
    def setUp(self):
        self.rows = [{"split": "train", "kind": "covered" if i < 51 or i >= 84 else "uncovered"}
                     for i in range(91)]
        self.fixtures = [{"case_id": i, "style": "clear" if i % 5 == 0 else "reflection"} for i in range(280)]

    def test_isolated_camera_arm_replay_and_cohort_caps(self):
        result = fixed_schedules(self.rows, self.fixtures)
        self.assertEqual(result, fixed_schedules(self.rows, self.fixtures))
        self.assertEqual(tuple(result), ARMS)
        for epoch in range(EPOCHS):
            self.assertEqual(len(result["native83"][epoch]), STEPS)
            for a, b, c in zip(*(result[arm][epoch] for arm in ARMS)):
                self.assertEqual(a["fixture"], b["fixture"])
                self.assertEqual(a["fixture"], c["fixture"])
                self.assertEqual([r["index"] for r in a["real"]], [r["index"] for r in b["real"]])
                self.assertTrue(all(not r["degraded"] for r in a["real"]))
            self.assertEqual(Counter(i for batch in result["camera91"][epoch] for i in batch["fixture"]), Counter(range(280)))
            uses = Counter(r["index"] for batch in result["camera91"][epoch] for r in batch["real"])
            self.assertEqual(set(uses), set(range(91)))
            self.assertEqual(uses[83], 3)
            for i in range(84, 91):
                self.assertEqual(uses[i], 2)
                self.assertEqual({r["degraded"] for batch in result["camera91"][epoch] for r in batch["real"] if r["index"] == i}, {True, False})

    def test_held_out_membership_rejected_before_schedule(self):
        self.rows[40]["split"] = "validation"
        with self.assertRaises(ValueError):
            fixed_schedules(self.rows, self.fixtures)


if __name__ == "__main__":
    unittest.main()
