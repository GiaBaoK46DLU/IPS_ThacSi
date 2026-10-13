"""EFIM is never trusted on its own output: every result is compared with a brute-force oracle
that applies the HUI definition with no pruning, plus utilities computed by hand."""
import random
import unittest
from itertools import combinations

from ips.algorithms.huim.efim import EFIM, ResourceLimitExceeded, mine_efim

# Running example of the HUIM literature (Zida et al. 2017; Liu & Qu 2012).
RUNNING_DB = [{"a": 1, "c": 1, "d": 1},
              {"a": 2, "c": 6, "e": 2, "g": 5},
              {"a": 1, "b": 2, "c": 1, "d": 6, "e": 1, "f": 5},
              {"b": 4, "c": 3, "d": 3, "e": 1},
              {"b": 2, "c": 2, "e": 1, "g": 2}]
RUNNING_P = {"a": 5, "b": 2, "c": 1, "d": 2, "e": 3, "f": 1, "g": 1}


def utility(itemset, database, p):
    return sum(sum(t[i] * p[i] for i in itemset) for t in database if all(i in t for i in itemset))


def support(itemset, database):
    return sum(all(i in t for i in itemset) for t in database)


def brute_force(database, p, min_util):
    items = sorted({i for t in database for i in t}, key=repr)
    return {frozenset(combo): (round(utility(combo, database, p), 9), support(combo, database))
            for k in range(1, len(items) + 1) for combo in combinations(items, k)
            if utility(combo, database, p) >= min_util}


def as_map(huis):
    return {frozenset(x): (round(float(u), 9), s) for x, u, s in huis}


class EFIMTests(unittest.TestCase):
    def test_running_example_utilities_by_hand(self):
        # u({b,d}) = (2*2 + 6*2) + (4*2 + 3*2) = 16 + 14 = 30; u({b}) = 4 + 8 + 4 = 16
        self.assertEqual(utility(("b", "d"), RUNNING_DB, RUNNING_P), 30)
        self.assertEqual(utility(("b",), RUNNING_DB, RUNNING_P), 16)
        huis = as_map(EFIM(min_util=30).mine(RUNNING_DB, RUNNING_P))
        self.assertEqual(huis[frozenset("bd")], (30.0, 2))
        self.assertNotIn(frozenset("b"), huis)        # utility is not anti-monotone

    def test_matches_brute_force_on_running_example(self):
        for min_util in (1, 10, 20, 25, 30, 40, 60, 100):
            with self.subTest(min_util=min_util):
                huis, stats = mine_efim(RUNNING_DB, RUNNING_P, min_util)
                self.assertEqual(stats.status, "COMPLETED")
                self.assertEqual(as_map(huis), brute_force(RUNNING_DB, RUNNING_P, min_util))

    def test_matches_brute_force_on_random_databases(self):
        rng = random.Random(20251007)
        for case in range(40):
            items = [f"i{k}" for k in range(rng.randint(2, 8))]
            p = {i: rng.choice([0.5, 1, 1.5, 2, 3, 7]) for i in items}
            database = [{i: rng.randint(1, 9) for i in rng.sample(items, rng.randint(1, len(items)))}
                        for _ in range(rng.randint(1, 12))]
            total = sum(q * p[i] for t in database for i, q in t.items())
            min_util = total * rng.choice([0.01, 0.05, 0.1, 0.2, 0.4])
            with self.subTest(case=case):
                self.assertEqual(as_map(mine_efim(database, p, min_util)[0]), brute_force(database, p, min_util))

    def test_budget_abort_is_controlled_and_keeps_partial_results(self):
        huis, stats = mine_efim(RUNNING_DB, RUNNING_P, 1, max_huis=3)
        self.assertEqual(stats.status, "ABORTED_RESOURCE_LIMIT")
        self.assertEqual(len(huis), 4)
        with self.assertRaises(ResourceLimitExceeded):
            EFIM(min_util=1, max_huis=3).mine(RUNNING_DB, RUNNING_P)

    def test_nonpositive_utilities_are_rejected(self):
        with self.assertRaises(ValueError):
            mine_efim(RUNNING_DB, {**RUNNING_P, "a": 0}, 10)


if __name__ == "__main__":
    unittest.main()
