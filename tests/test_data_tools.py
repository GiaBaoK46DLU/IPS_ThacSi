"""Shared preparation remains usable without private docs/references folders."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ips.cli import main
from ips.data import ROOT, OUTPUT


class DataToolsTests(unittest.TestCase):
    def test_prepare_selects_shared_tools_and_preserves_existing_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for existing, verify_only, expected in [
                (False, False, ["inspect_data.py", "make_splits.py", "verify_splits.py"]),
                (True, False, ["inspect_data.py", "verify_splits.py"]),
                (True, True, ["verify_splits.py"]),
            ]:
                with self.subTest(existing=existing, verify_only=verify_only):
                    if existing:
                        (output / "manifest.json").write_text("preserve-me")
                    with patch("ips.cli.OUTPUT", output), patch("ips.cli.subprocess.run") as run:
                        main(["prepare"] + (["--verify-only"] if verify_only else []))
                    paths = [Path(c.args[0][1]) for c in run.call_args_list]
                    self.assertEqual([p.name for p in paths], expected)
                    self.assertTrue(all(p.parent == ROOT / "scripts/data" and p.is_file() for p in paths))
                    if existing:
                        self.assertEqual((output / "manifest.json").read_text(), "preserve-me")

    def test_shared_policy_matches_frozen_split_and_has_no_personal_models(self):
        policy = json.loads((ROOT / "scripts/data/split_policy.json").read_text())
        manifest = json.loads((OUTPUT / "manifest.json").read_text())
        classes = json.loads((OUTPUT / "class_mapping.json").read_text())
        self.assertEqual({k: v for k, v in policy["split"].items() if k != "state"}, manifest["split_settings"])
        self.assertEqual(policy["class_pairs"], [[c["building"], c["floor"]] for c in classes])
        self.assertEqual(policy["development_file"], manifest["sources"]["development"]["path"])
        self.assertEqual(policy["final_test_file"], manifest["sources"]["final_test"]["path"])
        self.assertGreater(policy["preprocessing"]["coordinate_scale_minimum"], 0)
        self.assertEqual(set(policy), {"development_file", "final_test_file", "class_pairs", "split", "preprocessing"})
