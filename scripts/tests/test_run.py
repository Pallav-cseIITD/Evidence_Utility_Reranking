"""Check command routing and failure handling without training or generation."""

from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from scripts.run import ROOT, absolute, command_plan, execute, parser


class CommandTests(unittest.TestCase):
    def plan(self, *arguments):
        return command_plan(parser().parse_args(arguments))

    def test_training_uses_the_central_configuration(self):
        command, = self.plan("train")
        self.assertIn("method.train", command)
        self.assertEqual(command[command.index("--manifest") + 1], str(ROOT / "configs/hotpot_repair.json"))
        self.assertEqual(command[-2:], [str(ROOT / "data/tokenizers/qwen25.json"),
                                       str(ROOT / "data/tokenizers/qwen3.json")])

    def test_custom_training_configuration_and_output(self):
        command, = self.plan("train", "--config", "configs/alternate.json", "--output", "outputs/another_model")
        self.assertIn(str(ROOT / "configs/alternate.json"), command)
        self.assertIn(str(ROOT / "outputs/another_model"), command)

    def test_selection_verifies_inputs_before_running(self):
        verify, select = self.plan("select")
        self.assertIn("verify", verify)
        self.assertIn("method.select", select)
        self.assertIn("--batch", select)

    def test_reproduction_compares_with_the_saved_selections(self):
        _, command = self.plan("reproduce")
        self.assertIn("method.check_reproduction", command)
        self.assertIn(str(ROOT / "data/prepared/hotpot_eval/selections.json"), command)

    def test_custom_data_root_is_used_for_every_input(self):
        verify, command = self.plan("reproduce", "--data-dir", "other_data")
        self.assertIn(str(ROOT / "other_data"), verify)
        self.assertIn(str(ROOT / "other_data/prepared/hotpot_eval/pools.json"), command)
        self.assertIn(str(ROOT / "other_data/tokenizers/qwen25.json"), command)

    def test_restore_uses_the_selected_local_archive_folder(self):
        command, = self.plan("restore", "--archives-dir", "local_assets", "--groups", "alce_transfer", "tokenizers")
        self.assertIn(str(ROOT / "local_assets"), command)
        self.assertIn("alce_transfer", command)

    def test_restore_can_use_a_published_release(self):
        command, = self.plan("restore", "--base-url", "https://example.org/data")
        self.assertEqual(command[-2:], ["--base-url", "https://example.org/data"])

    def test_software_check_includes_all_three_suites(self):
        commands = self.plan("check")
        self.assertEqual([command[-2] for command in commands], ["method/tests", "data/tests", "scripts/tests"])

    def test_subprocess_failure_stops_later_commands(self):
        with patch("scripts.run.subprocess.run", side_effect=subprocess.CalledProcessError(7, ["first"])) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                execute([["first"], ["second"]])
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.kwargs, dict(cwd=ROOT, check=True))

    def test_paths_are_relative_to_the_repository(self):
        self.assertEqual(absolute(Path("data")), ROOT / "data")


if __name__ == "__main__":
    unittest.main()
