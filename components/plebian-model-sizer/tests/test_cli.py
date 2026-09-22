from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from plebian_model_sizer.cli import main
from test_sizing import catalog, snapshot


class CliTests(unittest.TestCase):
    def test_empty_shortlist_is_a_report_not_an_execution_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "catalog.json").write_text(json.dumps(catalog()))
            data = snapshot(); data["ram_available_bytes"] = 0
            (root / "resources.json").write_text(json.dumps(data))
            output = StringIO()
            with redirect_stdout(output):
                code = main(["recommend", "help-llm", "--catalog", str(root / "catalog.json"),
                             "--resources", str(root / "resources.json"), "--json"])
            result = json.loads(output.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(result["resource_source"], "provided")
            self.assertEqual(result["shortlist"], [])
            self.assertIsNone(result["selected_model"])
            self.assertFalse(result["qualification_eligible"])

    def test_release_replay_and_mutating_commands_are_not_implemented(self):
        for args in (["plan", "local-ai-balanced"], ["install", "plan.json"],
                     ["install-status", "id"], ["cancel", "id"],
                     ["recommend", "tts"], ["train"]):
            with self.subTest(args=args), redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                main(args)
            self.assertEqual(error.exception.code, 2)

    def test_bad_json_reports_structured_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text('{"schema": 1, "schema": 2}')
            output = StringIO()
            with redirect_stdout(output):
                code = main(["recommend", "help-llm", "--catalog", str(path), "--json"])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["error"], "duplicate JSON key")


if __name__ == "__main__":
    unittest.main()
