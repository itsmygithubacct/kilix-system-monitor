"""A development executable cannot silently become a qualified staged child."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("readiness", ROOT / "tools/check_trusted_launcher_consumer_readiness.py")
readiness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readiness)


class DevelopmentBoundaryTests(unittest.TestCase):
    def test_development_entry_point_allowed_but_release_promotion_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for component in ("plebian-hardware", "plebian-model-sizer"):
                path = root / "components" / component / "pyproject.toml"
                path.parent.mkdir(parents=True)
                path.write_bytes((ROOT / "components" / component / "pyproject.toml").read_bytes())
            manifest = (ROOT / "manifest.toml").read_text()
            (root / "manifest.toml").write_text(manifest)
            with patch.object(readiness, "ROOT", root):
                readiness._validate_consumer_paths([])
                (root / "manifest.toml").write_text(manifest.replace("development-estimates-unqualified", "release-qualified"))
                with self.assertRaises(readiness.ReadinessFailure):
                    readiness._validate_consumer_paths([])


if __name__ == "__main__":
    unittest.main()
