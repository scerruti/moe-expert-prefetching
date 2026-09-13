import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_environment import get_required_requirements

spec = importlib.util.spec_from_file_location("autonomous_agent", ROOT / "agents" / "autonomous_agent.py")
autonomous_agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(autonomous_agent)


class VerifyEnvironmentTests(unittest.TestCase):
    def test_get_required_requirements_includes_core_packages(self):
        requirements = get_required_requirements()
        names = {pkg[0] for pkg in requirements}

        self.assertIn("torch", names)
        self.assertIn("transformers", names)
        self.assertIn("datasets", names)
        self.assertIn("pyarrow", names)
        self.assertIn("tqdm", names)

    def test_extract_referenced_paths_ignores_prose_slash_tokens(self):
        issue_body = """
        ## Acceptance Criteria
        - Create phase_1/scripts/verify_environment.py
        - Use yes/no/feedback for the UI prompt
        - Mark done/feedback as the final status
        """

        paths = autonomous_agent.extract_referenced_paths(issue_body)

        self.assertIn("phase_1/scripts/verify_environment.py", paths)
        self.assertNotIn("yes/no/feedback", paths)
        self.assertNotIn("done/feedback", paths)


if __name__ == "__main__":
    unittest.main()
