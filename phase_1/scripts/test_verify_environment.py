import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

spec = importlib.util.spec_from_file_location("autonomous_agent", ROOT / "agents" / "autonomous_agent.py")
autonomous_agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(autonomous_agent)


class VerifyEnvironmentTests(unittest.TestCase):
    def test_extract_referenced_paths_ignores_prose_slash_tokens(self):
        issue_body = """
        ## Acceptance Criteria
        - Create phase_1/scripts/verify_environment.py
        - Use yes/no/feedback for the UI prompt
        - Mark done/feedback as the final status
        - Software requires Python 3.10+
        """

        paths = autonomous_agent.extract_referenced_paths(issue_body)

        self.assertIn("phase_1/scripts/verify_environment.py", paths)
        self.assertNotIn("yes/no/feedback", paths)
        self.assertNotIn("done/feedback", paths)
        self.assertNotIn("3.10", paths)


if __name__ == "__main__":
    unittest.main()
