import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_environment import get_required_requirements


class VerifyEnvironmentTests(unittest.TestCase):
    def test_get_required_requirements_includes_core_packages(self):
        requirements = get_required_requirements()
        names = {pkg[0] for pkg in requirements}

        self.assertIn("torch", names)
        self.assertIn("transformers", names)
        self.assertIn("datasets", names)
        self.assertIn("pyarrow", names)
        self.assertIn("tqdm", names)


if __name__ == "__main__":
    unittest.main()
