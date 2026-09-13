import importlib.util
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

    def test_extract_referenced_paths_ignores_version_strings(self):
        repo_root = Path(__file__).resolve().parents[2]
        agent_path = repo_root / "agents" / "autonomous_agent.py"

        spec = importlib.util.spec_from_file_location("autonomous_agent", agent_path)
        autonomous_agent = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(autonomous_agent)

        issue_body = """## Description
Set up the Python environment for Phase 1 data collection.

## Tasks
- [ ] Create requirements.txt with all dependencies (torch, transformers, datasets, pyarrow, tqdm)
- [ ] Test Python environment (3.10+ required)
- [ ] Verify GPU access (CUDA, torch install)
- [ ] Create data directory structure
- [ ] Create phase_1/scripts directory

## Acceptance Criteria
- requirements.txt is committed
- All dependencies install without errors
- GPU is accessible via PyTorch
- Directory structure is in place"""

        paths = autonomous_agent.extract_referenced_paths(issue_body)

        self.assertIn("requirements.txt", paths)
        self.assertIn("phase_1/scripts", paths)
        self.assertNotIn("3.10", paths)


if __name__ == "__main__":
    unittest.main()
