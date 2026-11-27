"""Reject a mismatched environment even when only one test file is selected."""

import pytest

from dependency_versions import version_mismatches


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    try:
        mismatches = version_mismatches()
    except (OSError, ValueError) as error:
        pytest.exit(
            f"Cannot validate requirements.txt: {error}",
            returncode=pytest.ExitCode.USAGE_ERROR,
        )
    if mismatches:
        pytest.exit(
            "Dependency versions do not match requirements.txt:\n"
            + "\n".join(f"  {mismatch}" for mismatch in mismatches)
            + "\nInstall requirements.txt in the Python environment running pytest.",
            returncode=pytest.ExitCode.USAGE_ERROR,
        )
