"""Shared fixtures. All tests use a small REAL-data subset (tests/fixtures/real_cube, built by
scripts/make_fixture.py from downloaded Sentinel-2 + ERA5 + DEM/WorldCover). No synthetic data."""
from pathlib import Path

import pytest

from cfveg.data.dataset import Cube

FIXTURE = Path(__file__).parent / "fixtures" / "real_cube"


@pytest.fixture(scope="session")
def real_cube():
    return Cube.load(FIXTURE, mmap=False)
