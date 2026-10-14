import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture(scope="session")
def d():
    from cust360 import data
    from cust360.features import Data

    return Data(tables=data.generate(n_customers=2500, seed=9)[:3] + (data._catalogue(),))
