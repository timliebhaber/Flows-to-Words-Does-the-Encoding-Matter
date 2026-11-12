import copy
from pathlib import Path

import pytest

from src.config import ConfigError, validate_data_config
from src.data.load import check_header


def test_real_config_is_valid(cfg):
    cols = cfg["columns"]
    assert not set(cols["leaky"]) & set(cols["features"])
    assert "L4_DST_PORT" in cols["features"]
    for name in ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "FLOW_START_MILLISECONDS",
                 "FLOW_END_MILLISECONDS", "DNS_QUERY_ID"]:
        assert name in cols["leaky"]
    assert set(cfg["dataset_configs"]) == set(cfg["datasets"])


def test_raw_headers_match_config(cfg):
    """Header-only check against the real raw files that are present."""
    files = [Path(cfg["paths"]["raw_dir"]) / ds["raw_file"] for ds in cfg["dataset_configs"].values()]
    present = [f for f in files if f.exists()]
    if not present:
        pytest.skip("no raw files present")
    for f in present:
        check_header(f, cfg["columns"])


@pytest.mark.parametrize("mutate", [
    lambda c: c["sampling"].pop("sample_size"),
    lambda c: c["columns"]["features"].append(c["columns"]["leaky"][0]),
    lambda c: c["split"].update({next(iter(c["split"])): 0.9}),
    lambda c: c["dedupe"].update({"conflict_policy": "something_else"}),
])
def test_invalid_config_raises(cfg, mutate):
    bad = copy.deepcopy({k: v for k, v in cfg.items() if k != "dataset_configs"})
    mutate(bad)
    with pytest.raises(ConfigError):
        validate_data_config(bad)
