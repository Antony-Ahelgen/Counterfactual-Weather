from cfveg.utils.provenance import load_preset


def test_presets_load():
    for p in ["smoke", "cpu", "small"]:
        cfg = load_preset(p)
        assert cfg["preset"] == p
        assert cfg["window"]["context"] > 0 and cfg["window"]["horizon"] > 0


def test_splits_years_disjoint():
    s = load_preset("cpu")["splits"]
    tr, va, te = set(s["train_years"]), set(s["val_years"]), set(s["test_years"])
    assert not (tr & va) and not (tr & te) and not (va & te)
