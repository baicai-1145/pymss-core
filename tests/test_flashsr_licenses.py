from importlib.resources import files

import pytest


@pytest.mark.parametrize("name,notice", [
    ("Apache-2.0.txt", "Apache License"),
    ("Snake-MIT.txt", "Copyright (c) 2020 Edward Dixon"),
    ("Julius-MIT.txt", "Copyright 2020 Alexandre D\u00e9fossez"),
    ("HierSpeechpp-MIT.txt", "Copyright (c) 2024 Sang-Hoon Lee"),
])
def test_flashsr_ships_complete_third_party_licenses(name, notice):
    text = files("pymss_core.modules.flashsr").joinpath("incl_licenses", name).read_text(encoding="utf-8")
    assert notice in text
    if name == "Apache-2.0.txt":
        assert "Redistribution" in text
        assert "END OF TERMS AND CONDITIONS" in text
    else:
        assert "Permission is hereby granted" in text
        assert "THE SOFTWARE IS PROVIDED" in text


def test_flashsr_ships_sources_revisions_and_modification_notice():
    text = files("pymss_core.modules.flashsr").joinpath("NOTICE.txt").read_text(encoding="utf-8")
    for revision in (
        "2a69326250613c0a0f6c1c8d9f0c48cb779842b8",
        "318c633cdc24cf3239d62c95ea8afae200ed2d13",
        "90343ed9458364dc3e622fc37be49d35afc26a6b",
        "390ccc265a4087b78cf1ee9098d48721d848fdc5",
        "2ff87901b0841a533053a5c34d0fa43486764086",
    ):
        assert revision in text
    assert "Modifications" in text
