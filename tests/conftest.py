"""Shared fixtures. Tests that need real data read the newest local snapshot."""

from __future__ import annotations

import pytest

from fplr.ingest import finalised_gameweeks, latest_snapshot


@pytest.fixture(scope="session")
def snapshot():
    snap = latest_snapshot()
    if snap is None:
        pytest.skip("no snapshot on disk; run `fpl sync` first")
    return snap


@pytest.fixture(scope="session")
def bootstrap(snapshot):
    return snapshot.read("bootstrap")


@pytest.fixture(scope="session")
def positions(bootstrap):
    return {e["id"]: e["element_type"] for e in bootstrap["elements"]}


@pytest.fixture(scope="session")
def live_gameweeks(snapshot, bootstrap):
    """Every finalised gameweek present in the snapshot, as {gw: live payload}."""
    available = {
        gw: snapshot.read(f"live_gw{gw:02d}")
        for gw in finalised_gameweeks(bootstrap)
        if snapshot.has(f"live_gw{gw:02d}")
    }
    if not available:
        pytest.skip("snapshot contains no finalised gameweeks")
    return available
