#!/usr/bin/env python3
"""Tests for forge-test.py training data and formatting.

Tests the FLEET_TILES data and format_training_pair function
(without requiring PyTorch/transformers).
"""
import sys
import os
import types

# Load forge-test.py source (before main)
test_path = os.path.join(os.path.dirname(__file__), '..', 'forge-test.py')
with open(test_path) as f:
    source = f.read()

# Cut at main()
marker = "def main():"
idx = source.find(marker)
if idx > 0:
    source = source[:idx]

# forge-test.py imports torch and transformers at the top
# We need to stub them to avoid the import requirement
import types as types_mod

# Create stub modules
torch_stub = types_mod.ModuleType("torch")
torch_stub.nn = types_mod.ModuleType("torch.nn")
torch_stub.optim = types_mod.ModuleType("torch.optim")
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("torch.nn", torch_stub.nn)
sys.modules.setdefault("torch.optim", torch_stub.optim)

transformers_stub = types_mod.ModuleType("transformers")
transformers_stub.AutoModelForCausalLM = type("AutoModelForCausalLM", (), {})
transformers_stub.AutoTokenizer = type("AutoTokenizer", (), {})
sys.modules.setdefault("transformers", transformers_stub)

mod = types_mod.ModuleType("forge_test_fns")
mod.__file__ = test_path
exec(source, mod.__dict__)

FLEET_TILES = mod.FLEET_TILES
format_training_pair = mod.format_training_pair


class TestFleetTiles:
    """Test the FLEET_TILES training data."""

    def test_tiles_non_empty(self):
        """Should have training data."""
        assert len(FLEET_TILES) > 0

    def test_each_tile_has_required_fields(self):
        """Every tile should have query, good, bad, domain, level."""
        required = {"query", "good", "bad", "domain", "level"}
        for i, tile in enumerate(FLEET_TILES):
            missing = required - set(tile.keys())
            assert not missing, f"Tile {i} missing fields: {missing}"

    def test_good_responses_are_substantive(self):
        """Good answers should be detailed (not one-liners)."""
        for tile in FLEET_TILES:
            assert len(tile["good"]) > 50, (
                f"Good answer too short for '{tile['query'][:40]}...': {len(tile['good'])} chars"
            )

    def test_bad_responses_are_vague(self):
        """Bad answers should be short/vague (that's the point)."""
        for tile in FLEET_TILES:
            assert len(tile["bad"]) < len(tile["good"]), (
                f"Bad answer should be shorter than good for '{tile['query'][:40]}...'"
            )

    def test_domains_are_valid(self):
        """Domains should be from expected set."""
        valid_domains = {"plato", "forge", "fleet", "math"}
        for tile in FLEET_TILES:
            assert tile["domain"] in valid_domains, (
                f"Unknown domain '{tile['domain']}' in '{tile['query'][:40]}...'"
            )

    def test_levels_are_valid(self):
        """Levels should be from expected set."""
        valid_levels = {"greenhorn", "operator", "specialist"}
        for tile in FLEET_TILES:
            assert tile["level"] in valid_levels, (
                f"Unknown level '{tile['level']}' in '{tile['query'][:40]}...'"
            )

    def test_queries_are_unique(self):
        """Each query should be unique."""
        queries = [t["query"] for t in FLEET_TILES]
        assert len(queries) == len(set(queries)), "Duplicate queries found"

    def test_no_empty_fields(self):
        """No field should be empty."""
        for i, tile in enumerate(FLEET_TILES):
            for key, val in tile.items():
                assert val, f"Tile {i} has empty '{key}'"


class TestFormatTrainingPair:
    """Test the training pair formatter."""

    def test_produces_string(self):
        """Should return a string."""
        tile = FLEET_TILES[0]
        result = format_training_pair(tile)
        assert isinstance(result, str)

    def test_contains_query(self):
        """Should contain the query."""
        tile = FLEET_TILES[0]
        result = format_training_pair(tile)
        assert tile["query"] in result

    def test_contains_good_answer(self):
        """Should contain the good answer."""
        tile = FLEET_TILES[0]
        result = format_training_pair(tile)
        assert tile["good"] in result

    def test_contains_bad_answer(self):
        """Should contain the bad answer."""
        tile = FLEET_TILES[0]
        result = format_training_pair(tile)
        assert tile["bad"] in result

    def test_contains_domain(self):
        """Should contain the domain."""
        tile = FLEET_TILES[0]
        result = format_training_pair(tile)
        assert tile["domain"] in result

    def test_has_q_prefix(self):
        """Should start with Q:."""
        tile = FLEET_TILES[0]
        result = format_training_pair(tile)
        assert result.startswith("Q:")

    def test_all_tiles_format_cleanly(self):
        """Every tile should format without errors."""
        for tile in FLEET_TILES:
            result = format_training_pair(tile)
            assert len(result) > 0
            assert "Q:" in result
            assert "Good:" in result
            assert "Bad:" in result


class TestTrainingDataCoverage:
    """Test coverage of topics across the training data."""

    def test_covers_multiple_domains(self):
        """Training data should cover multiple domains."""
        domains = set(t["domain"] for t in FLEET_TILES)
        assert len(domains) >= 3, f"Only {len(domains)} domains covered"

    def test_covers_multiple_levels(self):
        """Training data should cover multiple difficulty levels."""
        levels = set(t["level"] for t in FLEET_TILES)
        assert len(levels) >= 2, f"Only {len(levels)} levels covered"

    def test_has_at_least_one_p0_example(self):
        """At least one tile should mention P0/deadband."""
        p0_tiles = [t for t in FLEET_TILES if "P0" in t["good"] or "deadband" in t["good"].lower()]
        assert len(p0_tiles) >= 1, "No P0/deadband examples in training data"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
