from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_defines_wearable_recovery_goal():
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()

    assert "need more recovery" in readme
    assert "real fitbit" in readme
    assert "active minutes" in readme
    assert "does not contain a measured fatigue answer" in readme
