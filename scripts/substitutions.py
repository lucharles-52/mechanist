"""List the calls of a Mechanist run that a safety classifier refused and a fallback model answered.

    python3 scripts/substitutions.py runs/<run>      # prints the list and writes runs/<run>/substitutions.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mechanist.report import substitution_lines

run = Path(sys.argv[1])
lines = substitution_lines(run) or ["No calls were answered by a fallback model."]
(run / "substitutions.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
