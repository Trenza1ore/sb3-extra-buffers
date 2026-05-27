#!/usr/bin/env python3
"""Rewrite pepy.tech personalized badge URLs with a new uuid query segment.

Scans all ``README*.md`` files under the repository root (recursive) and applies
``re.sub`` so each matched badge URL ends with ``&uuid=<new hex>`` before the
closing parenthesis in Markdown links.
"""

import random
import re
import uuid
from functools import cache
from pathlib import Path


@cache
def _get_mutated_uuid() -> str:
    uuid_str = list(uuid.uuid4().hex)
    change_idx = list(range(32))
    random.shuffle(change_idx)
    for idx in change_idx[: random.randrange(0, 17)]:
        uuid_str[idx] = hex(random.randrange(0, 16))[-1]
    return "".join(uuid_str)


ROOT = Path(__file__).resolve().parent.parent
BADGES = [
    re.compile(
        r"(https://img\.shields\.io/pepy/dt/sb3-extra-buffers)[^)]*(\))",
    ),
]
SEMANTIC_VER = (ROOT / "sb3_extra_buffers" / "version.txt").read_text().strip()


def main() -> None:
    random.seed(SEMANTIC_VER)
    total = 0
    for path in sorted(ROOT.glob("**/README*.md")):
        new_text = path.read_text(encoding="utf-8")
        n_total = 0
        for badge in BADGES:
            new_text, n = badge.subn(
                lambda m: m.group(1) + "?uuid=" + _get_mutated_uuid() + m.group(2),
                new_text,
            )
            n_total += n
        if n_total:
            path.write_text(new_text, encoding="utf-8")
            total += n_total
            print(f"{path.relative_to(ROOT)}: {n_total} replacement(s)")
    if total == 0:
        print("No matches.")


if __name__ == "__main__":
    main()
