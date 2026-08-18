"""Check the Rayline ARC turn parser's block tables against the vendor SDK unions.

The parser is fail-closed. It renders a small set of block and item types,
drops a known table of others, and raises `unknown_item` for anything else.
An unrecognised block therefore fails the whole routing consultation rather
than degrading it, which surfaces in production as a 503.

So every discriminator string a vendor SDK can put on the wire must be either
rendered by the parser or named in its drop table. This script set-differences
the two and exits non-zero when a member is in neither. It turns a request-time
outage into a build failure.

Run it with the SDK versions pinned in tools/agent/requirements-sdk-drift.txt:

    make agent-sdk-drift

Members the parser knows but the SDK no longer declares are reported and do not
fail the check. A stale drop-table entry simply never matches, so it costs
nothing; a vendor removing a type must not break the build.
"""

from __future__ import annotations

import glob
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PARSER_DIR = os.path.join(
    REPO_ROOT, "src", "semantic-router", "pkg", "selection", "raylinearc")

# `type: Literal["x"]` under Required[...], Optional[...], or bare.
LITERAL = re.compile(
    r'^\s{4}type:\s*(?:Required\[|Optional\[)?Literal\["([^"]+)"\]', re.M)
# Class bodies: plain TypedDict/BaseModel and the reserved-keyword shape
# `class BetaFallbackBlockParam(_BetaFallbackBlockParamReservedKeywords, ...)`.
CLASS = re.compile(r"^class (\w+)\([^)]*\):(.*?)(?=\n\nclass |\Z)", re.S | re.M)
# Union aliases, with or without an Annotated[...] discriminator wrapper.
ALIAS = re.compile(
    r"^(\w+): TypeAlias = (?:Annotated\[\s*)?Union\[(.*?)\n(?:    )?\]", re.S | re.M)

ANTHROPIC_UNIONS = [
    ("GA   ContentBlockParam", "ContentBlockParam"),
    ("GA   ContentBlock", "ContentBlock"),
    ("BETA BetaContentBlockParam", "BetaContentBlockParam"),
    ("BETA BetaContentBlock", "BetaContentBlock"),
]


def sdk_types_dir(module_name):
    """Locate an installed SDK's `types` directory, or exit asking for it."""
    try:
        module = __import__(module_name)
    except ImportError:
        sys.exit(
            "ERROR: %s is not installed.\n"
            "       Install the pinned SDKs first:\n"
            "         pip install -r tools/agent/requirements-sdk-drift.txt"
            % module_name)
    return os.path.join(os.path.dirname(module.__file__), "types")


def sdk_version(module_name):
    return __import__(module_name).__version__


def index(root):
    """Map class name -> discriminator, and alias name -> member class names."""
    literals, aliases = {}, {}
    for path in glob.glob(root + "/**/*.py", recursive=True):
        with open(path, encoding="utf-8") as handle:
            src = handle.read()
        for match in CLASS.finditer(src):
            discriminator = LITERAL.search(match.group(2))
            if discriminator:
                literals[match.group(1)] = discriminator.group(1)
        for match in ALIAS.finditer(src):
            aliases.setdefault(match.group(1), [
                member.strip().rstrip(",")
                for member in match.group(2).strip().split("\n") if member.strip()])
    return literals, aliases


def resolve(name, literals, aliases, depth=0):
    """Flatten a union alias to the set of discriminator strings it can carry."""
    members = set()
    for member in aliases.get(name, []):
        if member in literals:
            members.add(literals[member])
        elif member in aliases and depth < 8:
            members |= resolve(member, literals, aliases, depth + 1)
        else:
            members.add("!!UNRESOLVED:" + member)
    return members


def go_source(filename):
    with open(os.path.join(PARSER_DIR, filename), encoding="utf-8") as handle:
        return handle.read()


def drop_table(src, var):
    match = re.search(r"var %s = map\[string\]bool\{(.*?)\n\}" % var, src, re.S)
    if not match:
        sys.exit("ERROR: could not find drop table %s in the parser source" % var)
    return set(re.findall(r'"([^"]+)":\s+true', match.group(1)))


def report(label, union, known):
    unhandled = sorted(union - known)
    print("  %-30s union=%-3d unhandled=%s"
          % (label, len(union), unhandled if unhandled else "none"))
    return unhandled


def check_anthropic():
    types_dir = sdk_types_dir("anthropic")
    literals, aliases = index(types_dir)
    src = go_source("turns_anthropic.go")
    dropped = drop_table(src, "anthropicDroppedBlockTypes")

    # Tool-result blocks nest their own type switch; those discriminators belong
    # to the nested content union, not the top-level block union.
    nested = re.search(r"func toolResultBlockText\(.*?\n\}", src, re.S)
    top_level = src.replace(nested.group(0), "") if nested else src
    rendered = set(re.findall(r'blockType == "([a-z_0-9]+)"', top_level))
    known = rendered | dropped

    print("anthropic %s  rendered=%s  dropped=%d"
          % (sdk_version("anthropic"), sorted(rendered), len(dropped)))
    unhandled, seen = [], set()
    for label, alias in ANTHROPIC_UNIONS:
        union = resolve(alias, literals, aliases)
        seen |= union
        unhandled += report(label, union, known)
    stale = sorted(dropped - seen)
    print("  dropped but in no union (harmless): %s" % (stale if stale else "none"))
    return unhandled


def check_openai_responses():
    types_dir = sdk_types_dir("openai")
    literals, aliases = index(types_dir)
    src = go_source("turns_openai_responses.go")
    dropped = drop_table(src, "droppedResponsesItemTypes")
    rendered = set(re.findall(r'^\tcase "([a-z_0-9]+)":', src, re.M))
    known = rendered | dropped

    print("\nopenai %s  rendered=%s  dropped=%d"
          % (sdk_version("openai"), sorted(rendered), len(dropped)))
    union = resolve("ResponseInputItemParam", literals, aliases)
    unhandled = report("ResponseInputItemParam", union, known)
    stale = sorted(dropped - union)
    print("  dropped but in no union (harmless): %s" % (stale if stale else "none"))
    return unhandled


def main():
    unhandled = check_anthropic() + check_openai_responses()
    print()
    if unhandled:
        print("FAIL: the vendor SDKs declare block types the parser handles "
              "neither way:")
        for member in sorted(set(unhandled)):
            print("  - %s" % member)
        print()
        print("Each one fails the episode closed at runtime, which is a 503.")
        print("Render it, or add it to the drop table with a reason.")
        return 1
    print("OK: every SDK union member is either rendered or dropped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
