"""Objective-C class names are global in a process: two PyObjC classes with the same name (even
in different Python files) crash the app the moment the second one loads.
(Nexus.app, 2026-09-29: "_Target is overriding existing Objective-C class".)"""
import ast
from collections import Counter
from pathlib import Path

OBJC_BASES = {"NSObject", "NSView", "NSPanel", "NSWindow", "NSWindowController", "NSViewController"}


def test_objc_class_names_are_unique():
    names = []
    for path in sorted(Path("nexus").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef):
                bases = {b.id for b in node.bases if isinstance(b, ast.Name)} | \
                        {b.attr for b in node.bases if isinstance(b, ast.Attribute)}
                if bases & OBJC_BASES:
                    names.append((node.name, path.name))
    counts = Counter(n for n, _ in names)
    dupes = {n: [f for m, f in names if m == n] for n, c in counts.items() if c > 1}
    assert not dupes, f"Objective-C class names must be unique across nexus/: {dupes}"
    assert names, "expected to find the app's PyObjC classes"
