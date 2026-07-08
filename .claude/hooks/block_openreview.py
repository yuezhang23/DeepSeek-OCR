#!/usr/bin/env python3
"""PreToolUse hook — deny any WebSearch/WebFetch that targets OpenReview.

Prevents the dimensional-scoring agents (review_pipeline/scorer.py, driven via the
Claude Agent SDK with setting_sources=["project"]) from probing the paper's human peer
reviews. Registered in .claude/settings.json for the WebSearch|WebFetch matcher.

Protocol: reads the PreToolUse event JSON on stdin and, when the tool input mentions
OpenReview, prints a deny decision on stdout. Anything else is allowed (empty output).
"""
import json
import sys


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # fail-open on unparseable input rather than wedge the agent

    tool_name = event.get("tool_name", "")
    if tool_name in ("WebSearch", "WebFetch"):
        blob = json.dumps(event.get("tool_input", {})).lower()
        if "openreview" in blob:
            print(json.dumps({
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        "Probing human peer reviews from OpenReview is forbidden during "
                        "autonomous scoring. Score only from the paper's own content and "
                        "general prior art."
                    ),
                }
            }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
