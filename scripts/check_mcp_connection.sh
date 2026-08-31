#!/usr/bin/env bash
# Quick sanity check that both MCP servers are registered correctly
# before starting an annotation session. Run from the project root.
set -euo pipefail

echo "== Registered MCP servers (project scope) =="
codex mcp list

echo
echo "== Roboflow entry (should show a 'url' field, not 'command') =="
codex mcp get roboflow

echo
echo "== Label Studio entry (should show a 'command' field) =="
codex mcp get label_studio

echo
echo "If either check looks wrong, fix with:"
echo "  codex mcp remove <name>"
echo "  codex mcp add <name> --url <url>          # for HTTP servers"
echo "  codex mcp add <name> -- <command> <args>   # for stdio servers"
echo
echo "Then start a session and run /mcp inside Codex to confirm tools load."
