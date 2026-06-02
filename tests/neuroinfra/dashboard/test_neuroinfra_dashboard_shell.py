"""Smoke tests for the shared dashboard shell renderer."""

from __future__ import annotations

from neuroinfra.dashboard import ShellTabSpec, render_dashboard_shell


html = render_dashboard_shell(
    title="Control Center",
    subtitle="Unified maintained shell",
    tabs=(
        ShellTabSpec(key="audits", label="Audits", src="/audits/index.html", badge="FAIL"),
        ShellTabSpec(key="docs", label="Docs", src="/docs/index.html"),
    ),
    initial_tab="docs",
)

assert "Control Center" in html
assert "Unified maintained shell" in html
assert "tab-audits" in html
assert "tab-docs" in html
assert "/audits/index.html" in html
assert "/docs/index.html" in html
assert "FAIL" in html

print("neuroinfra_dashboard_shell: OK")
