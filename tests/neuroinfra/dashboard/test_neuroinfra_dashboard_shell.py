"""Smoke tests for the shared dashboard shell renderer."""

from __future__ import annotations

from neuroinfra.dashboard import ShellTabSpec, render_dashboard_shell


html = render_dashboard_shell(
    title="Control Center",
    subtitle="Unified maintained shell",
    tabs=(
        ShellTabSpec(key="audits", label="Audits", src="/audits/index.html", badge="FAIL", badge_tone="fail"),
        ShellTabSpec(key="docs", label="Docs", src="/docs/index.html"),
    ),
    initial_tab="docs",
    toolbar_html="<section>toolbar</section>",
    shell_state={"tabs": {"audits": {"badge": "FAIL", "badge_tone": "fail", "src": "/audits/index.html", "revision": "123"}}},
    state_endpoint="/__control_center_state__",
    state_poll_interval_ms=1500,
)

assert "Control Center" in html
assert "Unified maintained shell" in html
assert "tab-audits" in html
assert "tab-docs" in html
assert "/audits/index.html" in html
assert "/docs/index.html" in html
assert "FAIL" in html
assert "toolbar" in html
assert "/__control_center_state__" in html
assert "dashboard-shell-state" in html
assert "tab-badge tone-fail" in html

print("neuroinfra_dashboard_shell: OK")
