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
    panel_toolbar_html_by_key={"audits": "<section>audit runner</section>"},
    shell_state={
        "tabs": {"audits": {"badge": "FAIL", "badge_tone": "fail", "src": "/audits/index.html", "revision": "123"}},
        "progress": {"active": True, "label": "Running audit", "value_text": "1/4", "fraction": 0.25, "indeterminate": False},
    },
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
assert "audit runner" in html
assert "/__control_center_state__" in html
assert "dashboard-shell-state" in html
assert "shell-status-chip" in html
assert "shell-status-strip" in html
assert "tone-fail" in html
assert "shell-progress" in html
assert "Running audit" in html
assert "dashboard-font-mode" in html
assert ">Font<" in html
assert "FONT_FAMILY_BY_MODE" in html
assert 'helvetica: \'"Helvetica Neue\", Helvetica, Arial, sans-serif\'' in html
assert 'verdana:' in html

print("neuroinfra_dashboard_shell: OK")
