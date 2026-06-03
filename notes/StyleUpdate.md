## Highest-impact aesthetic changes

1. **Reduce the empty white space in the “Audit runner” card.**
   In the idle state, the card has a huge blank area on the right. Make the runner a compact horizontal form instead:

   * Put `Audit id` and `Audit arguments` on the same row.
   * Make `Audit id` about `260px` wide.
   * Let `Audit arguments` fill the remaining width.
   * Put `Run selected audit` and `Reset defaults` directly below or right-aligned in the same card.
   * Do not leave a large blank white panel.

2. **Make the post-run state visually denser but cleaner.**
   After an audit runs, collapse the runner into a compact toolbar:

   * `Audit: all`
   * `Args: none`
   * `Run again`
   * `Reset`
   * `Last run: 2026-06-03 01:22`

   The current results view needs more vertical space for findings, so the runner should not stay large.

3. **Use a stricter spacing system.**
   The UI currently mixes cramped areas with oversized empty areas. Use these spacing rules everywhere:

   * Page padding: `24px`
   * Gap between major sections: `16px` or `20px`
   * Card padding: `20px`
   * Small internal gaps: `8px` or `12px`
   * Result card gap: `16px`
   * Avoid random one-off spacing values.

4. **Make the visual hierarchy stronger.**
   Right now many elements have similar visual weight. Use this hierarchy:

   * Page title: largest and boldest.
   * Section titles: strong but smaller.
   * Card titles: medium-bold.
   * Metadata/help text: smaller and muted.
   * Status pills: visually distinct but not louder than the actual content.

---

## Color improvements

Use a more intentional palette. The current blue-gray look is good, but it is slightly washed out and inconsistent.

Recommended tokens:

```css
--page-bg: #f5f7fb;
--surface: #ffffff;
--surface-muted: #f8fafd;
--border: #d8e1ef;
--border-strong: #b7c7dd;

--text-main: #172033;
--text-muted: #64748b;
--text-soft: #8291a8;

--primary: #2563eb;
--primary-hover: #1d4ed8;
--primary-soft: #dbeafe;

--pass: #16a34a;
--pass-bg: #dcfce7;
--pass-border: #86efac;

--fail: #dc2626;
--fail-bg: #fee2e2;
--fail-border: #fca5a5;

--warn: #d97706;
--warn-bg: #fef3c7;
--warn-border: #fcd34d;
```

Specific changes:

* Keep the page background light blue-gray.
* Keep cards white.
* Use colored backgrounds only for status chips and alerts.
* Avoid large pale-blue blocks unless they clearly mean a temporary alert or emphasis state.
* The `AUDITS` top card should not be so heavily filled. Use a subtle blue border, left accent bar, or light blue background instead.

---

## Header improvements

1. **Make the top header feel like one intentional dashboard bar.**
   Current header elements are aligned well, but the `Typography` control feels like a developer/debug setting competing with real product cards.

   Better layout:

   * Left: app title and subtitle.
   * Center/right: status cards.
   * Far right or secondary row: typography/settings control.

2. **Move or downplay the typography selector.**
   It is too prominent for a production control center. Options:

   * Move it into a small settings button.
   * Rename it to `Font`.
   * Make it smaller.
   * Do not let it visually compete with `AUDITS`, `OPTIMIZATION`, and `DOCS`.

3. **Make the top status cards consistent.**
   Each card should follow the same structure:

   * Small uppercase label.
   * Status/value pill in the top-right.
   * One short description line.
   * Same height.
   * Same padding.
   * Same border radius.

4. **Use clearer top-card status styling.**

   * `AUDITS idle`: neutral blue-gray pill.
   * `AUDITS fail`: red pill.
   * `OPTIMIZATION 77 packets`: blue pill.
   * `DOCS ready`: green or blue pill, but choose one convention and reuse it.

---

## Audit runner improvements

Current issue: the runner looks like a form pasted into a huge empty container.

Make it compact:

```text
Audit runner
Run a new sweep across every registered audit.

[Audit id: all        ] [Audit arguments: optional flags...                         ]

[Run selected audit] [Reset defaults]   Last result: no audit has been run yet.
```

Rules:

* Do not let the argument input stretch across the entire page if there is not enough content around it.
* Keep input height around `44px`.
* Keep button height around `44px`.
* Align labels, inputs, and buttons to the same grid.
* Keep help text short.
* Replace the multi-line explanation under `Audit id` with one concise sentence.

Better `Audit id` help text:

```text
Use “all” to run every registered audit.
```

Better `Audit arguments` help text:

```text
Optional command-line flags.
```

---

## Control center audit section improvements

1. **Make the final status chip larger or more prominent.**
   The overall audit result should be visually distinct from the counts:

   * Counts are summary chips.
   * Overall result is the final state.
   * Use `Overall: Failed` or a larger `Failed` pill.

---

## Sidebar audit group improvements

The sidebar is useful but visually cramped.

Change each group row to this structure:

```text
[status dot] Environment/install audit                 [FAIL]
             10 items
```

Rules:

* Use a small colored dot or left border for status.
* Keep the status pill aligned right.
* Keep item count muted and smaller.
* Use consistent row height.
* The sidebar audit-group rows are navigation only. Do not render a selected or current-group state there.
* Avoid title wrapping where possible.
* For long titles, allow two lines max, then truncate.

Current problem example:
`Burton & Urban firing-rate-versus-current validation audit` wraps heavily and makes the sidebar feel uneven.

Better:

```text
Burton & Urban firing-rate...
52 items
[FAIL]
```

or:

```text
Burton & Urban validation
52 items
[FAIL]
```

4. **Add sticky behavior.**
   In the results view, the audit group sidebar should stay visible while scrolling through findings.

---

## Result card improvements

The result cards are the most important part of the page. They need the most polish.

Use a consistent card structure:

```text
[chevron] Small category label                         [PASS]

          Main finding title goes here.
          Keep it readable. Max 2–3 lines.

          Evidence/chart/details area
```

Specific changes:

* Align the chevron with the category label.
* Use the same top padding on every result card.
* Keep the status pill in the top-right corner.
* Use a clear separator between title and evidence area.
* Do not let charts look clipped at the bottom.
* Use the same card radius and border as other cards.
* Increase internal padding slightly.
* Make failed result cards easier to spot.

For failed cards:

* Add a subtle red left border: `4px solid var(--fail)`.
* Keep the background mostly white.
* Use a red status pill.
* Optionally tint the evidence area very lightly red.

For passing cards:

* Add a subtle green left border only if needed.
* Do not make pass cards too visually loud.
* Failures and warnings should receive more attention than passes.

---

## Fix the inconsistent highlighted card

In the first screenshot, one passing result card has a pale blue rectangle behind its content. It looks accidental.

Either:

* Remove the pale blue rectangle completely.

Or, if it means “selected”:

* Add a clear selected label/state.
* Use a blue border.
* Keep the highlight consistent across the whole card.

Do not highlight only the inner content area without explanation.

---

## Chart and evidence improvements

The mini charts are useful but need clearer visual treatment.

Improve them like this:

* Increase chart vertical padding.
* Make the accepted range bar easier to distinguish from the track.
* Make the reference mean line darker and slightly thicker.
* Make the observed marker larger.
* Add numeric labels near the observed marker when possible.
* Put the legend below the chart in one clean row.
* Use the same legend order everywhere:

  * accepted range
  * reference mean
  * observed

Current issue:
The charts look slightly like form sliders, which makes them ambiguous.

Make them look more like evidence visualizations, not controls.

---

## Empty state improvements

The idle screenshot shows `0 visible items across 0 visible groups`, but then still shows display controls. That feels broken.

Instead, when there are no audit results:

```text
No audit results yet

Run an audit to populate findings, groups, filters, and evidence.
[Run selected audit]
```

Rules:

* Do not show `0 visible items across 0 visible groups` as the main message.
* Use a centered empty state inside the results card.

---

## Display controls improvements

When visible:

* Put search on the left.
* Put filters in the middle.
* Put group controls on the right.
* Keep everything aligned to one row on large screens.
* Wrap gracefully on smaller screens.
* Use segmented controls or filter chips consistently.

Current controls are functional but look slightly bolted on.

Better layout:

```text
Search findings                         Filters                         Group view
[ Search...                         ]   [Failures] [Warnings] [Passes]   [Expand all] [Collapse all]
```

---

## Typography improvements

1. **Recommended type scale:**

```css
.page-title {
  font-size: 24px;
  font-weight: 700;
}

.section-title {
  font-size: 20px;
  font-weight: 700;
}

.card-title {
  font-size: 16px;
  font-weight: 650;
}

.body {
  font-size: 14px;
  font-weight: 400;
}

.label {
  font-size: 13px;
  font-weight: 600;
}

.metadata {
  font-size: 13px;
  font-weight: 400;
  color: var(--text-muted);
}
```

2. **Use tabular numbers for audit counts.**

```css
.status-chip,
.metric-value {
  font-variant-numeric: tabular-nums;
}
```

This makes counts like `34`, `12`, `95`, and `77` align better.

---

## Border, radius, and shadow improvements

Current cards have many borders. Keep borders, but make them more refined.

Use:

```css
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
```

For elevated/active cards:

```css
.card-active {
  border-color: #93b4e8;
  box-shadow: 0 6px 18px rgba(37, 99, 235, 0.10);
}
```

Rules:

* Do not overuse shadows.
* Do not use heavy blue glows.
* Use shadows only for top cards, modals, active states, or major containers.
* Keep most result cards flat and clean.

---

## Button improvements

Current buttons are fine but should be more consistent.

Use:

```css
.primary-button {
  height: 44px;
  padding: 0 18px;
  border-radius: 9px;
  background: var(--primary);
  color: white;
  font-weight: 650;
}

.secondary-button {
  height: 44px;
  padding: 0 18px;
  border-radius: 9px;
  background: white;
  color: var(--text-main);
  border: 1px solid var(--border);
  font-weight: 600;
}
```

Add hover states:

* Primary hover: darker blue.
* Secondary hover: light blue-gray background.
* Disabled: lower opacity and no strong hover.

---

## Accessibility improvements

* Increase contrast of muted blue-gray text.
* Do not use color alone for pass/fail/warn. Add text and icons.
* Add visible keyboard focus states.
* Make status chips readable at small sizes.
* Ensure clickable cards have clear hover and focus states.
* Avoid tiny italic labels.

Good status examples:

```text
✓ Passed
✕ Failed
! Warning
```

---

## Responsive behavior

For wide screens:

* Use two columns for result cards.
* Keep sidebar fixed width around `320px`.
* Keep content max width around `1600px` or `1680px`.

For medium screens:

* Sidebar can shrink to `280px`.
* Result cards may stay two columns if readable.

For small screens:

* Sidebar becomes a collapsible drawer or top filter list.
* Result cards become one column.
* Header cards wrap below the title.
* Runner form stacks vertically.

---

## Overall aesthetic direction

The current site already has a solid internal-tool/dashboard foundation. The biggest aesthetic problem is not the color scheme; it is the uneven density. Some areas are too empty, while others are too cramped.

The target look should be:

```text
calm technical dashboard
clean audit/reporting interface
high readability
failures easy to find
less empty space
more consistent spacing
fewer accidental-looking highlights
clearer cards and status states
```
