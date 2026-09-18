"""The pages live in static/ now, not embedded in the module.

Extracting them moved ~1,100 lines out of transcribe_server.py and let the CSP
drop both 'unsafe-inline' directives. That is only safe if no rule and no class
went missing on the way, which is what these pin: the CSS baselines below were
captured from the embedded pages before the extraction.
"""

from __future__ import annotations

import re

import pytest

import transcribe_server as s

STATIC = s.STATIC_DIR


def asset(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def selectors(css: str) -> set[str]:
    """Every selector in a stylesheet, media queries unwrapped."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    css = re.sub(r"@media[^{]*\{", "", css)
    return {
        " ".join(m.group(1).split())
        for m in re.finditer(r"([^{}]+)\{", css)
        if m.group(1).strip() and not m.group(1).strip().startswith("@")
    }


def classes_in(text: str) -> set[str]:
    """Class names used by markup and by classList/className in the scripts."""
    found: set[str] = set()
    for attr in re.findall(r'class="([^"]+)"', text):
        found.update(attr.split())
    for arg in re.findall(r'classList\.(?:add|remove|toggle)\(\s*"([\w-]+)"', text):
        found.add(arg)
    for arg in re.findall(r'className = "([\w -]+)"', text):
        found.update(arg.split())
    return found


def served_markup(page: str) -> str:
    """The page as the server sends it, with its placeholders resolved.

    The audit page ships as a template so a switched-off API cannot look like a
    rejected token; checking the raw file would flag the placeholder names as
    undefined classes.
    """
    raw = asset(page)
    out = ""
    for mode in ("token", "open", "off"):
        gate = "gate" if mode == "token" else "gate locked"
        off = "gate" if mode == "off" else "gate locked"
        out += (
            raw.replace("__MODE__", mode)
            .replace("__GATE_CLASS__", gate)
            .replace("__OFF_CLASS__", off)
        )
    return out


# Selectors expected beyond the pre-extraction baseline, which was captured
# before this branch was rebased onto the diarization work:
#   - #r-ffmpeg.bad, .offscreen: the inline-style removal above
#   - .seg .sp*: main's speaker column, which landed while this branch was open
#   - .staged*, .runbar, .run-note, #start*: files are staged and sent by a
#     button now, instead of the drop starting the job on the spot
INDEX_SELECTORS_ADDED = {
    "#r-ffmpeg.bad",
    ".offscreen",
    ".seg .sp",
    ".seg .sp.s0",
    ".seg .sp.s1",
    ".seg .sp.s2",
    "#start",
    "#start:active",
    "#start:disabled",
    "#start:hover",
    ".run-note",
    ".runbar",
    ".staged",
    ".staged-name",
    ".staged-row",
    ".staged-row + .staged-row",
    ".staged-size",
    ".staged-x",
}


INDEX_SELECTORS_BEFORE = {
    "*",
    ".actions",
    ".adv-grid",
    ".adv-grid textarea",
    ".adv-grid textarea:focus-visible",
    ".adv-row",
    ".controls",
    ".empty",
    ".field-block",
    ".field-block > span",
    ".gate",
    ".gate .row",
    ".gate p",
    ".gate-err",
    ".hint",
    ".intake",
    ".intake p",
    ".intake small",
    ".intake.hot",
    ".intake:focus-visible",
    ".intake:hover",
    ".job",
    ".job-body",
    ".job-head",
    ".job-meta",
    ".job-name",
    ".lamp",
    ".lamp.bad",
    ".lamp.on",
    ".locked",
    ".meter",
    ".meter i",
    ".meter i.done",
    ".meter i.fail",
    ".meter i.lit",
    ".readout",
    ".readout dd",
    ".readout dl",
    ".readout dt",
    ".seg",
    ".seg .ts",
    ".seg .tx",
    ".status",
    ".status.err",
    ".sub",
    ".transcript",
    ".transcript.paused",
    ".transcript:empty",
    ".wrap",
    ":root",
    "body",
    "button",
    "button.ghost",
    "button.ghost:hover",
    "button:active",
    "button:focus-visible",
    "button:hover",
    "details.advanced",
    "details.advanced summary",
    "details.advanced summary::-webkit-details-marker",
    "details.advanced summary::before",
    "details.advanced summary:focus-visible",
    "details.advanced[open] summary::before",
    "h1",
    "html,body",
    "input[type=checkbox]",
    "input[type=file]",
    "input[type=number]",
    "input[type=number]:focus-visible",
    "input[type=password]",
    "input[type=password]:focus-visible",
    "label.field",
    "select",
    "select:focus-visible",
}

AUDIT_SELECTORS_BEFORE = {
    "#events",
    "*",
    ".chip",
    ".chip b",
    ".chips",
    ".controls",
    ".empty",
    ".ev",
    ".ev-event",
    ".ev-head",
    ".ev-src",
    ".ev-time",
    ".ev.bad",
    ".ev.good",
    ".ev.warn",
    ".gate",
    ".gate .row",
    ".gate p",
    ".gate-err",
    ".locked",
    ".meta",
    ".prompt",
    ".prompt b",
    ".sub",
    ".topbar",
    ".wrap",
    ":root",
    "a.back",
    "a.back:hover",
    "body",
    "button",
    "button.ghost",
    "button.ghost:hover",
    "button:active",
    "button:focus-visible",
    "button:hover",
    "h1",
    "html,body",
    "input[type=checkbox]",
    "input[type=password],input[type=text]",
    "input[type=password]:focus-visible,input[type=text]:focus-visible",
    "label.field",
    "select,input[type=number]",
    "select:focus-visible,input[type=number]:focus-visible",
}


# --------------------------------------------------------------------------- #
# The files exist and the server serves them
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name",
    [
        "index.html",
        "audit.html",
        "app.css",
        "index.css",
        "audit.css",
        "common.js",
        "index.js",
        "audit.js",
    ],
)
def test_every_asset_is_present(name):
    assert (STATIC / name).is_file(), f"static/{name} is missing"


@pytest.mark.parametrize(
    "path,needle",
    [("/", 'id="intake"'), ("/audit", "Audit trail")],
)
def test_the_pages_are_served(client, path, needle):
    response = client.get(path)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert needle in response.text


@pytest.mark.parametrize(
    "name,mime",
    [("app.css", "text/css"), ("common.js", "javascript"), ("index.js", "javascript")],
)
def test_static_assets_are_served_with_a_usable_type(client, name, mime):
    response = client.get(f"/static/{name}")

    assert response.status_code == 200
    assert mime in response.headers["content-type"]


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/audit",
        "/static/index.js",
        "/static/common.js",
        "/static/audit.js",
        "/static/app.css",
        "/static/index.css",
        "/static/audit.css",
    ],
)
def test_the_page_and_its_assets_are_revalidated(client, path):
    """A reload after an upgrade must not mix a new page with the old script.

    StaticFiles answers with ETag and Last-Modified but no Cache-Control, which
    lets a browser reuse its own copy without asking; the HTML routes send no
    validators at all, so those are always refetched. Together that served a new
    page next to the previous version of index.js — a Transcribe button that did
    nothing, and a bug already fixed in the script still happening, with nothing
    on screen saying which version was running. no-cache (not no-store) keeps the
    revalidation, and its 304, cheap.
    """
    response = client.get(path)

    assert response.headers.get("Cache-Control") == "no-cache", (
        f"{path} may be reused from the browser cache without asking"
    )


def test_revalidation_still_answers_304_with_the_header(client):
    """no-cache means revalidate, not re-download: the ETag must still work, and
    the 304 has to carry the header too or the copy goes back to being fresh."""
    first = client.get("/static/index.js")

    again = client.get(
        "/static/index.js", headers={"if-none-match": first.headers["etag"]}
    )

    assert again.status_code == 304
    assert again.headers.get("Cache-Control") == "no-cache"


def test_a_missing_static_dir_is_a_clear_error_not_a_blank_page(client, monkeypatch):
    monkeypatch.setattr(s, "STATIC_DIR", s.STATIC_DIR / "nope")

    response = client.get("/")

    assert response.status_code == 500
    assert "static" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# No rule was lost in the split
# --------------------------------------------------------------------------- #


def test_the_main_page_css_covers_every_selector_it_had_before():
    now = selectors(asset("app.css")) | selectors(asset("index.css"))
    # Plus the two the inline-style removal introduced (see INDEX_SELECTORS_ADDED).
    assert now - INDEX_SELECTORS_ADDED == INDEX_SELECTORS_BEFORE
    assert INDEX_SELECTORS_ADDED <= now


def test_the_audit_page_css_covers_every_selector_it_had_before():
    now = selectors(asset("app.css")) | selectors(asset("audit.css"))
    assert now == AUDIT_SELECTORS_BEFORE


def test_the_shared_rules_exist_once_not_twice():
    """The duplication this refactor removed must not creep back."""
    shared = selectors(asset("app.css"))
    assert shared, "app.css should carry the shared design system"
    for page_css in ("index.css", "audit.css"):
        overlap = shared & selectors(asset(page_css))
        assert not overlap, f"{page_css} re-declares shared rules: {sorted(overlap)}"


def test_a_name_that_merely_collides_stays_per_page():
    """`.wrap` is 660px on one page and 1040px on the other; it must not merge."""
    assert ".wrap" not in selectors(asset("app.css"))
    assert ".wrap" in selectors(asset("index.css"))
    assert ".wrap" in selectors(asset("audit.css"))


def test_locked_wins_against_the_rule_it_is_hiding():
    """`.locked` is a one-word off switch and has to beat the rule the element
    is styled by. A bare `.locked{display:none}` does not: `label.field` and
    `.adv-row` are element selectors in index.css, which loads after app.css, so
    the Precision field stayed visible on a server that pins precision and the
    speaker row stayed visible on one started with --no-diarize."""
    assert re.search(
        r"\.locked\s*\{\s*display:\s*none\s*!important", asset("app.css")
    ), ".locked must outrank the display rules it hides"

    # The two the bug was visible on, pinned by name so the reason is on record.
    html = asset("index.html")
    assert re.search(r'class="field locked"[^>]*id="compute-field"', html)
    assert re.search(r'class="adv-row"[^>]*id="diarize-row"', html)
    assert re.search(r"\.adv-row\{[^}]*display:flex", asset("index.css"))


# --------------------------------------------------------------------------- #
# Every class the markup uses is actually styled
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "html,css",
    [
        ("index.html", ("app.css", "index.css")),
        ("audit.html", ("app.css", "audit.css")),
    ],
)
def test_every_class_used_is_defined(html, css):
    used = classes_in(served_markup(html))
    defined: set[str] = set()
    for name in css:
        sheet = re.sub(r"/\*.*?\*/", "", asset(name), flags=re.S)
        defined.update(re.findall(r"\.([\w-]+)", sheet))

    missing = sorted(used - defined)
    assert not missing, f"{html} uses classes with no rule: {missing}"


# --------------------------------------------------------------------------- #
# The CSP no longer needs unsafe-inline
# --------------------------------------------------------------------------- #


def test_the_csp_allows_no_inline_script_or_style(client):
    csp = client.get("/").headers["Content-Security-Policy"]

    assert "script-src 'self'" in csp
    assert "style-src 'self'" in csp
    assert "unsafe-inline" not in csp
    assert "unsafe-eval" not in csp


@pytest.mark.parametrize("page", ["index.html", "audit.html"])
def test_the_markup_carries_no_inline_code(page):
    html = asset(page)

    assert "<style" not in html, "styles must live in a stylesheet"
    assert "<script>" not in html, "scripts must live in a file"
    assert not re.search(r"\son\w+=", html), "no inline event handlers"
    assert not re.search(r"\sstyle=", html), "no inline style attributes"


@pytest.mark.parametrize("page", ["index.html", "audit.html"])
def test_the_pages_load_the_shared_helpers_first(page):
    html = asset(page)

    assert '<script src="/static/common.js"></script>' in html
    assert html.index("common.js") < html.index(page.replace(".html", ".js")), (
        "common.js must be loaded before the page script that uses it"
    )


# --------------------------------------------------------------------------- #
# The escaping helper exists once
# --------------------------------------------------------------------------- #


def test_esc_is_defined_only_in_common_js():
    """It is the only thing between a filename and stored XSS; one copy only."""
    definers = [
        name
        for name in ("common.js", "index.js", "audit.js")
        if re.search(r"const esc\s*=", asset(name))
    ]
    assert definers == ["common.js"], f"esc is defined in {definers}"


def test_neither_page_script_handles_token_storage():
    """tokenStore owns that, so the two keys cannot drift apart.

    The Follow switch keeps its own sessionStorage entry, which is fine and is
    why this checks the token keys rather than the storage API.
    """
    for name, key in (("index.js", "tk"), ("audit.js", "atk")):
        script = asset(name)
        assert f'sessionStorage.setItem("{key}"' not in script
        assert f'sessionStorage.getItem("{key}"' not in script


def test_the_pages_use_different_storage_keys():
    """An app token and an audit token must not overwrite each other."""
    assert 'tokenStore("tk")' in asset("index.js")
    assert 'tokenStore("atk")' in asset("audit.js")


def test_every_at_rule_is_one_a_browser_can_parse():
    """`@media @media (...)` is not an error anyone sees: the browser drops the
    whole block, and the reduced-motion rule quietly never applied."""
    for name in ("app.css", "index.css", "audit.css"):
        css = asset(name)
        assert re.search(r"@(\w+)\s+@", css) is None, f"a doubled at-rule in {name}"
    assert "@media (prefers-reduced-motion:reduce)" in asset("app.css")
