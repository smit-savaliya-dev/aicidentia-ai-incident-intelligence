"""Save a high-resolution PNG of every dashboard page to reports/figures/dashboard/ (for slides and the report).

Usage:  uv run python scripts/capture_dashboard.py            # 1600 px wide, 2x pixel density
        uv run python scripts/capture_dashboard.py --scale 1  # smaller files

Starts the dashboard on a free local port, opens each page in a headless browser (Playwright: its Chromium, or the
Chrome / Edge already installed), waits for the charts, and captures the whole page. Also captures the Precedent
Finder after pressing the first demo case. Prints how long each page took to render.
"""

from __future__ import annotations

import argparse
import sys
import time

from aicidentia.config import PROJECT_ROOT, configure_logging
from aicidentia.dashboard.browser import (
    PAGES,
    dashboard_server,
    full_height,
    launch_browser,
    wait_until_rendered,
)

OUT = PROJECT_ROOT / "reports" / "figures" / "dashboard"
WIDTH = 1600


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=float, default=2.0, help="device pixel ratio (2 = retina)")
    args = parser.parse_args(argv)
    configure_logging()
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    with dashboard_server() as url, sync_playwright() as pw:
        browser = launch_browser(pw)
        context = browser.new_context(
            viewport={"width": WIDTH, "height": 1000}, device_scale_factor=args.scale
        )
        page = context.new_page()
        page.goto(url)  # warm-up: first load imports everything and fills the caches
        wait_until_rendered(page)
        for name, path, label in PAGES:
            start = time.perf_counter()
            page.set_viewport_size({"width": WIDTH, "height": 1000})
            page.goto(f"{url}/{path}")
            page.get_by_text("reported incidents only").first.wait_for(timeout=120_000)
            seconds = time.perf_counter() - start  # until the footer (end of the page) is on screen
            wait_until_rendered(page)
            if name == "05_precedent_finder":
                page.locator("button", has_text="·").first.click()
                page.get_by_text("What went wrong before").wait_for(timeout=180_000)
                wait_until_rendered(page, settle=3.0)
            page.set_viewport_size({"width": WIDTH, "height": min(full_height(page) + 40, 9000)})
            wait_until_rendered(page, settle=1.5)
            target = OUT / f"{name}.png"
            page.screenshot(path=str(target), full_page=True)
            written.append(target)
            print(
                f"{label:<26} footer visible after {seconds:4.1f}s -> {target.relative_to(PROJECT_ROOT)}"
            )
        browser.close()
    print(f"{len(written)} screenshots in {OUT.relative_to(PROJECT_ROOT)}")
    return 0 if len(written) == len(PAGES) else 1


if __name__ == "__main__":
    sys.exit(main())
