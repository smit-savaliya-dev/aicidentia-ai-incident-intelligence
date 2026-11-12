"""Record the 4-minute demo click path as a backup video: reports/demo/backup_demo.webm.

Usage:  uv run python scripts/record_demo.py              # about 4 minutes, 1600 x 900
        uv run python scripts/record_demo.py --fast       # same path with short pauses (a quick check)

Starts the dashboard locally, plays the click path from docs/presentation.md in a headless browser with
Playwright video recording, and saves the video. Use it if the live demo fails. The pauses are where you talk.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from typing import Any

from aicidentia.config import PROJECT_ROOT, configure_logging
from aicidentia.dashboard.browser import dashboard_server, launch_browser, wait_until_rendered

OUT_DIR = PROJECT_ROOT / "reports" / "demo"
OUT = OUT_DIR / "backup_demo.webm"
SIZE = {"width": 1600, "height": 900}


def scroll(page: Any, pixels: int, steps: int = 6, pause: float = 0.25) -> None:
    """Smooth scroll of Streamlit's main area."""
    for _ in range(steps):
        page.mouse.wheel(0, pixels / steps)
        time.sleep(pause)


def ensure_ffmpeg() -> None:
    """Playwright records video with its own ffmpeg build; install it once if missing (PLAYWRIGHT_BROWSERS_PATH
    in .env decides where)."""
    import subprocess

    subprocess.run([sys.executable, "-m", "playwright", "install", "ffmpeg"], check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="short pauses (about 1 minute)")
    args = parser.parse_args(argv)
    configure_logging()
    from playwright.sync_api import sync_playwright

    k = 0.25 if args.fast else 1.0

    def hold(seconds: float) -> None:
        time.sleep(seconds * k)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = OUT_DIR / "_raw"
    shutil.rmtree(raw_dir, ignore_errors=True)
    started = time.time()
    ensure_ffmpeg()
    with dashboard_server() as url, sync_playwright() as pw:
        browser = launch_browser(pw)
        warm = browser.new_page()  # fill the caches first so the recording shows fast pages
        warm.goto(url)
        wait_until_rendered(warm)
        warm.goto(f"{url}/Precedent_Finder")
        wait_until_rendered(warm)
        warm.locator("button", has_text="·").first.click()
        warm.get_by_text("What went wrong before").wait_for(timeout=180_000)
        warm.close()

        context = browser.new_context(
            viewport=SIZE, record_video_dir=str(raw_dir), record_video_size=SIZE
        )
        page = context.new_page()
        page.mouse.move(800, 450)

        # 0:00 Home: five findings (about 30 s)
        page.goto(url)
        wait_until_rendered(page, settle=1)
        hold(12)
        scroll(page, 500)
        hold(15)

        # 0:30 Themes and Trends: heatmap, emerging table, ChatGPT shift (about 45 s)
        page.get_by_role("link", name="Themes and Trends").first.click()
        wait_until_rendered(page, settle=1)
        hold(15)
        scroll(page, 750)
        hold(20)
        scroll(page, -750)
        hold(5)

        # 1:15 Risk Priority Matrix (about 25 s)
        page.get_by_role("link", name="Risk Priority Matrix").first.click()
        wait_until_rendered(page, settle=1)
        scroll(page, 350)
        hold(20)

        # 1:40 Risk Network: Meta's ego network, then search 'Tesla' (about 35 s)
        page.get_by_role("link", name="Risk Network").first.click()
        wait_until_rendered(page, settle=1)
        hold(15)
        box = page.get_by_placeholder("e.g. Meta, Tesla, deepfake")
        box.click()
        box.type("Tesla", delay=120)
        box.press("Enter")
        wait_until_rendered(page, settle=1)
        hold(15)

        # 2:15 Precedent Finder: demo case, prediction, five earlier incidents (about 60 s)
        page.get_by_role("link", name="Precedent Finder").first.click()
        wait_until_rendered(page, settle=1)
        hold(5)
        page.locator("button", has_text="·").first.click()
        page.get_by_text("What went wrong before").wait_for(timeout=180_000)
        wait_until_rendered(page, settle=1)
        hold(20)
        scroll(page, 700)
        hold(20)
        scroll(page, 700)
        hold(10)

        # 3:15 Label Reliability (about 30 s)
        page.get_by_role("link", name="Label Reliability").first.click()
        wait_until_rendered(page, settle=1)
        hold(15)
        scroll(page, 650)
        hold(12)

        # 3:45 Method and Limitations: active models, limitations (about 15 s)
        page.get_by_role("link", name="Method and Limitations").first.click()
        wait_until_rendered(page, settle=1)
        hold(6)
        scroll(page, 1500)
        hold(8)

        video = page.video
        context.close()  # finalises the video file
        source = video.path()
        browser.close()
    OUT.unlink(missing_ok=True)
    shutil.move(source, OUT)
    shutil.rmtree(raw_dir, ignore_errors=True)
    minutes = (time.time() - started) / 60
    print(
        f"Saved {OUT.relative_to(PROJECT_ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB; run took {minutes:.1f} min)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
