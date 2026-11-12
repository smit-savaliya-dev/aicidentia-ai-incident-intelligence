"""Run the dashboard on a local port and drive it with Playwright (screenshots and the backup demo video).

Browser choice: Playwright's own Chromium if it is installed, otherwise the Chrome or Edge already on the machine
(no download), and only as a last resort `playwright install chromium` (set PLAYWRIGHT_BROWSERS_PATH in .env to keep
it off a small system drive).
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from typing import Any

from aicidentia.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

PAGES = [
    ("01_home", "", "Home"),
    ("02_themes_and_trends", "Themes_and_Trends", "Themes and Trends"),
    ("03_risk_priority_matrix", "Risk_Priority_Matrix", "Risk Priority Matrix"),
    ("04_risk_network", "Risk_Network", "Risk Network"),
    ("05_precedent_finder", "Precedent_Finder", "Precedent Finder"),
    ("06_label_reliability", "Label_Reliability", "Label Reliability"),
    ("07_method_and_limitations", "Method_and_Limitations", "Method and Limitations"),
]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextlib.contextmanager
def dashboard_server(port: int | None = None, timeout: float = 90) -> Iterator[str]:
    """Start `streamlit run app/Home.py` headless and yield its base URL; stop it afterwards."""
    port = port or free_port()
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "app" / "Home.py"),
        "--server.headless=true",
        f"--server.port={port}",
        "--server.address=127.0.0.1",
        "--browser.gatherUsageStats=false",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=os.environ.copy(),
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError("the dashboard did not start")
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


def launch_browser(playwright: Any, headless: bool = True) -> Any:
    """Bundled Chromium, else installed Chrome or Edge, else install Chromium."""
    attempts = [{}, {"channel": "chrome"}, {"channel": "msedge"}]
    for kwargs in attempts:
        try:
            browser = playwright.chromium.launch(headless=headless, **kwargs)
            logger.info("Browser: %s", kwargs.get("channel", "playwright chromium"))
            return browser
        except Exception as exc:  # not installed: try the next option
            logger.info(
                "Browser %s unavailable: %s", kwargs or "chromium", str(exc).splitlines()[0]
            )
    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
    return playwright.chromium.launch(headless=headless)


def wait_until_rendered(page: Any, settle: float = 2.0, timeout: float = 120) -> None:
    """Wait until Streamlit has finished running the script and charts have drawn."""
    page.wait_for_selector("[data-testid='stMain']", timeout=timeout * 1000)
    deadline = time.time() + timeout
    time.sleep(1.0)
    while time.time() < deadline:
        running = page.locator("[data-testid='stStatusWidget']").count()
        if not running:
            break
        time.sleep(0.5)
    page.wait_for_load_state("networkidle")
    time.sleep(settle)


def full_height(page: Any) -> int:
    return int(
        page.evaluate(
            "() => Math.max(document.body.scrollHeight,"
            " ...Array.from(document.querySelectorAll('[data-testid=stMain], section.main, .main'))"
            ".map(e => e.scrollHeight))"
        )
    )
