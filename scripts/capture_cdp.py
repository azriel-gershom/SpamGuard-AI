"""High-precision CDP screenshot tool for capturing Ham live scan."""

import asyncio
import base64
import json
import subprocess
import time
import urllib.request
from pathlib import Path
import websockets

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
SCREENSHOTS_DIR = Path(__file__).resolve().parent.parent / "docs" / "screenshots"

async def send_command(ws, req_id, method, params=None):
    payload = {"id": req_id, "method": method}
    if params:
        payload["params"] = params
    await ws.send(json.dumps(payload))
    while True:
        raw = await ws.recv()
        msg = json.loads(raw)
        if msg.get("id") == req_id:
            return msg

async def capture_ham():
    proc = subprocess.Popen([
        CHROME_PATH,
        "--headless",
        "--disable-gpu",
        "--remote-debugging-port=9222",
        "--window-size=1920,1080",
        "about:blank",
    ])
    try:
        await asyncio.sleep(2)
        req = urllib.request.Request("http://127.0.0.1:9222/json/new?http://localhost:3000", method="PUT")
        with urllib.request.urlopen(req) as resp:
            page_info = json.loads(resp.read().decode("utf-8"))
        ws_url = page_info["webSocketDebuggerUrl"]

        async with websockets.connect(ws_url, max_size=50_000_000) as ws:
            await send_command(ws, 1, "Page.enable")
            await send_command(ws, 2, "Runtime.enable")
            await asyncio.sleep(3)

            # Scroll to scanner and click Ham sample
            await send_command(ws, 10, "Runtime.evaluate", {
                "expression": "document.querySelector('#scanner').scrollIntoView({behavior: 'instant'});"
            })
            await asyncio.sleep(1)

            # Click Ham template button
            await send_command(ws, 11, "Runtime.evaluate", {
                "expression": "Array.from(document.querySelectorAll('button')).find(b => b.textContent.includes('Engineering Standup'))?.click();"
            })
            await asyncio.sleep(0.5)

            # Click Analyze Email button
            await send_command(ws, 12, "Runtime.evaluate", {
                "expression": "Array.from(document.querySelectorAll('button')).find(b => b.textContent.includes('Analyze Email'))?.click();"
            })
            await asyncio.sleep(2)

            # Capture live Ham scan result
            reply = await send_command(ws, 13, "Page.captureScreenshot", {"format": "png"})
            if "result" in reply and "data" in reply["result"]:
                img_data = base64.b64decode(reply["result"]["data"])
                out_file = SCREENSHOTS_DIR / "04_email_scanner_result_ham.png"
                out_file.write_bytes(img_data)
                print(f"[OK] Live Ham scan captured: {out_file.name} ({len(img_data):,} bytes)")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    asyncio.run(capture_ham())
