# crawler/handlers/save_html.py

from typing import Any

import os
import json
import asyncio
import hashlib

from utils.logger import log


class SaveHTMLHandler:

    def __init__(
            self,
            size: int = 20,
            output_dir: str = "data"
    ):
        self.size = size
        self.output_dir = output_dir
        self.saved_count = 0

        os.makedirs(self.output_dir, exist_ok=True)

    async def __call__(
            self,
            html: str,
            fetch_info: dict[str, Any]
    ) -> None:
        if self.saved_count >= self.size:
            return

        url = fetch_info['url']

        # Generate MD5 from URL
        file_id = hashlib.md5(url.encode("utf-8")).hexdigest()
        html_path = os.path.join(self.output_dir, f"{file_id}.html")
        meta_path = os.path.join(self.output_dir, f"{file_id}.json")

        # Write
        try:
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)

            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(fetch_info, indent=2, ensure_ascii=False))

            self.saved_count += 1
        except Exception as e:
            log(f"ERROR saving {url}: {e}", "WARNING")

        # Yield control to event loop
        await asyncio.sleep(0)
