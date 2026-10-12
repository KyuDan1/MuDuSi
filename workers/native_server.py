"""Run upstream PersonaPlex with the requested bind address enforced.

The upstream server accepts --host but omits it from aiohttp.web.run_app.
Keep the correction at the launcher boundary without editing shared checkouts.
"""
import functools
import runpy
import sys

from aiohttp import web


if __name__ == "__main__":
    host = "127.0.0.1"
    if "--host" in sys.argv:
        host = sys.argv[sys.argv.index("--host") + 1]
    web.run_app = functools.partial(web.run_app, host=host)
    runpy.run_module("moshi.server", run_name="__main__")
