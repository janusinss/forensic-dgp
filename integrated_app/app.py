"""Compatibility ASGI entrypoint for the IMPLEMENTATION branch.

Use install.bat, then start.bat. Model inference runs only after an explicit
reviewed request, in a disposable CPU worker.
"""
from implementation.app import app
from implementation.completion_api import register

register(app)

from implementation.capture_api import register as register_capture
register_capture(app)
