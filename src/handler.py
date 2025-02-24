""" Example handler file. """

import runpod
import torch
import base64
import io
import sys
import time
import subprocess
import os
import tempfile
import numpy as np


def scriptHandler(event):
    try:
        script = event.get("input", {}).get("script", "")
        if not script:
            return {"error": "No script provided"}

        # Redirect stdout to capture print output
        stdout_backup = sys.stdout
        sys.stdout = io.StringIO()

        # Create a shared dictionary to capture JSON output
        exec_globals = {"event": event, "output": {}}

        try:
            exec(script, exec_globals)  # Execute script with access to `event` and `output`
            output_json = exec_globals["output"]  # Extract JSON output
            output_text = sys.stdout.getvalue()  # Extract printed output
        finally:
            sys.stdout = stdout_backup  # Restore original stdout

        return {
            "stdout": output_text.strip(),
            "output": output_json,  # JSON output from script
            "stderr": "",
            "return_code": 0
        }

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": scriptHandler})
