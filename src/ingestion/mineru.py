from __future__ import annotations

import shutil
import site
import subprocess
from pathlib import Path
from typing import List


def _resolve_mineru_command_template(command_template: str) -> str:
    """
    On Windows, `magic-pdf.exe` is often installed under user-site Scripts
    but that path is not in PATH. If so, rewrite the command template.
    """
    if not command_template.strip().startswith("magic-pdf"):
        return command_template
    if shutil.which("magic-pdf"):
        return command_template

    scripts_dir = Path(site.getuserbase()) / "Python311" / "Scripts"
    exe_path = scripts_dir / "magic-pdf.exe"
    if exe_path.exists():
        return command_template.replace("magic-pdf", f'"{exe_path}"', 1)
    return command_template


def run_mineru_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    command_template: str,
) -> int:
    """
    Run MinerU conversion for every PDF under input_dir.

    command_template example:
      magic-pdf -p "{input_file}" -o "{output_dir}"
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"MinerU input dir not found: {input_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_files: List[Path] = sorted(input_path.glob("**/*.pdf"))
    if not pdf_files:
        return 0

    command_template = _resolve_mineru_command_template(command_template)
    success_count = 0
    for pdf in pdf_files:
        cmd = command_template.format(
            input_file=str(pdf),
            output_dir=str(output_path),
        )
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        logs = f"{proc.stdout}\n{proc.stderr}"
        # magic-pdf CLI may swallow exceptions and still exit with code 0.
        has_error_in_logs = ("ERROR" in logs) or ("Traceback" in logs) or ("ModuleNotFoundError" in logs)
        if proc.returncode == 0 and not has_error_in_logs:
            success_count += 1
        else:
            raise RuntimeError(
                f"MinerU command failed for {pdf}\n"
                f"Command: {cmd}\n"
                f"STDOUT: {proc.stdout}\n"
                f"STDERR: {proc.stderr}"
            )
    return success_count
