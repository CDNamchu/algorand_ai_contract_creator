import subprocess
import tempfile
import logging


def auto_format_code(code_str: str) -> str:
    """
    Auto-format Python code string using black formatter.
    If black fails for any reason, return the original code and log the error.
    """
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.py') as tmp_file:
        tmp_file.write(code_str)
        tmp_file.flush()
        try:
            # Run black formatter on the temp file (silently)
            subprocess.run(['black', tmp_file.name], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            tmp_file.seek(0)
            formatted_code = tmp_file.read()
            return formatted_code
        except subprocess.CalledProcessError as e:
            # Black failed (parse error etc). Log stderr for diagnostics and return original code
            stderr = None
            try:
                stderr = e.stderr.decode('utf-8', errors='replace') if getattr(e, 'stderr', None) else None
            except Exception:
                stderr = str(e)
            logging.warning("Black formatting failed (CalledProcessError). Returning unformatted code. stderr: %s", stderr)
            tmp_file.seek(0)
            return tmp_file.read()
        except FileNotFoundError:
            # Black not installed / not found in PATH
            logging.warning("Black not found in PATH. Skipping formatting.")
            tmp_file.seek(0)
            return tmp_file.read()
        except Exception:
            logging.exception("Unexpected error while running black; returning unformatted code.")
            tmp_file.seek(0)
            return tmp_file.read()

def custom_syntax_fix(code_str: str) -> str:
    """
    Apply custom syntax fixes for common AI-generated code issues.
    Extend with regex or parsing-based fixes for known recurrent patterns.

    Example fixes might include:
    - Removing stray trailing commas
    - Closing unclosed brackets (requires complex parsing)
    - Fixing whitespace around operators
    """
    # For now, no custom fixes implemented; return input as-is.
    return code_str
