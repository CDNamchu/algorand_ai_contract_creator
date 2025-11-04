import subprocess
import tempfile

def auto_format_code(code_str: str) -> str:
    """
    Auto-format Python code string using black formatter.
    This function saves the code to a temporary file, runs black,
    then reads back the formatted result.
    """
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.py') as tmp_file:
        tmp_file.write(code_str)
        tmp_file.flush()
        # Run black formatter on the temp file silently
        subprocess.run(['black', tmp_file.name], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tmp_file.seek(0)
        formatted_code = tmp_file.read()
    return formatted_code

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
