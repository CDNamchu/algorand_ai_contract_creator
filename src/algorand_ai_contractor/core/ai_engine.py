"""
AI Contract Generator with Multi-Layer Validation
Compliance: EU AI Act Tier 2, IEEE EAD
Supports: OpenAI GPT-4 and Perplexity AI
"""

from openai import OpenAI
import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Optional
import subprocess
import tempfile
import traceback
from .syntax_fixer import auto_format_code, custom_syntax_fix


load_dotenv()

# Configure API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
AI_PROVIDER = os.getenv('AI_PROVIDER', 'perplexity')

# Configure structured logging
logging.basicConfig(
    filename='ai_generations.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

from pyteal import compileTeal, Mode

# -------------------- Syntax Fixer --------------------
# Use formatter/fixer implementation from `src/.../syntax_fixer.py` (imported above).

# -------------------- Validation Functions --------------------

def check_python_syntax(code_str: str) -> None:
    """
    Raises SyntaxError if the Python code string has syntax errors.
    """
    compile(code_str, "<string>", "exec")

def check_pyteal_compilation(pyteal_program) -> None:
    """
    Raises Exception if PyTeal program fails compilation to TEAL.
    """
    _ = compileTeal(pyteal_program, mode=Mode.Application, version=6)

# -------------------- Contract Generator --------------------

class ContractGenerator:
    """Deterministic PyTeal code generator with self-correction loop."""

    SYSTEM_PROMPT = """You are an expert Algorand blockchain developer specialized in PyTeal smart contracts.

YOU HAVE ACCESS TO REAL-TIME WEB SEARCH (if using Perplexity). If you need to verify PyTeal syntax or latest API changes, search for official documentation.

*CRITICAL REQUIREMENTS:*
1. Generate ONLY valid PyTeal code compatible with pyteal v0.24.0
2. Use proper approval/clear program structure
3. Include comprehensive inline comments
4. Follow Algorand ASC1 security standards
5. Avoid:
   - Hardcoded addresses or keys
   - Unbounded loops
   - Reentrancy vulnerabilities
   - Unsafe global state manipulation
   - Integer overflow risks
6. Always include proper fee checks and transaction validation
7. Use defensive programming patterns
8. NEVER use variable assignments inside And() or other expression contexts (use ScratchVar or separate statements)
9. Return ONLY the Python code - NO markdown code fences, NO ``` markers

*GROUPED TRANSACTION REQUIREMENTS:*
- When using grouped transactions (Global.group_size() > 1), ALWAYS assert Txn.group_index() to prevent reordering attacks
- Example: Assert(Txn.group_index() == Int(0)) to require app call is first in group
- Never assume transaction order in a group without explicit group_index checks

*OPT-IN AND STATE REQUIREMENTS:*
- Before using App.localPut() or App.localGet(), verify the account has opted in using App.optedIn(account, Int(0))
- For contracts that receive ASAs, document that the application account must opt-in to the ASA before use
- Always validate opt-in state before reading/writing local state

*CLOSEOUT PROTECTION:*
- If contract manages locked funds or tokens, prevent CloseOut when user has non-zero locked balances
- Example: Assert(App.localGet(Txn.sender(), LOCKED_AMOUNT) == Int(0)) before allowing closeout

*OUTPUT STRUCTURE:*
1. Complete PyTeal source code (plain Python, NO code fences)
2. Contract purpose summary (2-3 sentences)
3. Logic walkthrough (key conditions and branches)
4. Security considerations
5. Deployment parameters needed
"""

    def __init__(self, model: str = "sonar", temperature: float = 0.2):
        self.model = model
        self.temperature = temperature
        self.generation_history = []
        self.ai_provider = AI_PROVIDER
        self.client = None

    def _get_client(self, provider: str) -> OpenAI:
        """Get configured OpenAI client for different providers."""
        if provider == 'perplexity':
            return OpenAI(
                api_key=PERPLEXITY_API_KEY,
                base_url="https://api.perplexity.ai"
            )
        return OpenAI(api_key=OPENAI_API_KEY)

    def _get_model(self, provider: str, model: str) -> str:
        """Get appropriate model name for provider - uses correct Perplexity model names."""
        if provider == 'perplexity':
            if model in ['sonar', 'sonar-pro']:
                return model
            return 'sonar'
        else:  # openai
            if not model.startswith('gpt'):
                return "gpt-4"
        return model

    def generate_pyteal_contract(
        self,
        description: str,
        max_retries: int = 3,
        ai_provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generate PyTeal contract with automatic validation and retry.
        """
        provider = ai_provider or self.ai_provider
        selected_model = self._get_model(provider, model or self.model)
        client = self._get_client(provider)

        attempt = 0
        last_error = None

        while attempt < max_retries:
            try:
                logging.info(
                    f"Generation attempt {attempt + 1} for: "
                    f"{description[:100]} using {provider}/{selected_model}"
                )

                response = client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": self._build_user_prompt(description, last_error)}
                    ],
                    temperature=self.temperature,
                    max_tokens=2000
                )

                raw_output = response.choices[0].message.content
                logging.debug("Raw AI output:\n%s", raw_output)
                parsed = self._parse_ai_response(raw_output)
                logging.debug("Parsed code from AI response:\n%s", parsed['code'])
                # Sanitize the parsed code to remove common generator artifacts
                parsed['code'] = self._sanitize_code(parsed['code'])

                # **Syntax Fix Pipeline:**
                # Step 1. Auto-format code
                logging.debug("Code before fixer:\n%s", parsed.get('code'))
                formatted_code = auto_format_code(parsed['code'])
                # Step 2. Apply custom syntax fixes if any
                logging.debug("Code after auto-format:\n%s", formatted_code)
                fixed_code = custom_syntax_fix(formatted_code)
                logging.debug("Code after custom syntax fix:\n%s", fixed_code)

                # Validate after fixing
                validation_result = self._validate_pyteal_syntax(fixed_code)

                if not validation_result['valid']:
                    last_error = validation_result['error']
                    attempt += 1
                    logging.warning(f"Validation failed: {last_error}")
                    continue

                # Validate python syntax explicitly
                try:
                    check_python_syntax(fixed_code)
                except SyntaxError as e:
                    last_error = str(e)
                    attempt += 1
                    logging.warning(f"Python syntax error: {last_error}")
                    continue

                # Exec and compile PyTeal program to validate deeper
                try:
                    pyteal_program = self._compile_program_from_code(fixed_code)
                    check_pyteal_compilation(pyteal_program)
                except Exception as e:
                    last_error = str(e)
                    attempt += 1
                    logging.warning(f"PyTeal compilation error: {last_error}")
                    continue

                # On success log and return
                self._log_generation(
                    description, parsed, attempt + 1, provider, selected_model
                )
                return {
                    'success': True,
                    'code': fixed_code,
                    'explanation': parsed['explanation'],
                    'deployment': parsed['deployment'],
                    'audit': parsed['audit'],
                    'metadata': {
                        'model': selected_model,
                        'provider': provider,
                        'attempts': attempt + 1,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                }

            except Exception as e:
                # Log full traceback for easier debugging and propagate a detailed last_error
                logging.exception("Generation error")
                last_error = traceback.format_exc()
                attempt += 1

        return {
            'success': False,
            'error': f"Failed after {max_retries} attempts. Last error: {last_error}",
            'partial_code': None
        }

    def _build_user_prompt(self, description: str, previous_error: str = None) -> str:
        """Construct user prompt with self-correction context."""
        base = f"""Generate a PyTeal smart contract for the following requirement:

{description}

Ensure the contract is production-ready and follows all security guidelines."""
        if previous_error:
            base += f"""

PREVIOUS ATTEMPT FAILED WITH ERROR:
{previous_error}
"""
        return base

    def _parse_ai_response(self, raw_output: str) -> Dict[str, str]:
        """Parse the AI response and extract the PyTeal code block."""
        code = raw_output
        explanation = ""
        deployment = ""
        audit = ""

        if "```" in raw_output:
            start = raw_output.find("```")
            end = raw_output.find("```", start + 3)
            if end != -1:
                inner = raw_output[start + 3:end]
                inner = inner.lstrip()
                if inner.startswith("python"):
                    inner = inner[len("python"):].lstrip()
                code = inner.strip()
                tail = raw_output[end + 3 :].strip()
                if tail:
                    explanation = tail
        else:
            for sep in ["\n\n---", "\n---", "\n**Contract Purpose Summary:", "\n**Contract Purpose Summary**"]:
                if sep in raw_output:
                    parts = raw_output.split(sep, 1)
                    code = parts[0].strip()
                    explanation = parts[1].strip() if len(parts) > 1 else ""
                    break

        # final fallback: if nothing parsed, treat entire output as code
        if not code and raw_output:
            code = raw_output.strip()

        return {"code": code, "explanation": explanation, "deployment": deployment, "audit": audit}

    def _validate_pyteal_syntax(self, code: str) -> Dict[str, str]:
        """Basic safety and content checks on the generated code."""
        # Split code into main block and contract logic
        main_block_start = code.find('if __name__ == "__main__"')
        if main_block_start == -1:
            main_block_start = code.find("if __name__ == '__main__'")
        
        # Check contract logic (before __main__ block) for dangerous patterns
        contract_code = code[:main_block_start] if main_block_start != -1 else code
        
        # These patterns are dangerous in contract logic but OK in test harness
        dangerous_in_contract = ["eval(", "exec(", "subprocess", "os.system", "pickle.loads"]
        for pat in dangerous_in_contract:
            if pat in contract_code:
                return {"valid": False, "error": f"Dangerous pattern detected in contract logic: {pat}"}
        
        # open() is only allowed in __main__ test block
        if "open(" in contract_code:
            return {"valid": False, "error": "Dangerous pattern detected in contract logic: open()"}

        lower = code.lower()
        if "from pyteal" in lower or "import pyteal" in lower or "txn" in lower or "app.globalput" in lower:
            return {"valid": True}

        return {"valid": False, "error": "Missing expected PyTeal content or approval program."}

    def _sanitize_code(self, code: str) -> str:
        """Sanitize AI-generated code before saving/validating."""
        import re

        if not code:
            return code

        sanitized = code.strip()
        
        # Remove markdown code fences more aggressively
        # Handle cases like: ```python\ncode``` or ```\ncode```
        if sanitized.startswith('```'):
            # Remove opening fence
            sanitized = sanitized[3:]
            # Remove language identifier (python, py, etc)
            sanitized = re.sub(r'^\s*(?:python|py)\s*\n', '', sanitized, flags=re.IGNORECASE)
            # Remove closing fence if present
            if '```' in sanitized:
                sanitized = sanitized[:sanitized.rfind('```')]
            sanitized = sanitized.strip()

        # Legacy fence extraction (if not handled above)
        if '```' in sanitized:
            start = sanitized.find('```')
            end = sanitized.find('```', start + 3)
            if end != -1:
                inner = sanitized[start + 3:end]
                inner = re.sub(r'^\s*python\s*\n', '', inner, flags=re.IGNORECASE)
                sanitized = inner.strip()

        # Remove trailing explanation sections
        for sep in ['\n\n---', '\n---', '\n**Contract Purpose Summary:', '\n**Logic Walkthrough:', '\n**Security Considerations:']:
            if sep in sanitized:
                sanitized = sanitized.split(sep, 1)[0]

        def _addr_repl(m: re.Match) -> str:
            inner = m.group(1).strip()
            if re.match(r'^["\']', inner):
                return f'Addr({inner})'
            return inner

        sanitized = re.sub(r'Addr\(([^)]+)\)', _addr_repl, sanitized)
        sanitized = sanitized.strip() + '\n'

        return sanitized

    def _compile_program_from_code(self, code: str):
        """
        Execute the code string safely and extract approval_program callable.
        """
        import inspect
        from pyteal import Expr

        namespace = {}
        # Execute the generated code in a fresh namespace
        exec(code, namespace)

        # 1) Look for common names
        for name in ('approval_program', 'router', 'app'):
            if name in namespace:
                obj = namespace[name]
                # If it's a zero-arg callable, call it
                try:
                    if callable(obj):
                        sig = None
                        try:
                            sig = inspect.signature(obj)
                        except (ValueError, TypeError):
                            sig = None
                        if sig is None or len(sig.parameters) == 0:
                            return obj()
                    return obj
                except Exception:
                    # If calling fails, continue to other discovery heuristics
                    logging.exception("Error while resolving candidate '%s'", name)

        # 2) Look for a pyteal.Expr directly in the namespace
        for obj in namespace.values():
            try:
                if isinstance(obj, Expr):
                    return obj
            except Exception:
                continue

        # 3) Try zero-arg callables and test-compile their return value
        for obj in namespace.values():
            if callable(obj):
                try:
                    sig = None
                    try:
                        sig = inspect.signature(obj)
                    except (ValueError, TypeError):
                        sig = None
                    if sig is None or len(sig.parameters) == 0:
                        candidate = obj()
                        try:
                            # quick compile test to verify it's a PyTeal program
                            compileTeal(candidate, Mode.Application, version=6)
                            return candidate
                        except Exception:
                            continue
                except Exception:
                    continue

        # If we reach here, no suitable program was found — include namespace keys in error
        keys = ','.join(sorted(list(namespace.keys())))
        raise RuntimeError(f"Generated code does not define a PyTeal program. Namespace keys: {keys}")

    def _log_generation(
        self,
        description: str,
        parsed: Dict[str, str],
        attempt: int,
        provider: str,
        model: str
    ) -> None:
        """Log successful generations to file."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "description": description,
            "attempt": attempt,
            "provider": provider,
            "model": model,
            "code_snippet": parsed['code'][:200]
        }
        logging.info(json.dumps(log_entry, indent=2))


# ---------------------------------------------------------------------
# Add-on utility function for contract explanation
# ---------------------------------------------------------------------

def explain_contract(code: str, ai_provider: Optional[str] = None) -> str:
    """
    Use AI to provide human-readable explanation of existing PyTeal code.
    """
    try:
        provider = ai_provider or AI_PROVIDER

        if provider == 'perplexity':
            client = OpenAI(
                api_key=PERPLEXITY_API_KEY,
                base_url="https://api.perplexity.ai"
            )
            model = "sonar"  # Use latest sonar model
        else:
            client = OpenAI(api_key=OPENAI_API_KEY)
            model = "gpt-4"

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert at explaining blockchain smart contracts in simple terms. "
                        "Provide a clear, non-technical summary suitable for business stakeholders."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Explain this PyTeal smart contract:\n\n{code}\n\n"
                        "Include: purpose, key operations, user interactions, and risks."
                    )
                }
            ],
            temperature=0.3,
            max_tokens=800
        )
        return response.choices[0].message.content

    except Exception as e:
        logging.error(f"Explanation generation failed: {e}")
        return f"Error generating explanation: {str(e)}"
