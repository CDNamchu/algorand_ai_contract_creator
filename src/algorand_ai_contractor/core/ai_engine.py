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

*OUTPUT STRUCTURE:*
1. Complete PyTeal source code
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

        Args:
            description: Natural language contract description
            max_retries: Maximum retry attempts
            ai_provider: 'perplexity' or 'openai' (overrides default)
            model: Specific model to use (overrides default)

        Returns:
            Dict with keys: code, explanation, deployment, audit
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
                parsed = self._parse_ai_response(raw_output)
                # Sanitize the parsed code to remove common generator artifacts
                parsed['code'] = self._sanitize_code(parsed['code'])
                validation_result = self._validate_pyteal_syntax(parsed['code'])

                if validation_result['valid']:
                    self._log_generation(
                        description, parsed, attempt + 1, provider, selected_model
                    )
                    return {
                        'success': True,
                        'code': parsed['code'],
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

                last_error = validation_result['error']
                attempt += 1
                logging.warning(f"Validation failed: {last_error}")

            except Exception as e:
                last_error = str(e)
                attempt += 1
                logging.error(f"Generation error: {e}")

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
                """Parse the AI response and extract the PyTeal code block.

                This parser is defensive:
                - If the model returns a fenced code block (```...```), extract the first block
                    and drop any leading language tag (e.g. ```python).
                - Otherwise, attempt to split the response from trailing human-readable
                    sections (markdown separators or headings).
                - Returns a dict with keys: code, explanation, deployment, audit. The latter
                    fields may be empty if the model did not provide structured sections.
                """
                code = raw_output
                explanation = ""
                deployment = ""
                audit = ""

                # If the AI returned a fenced code block, extract the inner content.
                if "```" in raw_output:
                        start = raw_output.find("```")
                        # find closing fence after the opening
                        end = raw_output.find("```", start + 3)
                        if end != -1:
                                inner = raw_output[start + 3:end]
                                # remove optional language token like 'python' at the start
                                inner = inner.lstrip()
                                if inner.startswith("python"):
                                        inner = inner[len("python"):].lstrip()
                                code = inner.strip()

                                # Anything after the first code fence is treated as explanation/audit text
                                tail = raw_output[end + 3 :].strip()
                                if tail:
                                        explanation = tail
                else:
                        # Try to split on a common markdown separator used by the generator
                        for sep in ["\n\n---", "\n---", "\n**Contract Purpose Summary:", "\n**Contract Purpose Summary**"]:
                                if sep in raw_output:
                                        parts = raw_output.split(sep, 1)
                                        code = parts[0].strip()
                                        explanation = parts[1].strip()
                                        break

                return {"code": code, "explanation": explanation, "deployment": deployment, "audit": audit}

    def _validate_pyteal_syntax(self, code: str) -> Dict[str, str]:
        """Basic safety and content checks on the generated code.

        This is intentionally lightweight: it rejects obviously dangerous patterns
        and ensures the response contains expected PyTeal markers. A downstream
        compilation step should be used to validate full correctness.
        """
        # Reject obviously dangerous Python patterns
        dangerous = ["eval(", "exec(", "open(", "subprocess", "os.system", "pickle.loads"]
        for pat in dangerous:
            if pat in code:
                return {"valid": False, "error": f"Dangerous pattern detected: {pat}"}

        # Heuristic: ensure the response contains PyTeal-related content
        lower = code.lower()
        if "from pyteal" in lower or "import pyteal" in lower or "txn" in lower or "app.globalput" in lower:
            return {"valid": True}

        return {"valid": False, "error": "Missing expected PyTeal content or approval program."}

    def _sanitize_code(self, code: str) -> str:
        """Sanitize AI-generated code before saving/validating.

        Operations performed:
        - Extract first fenced code block if present (```...```).
        - Remove trailing markdown sections after common separators.
        - Replace Addr(some_variable_expression) with the inner expression when the
          argument is not a string literal (e.g., Addr(Txn.application_args[0]) -> Txn.application_args[0]).
        - Keep Addr("literal") intact.
        - Trim leading/trailing whitespace and ensure a final newline.
        """
        import re

        if not code:
            return code

        sanitized = code

        # 1) If there's a fenced code block, extract the first one
        if '```' in sanitized:
            start = sanitized.find('```')
            end = sanitized.find('```', start + 3)
            if end != -1:
                inner = sanitized[start + 3:end]
                # Remove optional language token (e.g., python) from the start
                inner = re.sub(r'^\s*python\s*\n', '', inner, flags=re.IGNORECASE)
                sanitized = inner

        # 2) Remove trailing markdown after common separators
        for sep in ['\n\n---', '\n---', '\n**Contract Purpose Summary:', '\n**Logic Walkthrough:', '\n**Security Considerations:']:
            if sep in sanitized:
                sanitized = sanitized.split(sep, 1)[0]

        # 3) Replace Addr(<non-literal>) with the inner expression
        # Keep Addr("literal") intact
        def _addr_repl(m: re.Match) -> str:
            inner = m.group(1).strip()
            # If inner is quoted (literal), keep the Addr(...) as-is
            if re.match(r'^["\']', inner):
                return f'Addr({inner})'
            # otherwise, return the inner expression without Addr()
            return inner

        sanitized = re.sub(r'Addr\(([^)]+)\)', _addr_repl, sanitized)

        # 4) Ensure final newline
        sanitized = sanitized.strip() + '\n'

        return sanitized

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
