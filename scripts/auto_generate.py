#!/usr/bin/env python3
import os
from pathlib import Path
from datetime import datetime

# Ensure src package is importable when run directly
# Run this script with: PYTHONPATH=src python scripts/auto_generate.py

from algorand_ai_contractor.core.ai_engine import ContractGenerator

GENERATED_CONTRACTS_PATH = Path(__file__).parent.parent / "outputs" / "contracts"


def save_contract_to_file(contract_code: str, description: str):
    """Save generated contract to outputs/contracts/ folder."""
    GENERATED_CONTRACTS_PATH.mkdir(parents=True, exist_ok=True)

    filename = f"contract_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    filepath = GENERATED_CONTRACTS_PATH / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f'"""\n')
        f.write(f'AI-Generated Smart Contract\n')
        f.write(f'Generated: {datetime.now().isoformat()}\n')
        f.write(f'Description: {description}\n')
        f.write(f'"""\n\n')
        f.write(contract_code)

    return filepath


def main():
    gen = ContractGenerator(model="sonar", temperature=0.2)
    desc = "Create a contract that always approves"
    result = gen.generate_pyteal_contract(desc)
    print("Generation success:", result.get('success'))
    if result.get('success'):
        saved = save_contract_to_file(result['code'], desc)
        display = os.path.relpath(str(saved), str(Path.cwd()))
        print("Saved to:", display)
    else:
        print("Generation error:", result.get('error'))


if __name__ == '__main__':
    main()
