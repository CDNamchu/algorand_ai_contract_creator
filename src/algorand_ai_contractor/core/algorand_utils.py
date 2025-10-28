"""
Algorand TestNet Deployment & Validation Utilities
Supports compilation, simulation, and deployment
"""

import os
import base64
import logging
from typing import Dict, Optional
from algosdk import account, mnemonic
from algosdk.v2client import algod
from algosdk.transaction import ApplicationCreateTxn, OnComplete, StateSchema, wait_for_confirmation
from pyteal import compileTeal, Mode, Approve
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    filename='deployment.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)


class AlgorandDeployer:
    """Secure deployment manager for Algorand TestNet."""
    
    def __init__(self):
        self.algod_token = os.getenv('ALGOD_TOKEN', 'a' * 64)
        self.algod_address = os.getenv('ALGOD_ADDRESS', 'https://testnet-api.algonode.cloud')
        self.algod_client = algod.AlgodClient(
            self.algod_token,
            self.algod_address
        )
        self._verify_connection()
    
    def _verify_connection(self):
        """Test Algorand node connectivity."""
        try:
            status = self.algod_client.status()
            logging.info(f"Connected to Algorand TestNet - Round: {status['last-round']}")
        except Exception as e:
            logging.error(f"Algorand connection failed: {e}")
            raise ConnectionError("Cannot connect to Algorand node")
    
    def compile_pyteal_to_teal(self, pyteal_code: str, mode: Mode = Mode.Application) -> Dict:
        """
        Compile PyTeal source to TEAL bytecode with validation.
        
        Returns:
            Dict with 'success', 'teal', 'compiled', 'error'
        """
        try:
            # Create temporary namespace for exec
            namespace = {}
            
            # Execute PyTeal code to get program
            exec(pyteal_code, namespace)
            
            # Look for approval_program or router or app in the executed namespace
            approval_program = None
            for name in ['approval_program', 'router', 'app']:
                if name in namespace:
                    approval_program = namespace[name]
                    break

            # If we found a callable (function) try to call it if it accepts no arguments
            import inspect
            from pyteal import Expr

            def _resolve_candidate(obj):
                """If obj is a zero-arg callable, call it and return the result; otherwise return obj."""
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
                    # If calling raised, ignore and return original object
                    return obj

            if approval_program is not None:
                approval_program = _resolve_candidate(approval_program)

            # If not found yet, try to discover a PyTeal Expr in the namespace
            if approval_program is None:
                # First try to find objects that look like PyTeal Exprs directly
                for obj in namespace.values():
                    try:
                        if isinstance(obj, Expr):
                            approval_program = obj
                            break
                    except Exception:
                        pass

            # If still not found, try calling zero-arg callables and test compileTeal on result
            if approval_program is None:
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
                                # quick compile test to verify it's a PyTeal program
                                try:
                                    compileTeal(candidate, mode, version=6)
                                    approval_program = candidate
                                    break
                                except Exception:
                                    continue
                        except Exception:
                            continue

            if approval_program is None:
                return {
                    'success': False,
                    'error': 'No PyTeal program found. Ensure you define approval_program variable.'
                }

            # Compile to TEAL (use TEAL v6 for broader compatibility)
            teal_code = compileTeal(approval_program, mode, version=6)
            
            # Compile TEAL to bytecode
            compile_response = self.algod_client.compile(teal_code)
            
            return {
                'success': True,
                'teal': teal_code,
                'compiled': compile_response['result'],
                'hash': compile_response['hash'],
                'error': None
            }
            
        except SyntaxError as e:
            return {'success': False, 'error': f'Syntax Error: {e}'}
        except Exception as e:
            return {'success': False, 'error': f'Compilation failed: {str(e)}'}
    
    def deploy_contract(
        self,
        approval_teal: str,
        clear_teal: str,
        sender_private_key: str,
        global_schema: StateSchema = StateSchema(num_uints=1, num_byte_slices=1),
        local_schema: StateSchema = StateSchema(num_uints=0, num_byte_slices=0)
    ) -> Dict:
        """
        Deploy smart contract to Algorand TestNet.
        
        Args:
            approval_teal: TEAL approval program
            clear_teal: TEAL clear program
            sender_private_key: Private key for deployment account
            global_schema: Global state schema
            local_schema: Local state schema
        
        Returns:
            Dict with app_id, txn_id, address
        """
        try:
            # Derive address from private key
            sender_address = account.address_from_private_key(sender_private_key)
            
            # Compile programs
            approval_compiled = self.algod_client.compile(approval_teal)
            clear_compiled = self.algod_client.compile(clear_teal)
            
            approval_program = base64.b64decode(approval_compiled['result'])
            clear_program = base64.b64decode(clear_compiled['result'])
            
            # Get suggested params
            params = self.algod_client.suggested_params()
            
            # Create application transaction
            txn = ApplicationCreateTxn(
                sender=sender_address,
                sp=params,
                on_complete=OnComplete.NoOpOC,
                approval_program=approval_program,
                clear_program=clear_program,
                global_schema=global_schema,
                local_schema=local_schema
            )
            
            # Sign transaction
            signed_txn = txn.sign(sender_private_key)
            
            # Send transaction
            txid = self.algod_client.send_transaction(signed_txn)
            
            # Wait for confirmation
            confirmed_txn = wait_for_confirmation(self.algod_client, txid, 4)
            
            app_id = confirmed_txn['application-index']
            app_address = self._get_app_address(app_id)
            
            logging.info(f"Contract deployed - App ID: {app_id}, Txn: {txid}")
            
            return {
                'success': True,
                'app_id': app_id,
                'txn_id': txid,
                'address': app_address,
                'explorer_url': f'https://testnet.algoexplorer.io/application/{app_id}'
            }
            
        except Exception as e:
            logging.error(f"Deployment failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_app_address(self, app_id: int) -> str:
        """Calculate application address from app ID."""
        from algosdk.logic import get_application_address
        return get_application_address(app_id)
    
    def generate_test_account(self) -> Dict[str, str]:
        """Generate a new Algorand test account."""
        private_key, address = account.generate_account()
        mnem = mnemonic.from_private_key(private_key)
        
        return {
            'address': address,
            'private_key': private_key,
            'mnemonic': mnem,
            'faucet_url': f'https://testnet.algoexplorer.io/dispenser?account={address}'
        }
    
    def get_account_balance(self, address: str) -> Optional[int]:
        """Get account balance in microAlgos."""
        try:
            account_info = self.algod_client.account_info(address)
            return account_info.get('amount', 0)
        except Exception as e:
            logging.error(f"Balance check failed: {e}")
            return None


def create_simple_clear_program() -> str:
    """Generate minimal clear state program."""
    from pyteal import Approve, compileTeal, Mode
    return compileTeal(Approve(), Mode.Application, version=8)