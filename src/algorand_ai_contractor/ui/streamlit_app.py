"""
AI-Powered Algorand Smart Contract Creator
Web Interface with Streamlit
"""

import streamlit as st
import json
from datetime import datetime
from pathlib import Path
import os
from algorand_ai_contractor.core.ai_engine import ContractGenerator, explain_contract
from algorand_ai_contractor.core.algorand_utils import AlgorandDeployer, create_simple_clear_program

# Define outputs path
GENERATED_CONTRACTS_PATH = Path(__file__).parent.parent.parent.parent / "outputs" / "contracts"

# Page configuration
st.set_page_config(
    page_title="AI Smart Contract Creator",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to save contracts (defined BEFORE use)
def save_contract_to_file(contract_code: str, description: str):
    """Save generated contract to smart_contracts/ai_generated/ folder."""
    try:
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
    except Exception as e:
        st.warning(f"Could not save to file: {e}")
        return None

# Initialize session state
if 'generation_history' not in st.session_state:
    st.session_state.generation_history = []
if 'current_contract' not in st.session_state:
    st.session_state.current_contract = None

# Initialize components
@st.cache_resource
def init_generator():
    return ContractGenerator(model="sonar", temperature=0.2)

@st.cache_resource
def init_deployer():
    try:
        return AlgorandDeployer()
    except Exception as e:
        st.error(f"Algorand connection failed: {e}")
        return None

generator = init_generator()
deployer = init_deployer()

# Header
st.title("🔗 AI-Powered Smart Contract Creator")
st.markdown("""
*Algorand PyTeal Contract Generator* | Powered by GPT-4  
Generate, validate, and deploy smart contracts using natural language.

EU AI Act Tier 2 Compliant | IEEE EAD Aligned
""")

# Sidebar
with st.sidebar:
    st.header("⚙ Configuration")
    
    st.subheader("AI Settings")
    ai_provider = st.selectbox("AI Provider", ["perplexity", "openai"], index=0)
    
    if ai_provider == "perplexity":
        model_choice = st.selectbox("Model", [
            "sonar",      # Main model
            "sonar-pro"   # Pro model (if available)
        ], index=0)
    else:
        model_choice = st.selectbox("Model", ["gpt-4", "gpt-4-turbo"], index=0)
    
    temperature = st.slider("Temperature", 0.0, 0.5, 0.2, 0.05)
    
    st.subheader("Deployment")
    if deployer:
        try:
            status = deployer.algod_client.status()
            st.success(f"✅ TestNet Connected (Round {status['last-round']})")
        except:
            st.error("❌ TestNet Offline")
    
    st.divider()
    
    st.subheader("📊 Session Stats")
    st.metric("Contracts Generated", len(st.session_state.generation_history))
    
    if st.button("🗑 Clear History"):
        st.session_state.generation_history = []
        st.session_state.current_contract = None
        st.rerun()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎨 Generate", "🔍 Explain", "🚀 Deploy", "📜 History"])

# TAB 1: Generate Contract
with tab1:
    st.header("Generate Smart Contract")
    
    # Example templates
    with st.expander("📝 Example Prompts"):
        st.markdown("""
        **Simple Escrow:**
        - Create an escrow contract that holds funds until both buyer and seller confirm the transaction.
        
        **Token Voting:**
        - Build a voting contract where users lock tokens to vote on proposals with a 7-day deadline.
        
        **Time-Locked Vault:**
        - Design a vault that releases funds to a beneficiary only after a specified timestamp.
        
        **Multi-Signature Wallet:**
        - Create a 2-of-3 multi-signature wallet for secure fund management.
        """)
    
    # Input form
    user_description = st.text_area(
        "Describe your smart contract:",
        height=150,
        placeholder="Example: Create a simple auction contract where users can bid, and the highest bidder wins after the auction closes..."
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        generate_button = st.button("⚡ Generate Contract", type="primary", use_container_width=True)
    
    if generate_button and user_description:
        with st.spinner("🤖 AI is crafting your contract..."):
            result = generator.generate_pyteal_contract(
                user_description,
                ai_provider=ai_provider,
                model=model_choice
            )
            
            if result['success']:
                st.session_state.current_contract = result
                st.session_state.generation_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'description': user_description,
                    'result': result
                })
                
                # Save to smart_contracts/ai_generated/
                saved_path = save_contract_to_file(result['code'], user_description)
                
                st.success(f"✅ Contract generated in {result['metadata']['attempts']} attempt(s)")
                if saved_path:
                    st.info(f"📁 Saved to: `{os.path.relpath(str(saved_path), str(Path.cwd()))}`")
            else:
                st.error(f"❌ Generation failed: {result['error']}")
                st.stop()
    
    # Display current contract
    if st.session_state.current_contract and st.session_state.current_contract['success']:
        contract = st.session_state.current_contract
        
        st.divider()
        st.subheader("Generated Contract")
        
        # Code display
        st.code(contract['code'], language='python')
        
        col1, col2 = st.columns(2)
        with col1:
            # Download button
            st.download_button(
                label="💾 Download PyTeal Code",
                data=contract['code'],
                file_name=f"contract_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
                mime="text/x-python"
            )
        
        with col2:
            # Show saved location
            st.info(f"📁 Auto-saved to: `smart_contracts/ai_generated/`")
        
        # Explanation
        with st.expander("📖 Contract Explanation", expanded=True):
            st.markdown(contract['explanation'] or "No explanation provided")
        
        # Deployment instructions
        with st.expander("🛠 Deployment Instructions"):
            st.markdown(contract['deployment'] or "No deployment instructions provided")
        
        # Audit summary
        with st.expander("🔒 Security Audit Summary"):
            st.markdown(contract['audit'] or "No audit information provided")

# TAB 2: Explain Contract
with tab2:
    st.header("Explain Existing Contract")
    
    existing_code = st.text_area(
        "Paste PyTeal code to explain:",
        height=300,
        placeholder="from pyteal import *\n\ndef approval_program():\n    return Approve()"
    )
    
    if st.button("🔍 Explain Code", type="primary"):
        if existing_code:
            with st.spinner("Analyzing contract..."):
                explanation = explain_contract(existing_code)
                st.markdown("### Analysis")
                st.markdown(explanation)
        else:
            st.warning("Please provide PyTeal code to analyze")

# TAB 3: Deploy Contract
with tab3:
    st.header("Deploy to Algorand TestNet")
    
    if not deployer:
        st.error("⚠ Algorand deployer not initialized. Check your .env configuration.")
        st.stop()
    
    if not st.session_state.current_contract:
        st.info("👈 Generate a contract first in the Generate tab")
        st.stop()
    
    st.subheader("Step 1: Compile Contract")
    
    if st.button("⚙ Compile to TEAL"):
        with st.spinner("Compiling..."):
            contract_code = st.session_state.current_contract['code']
            compile_result = deployer.compile_pyteal_to_teal(contract_code)
            
            if compile_result['success']:
                st.success("✅ Compilation successful!")
                st.session_state['compiled_teal'] = compile_result['teal']
                st.session_state['compiled_hash'] = compile_result['hash']
                
                with st.expander("View TEAL Code"):
                    st.code(compile_result['teal'], language='teal')
                    st.caption(f"Hash: {compile_result['hash']}")
            else:
                st.error(f"❌ Compilation failed: {compile_result['error']}")
    
    if 'compiled_teal' in st.session_state:
        st.divider()
        st.subheader("Step 2: Deploy to TestNet")
        
        st.warning("⚠ Requires a funded TestNet account")
        
        with st.expander("🆕 Generate Test Account"):
            if st.button("Create New Account"):
                test_account = deployer.generate_test_account()
                st.json(test_account)
                st.info(f"Fund this account at: {test_account['faucet_url']}")
        
        private_key = st.text_input(
            "Deployment Account Private Key:",
            type="password",
            help="Your TestNet account private key (kept secure - never stored)"
        )
        
        if st.button("🚀 Deploy Contract", type="primary"):
            if not private_key:
                st.error("Please provide a private key")
            else:
                with st.spinner("Deploying to Algorand TestNet..."):
                    clear_program = create_simple_clear_program()
                    
                    deploy_result = deployer.deploy_contract(
                        approval_teal=st.session_state['compiled_teal'],
                        clear_teal=clear_program,
                        sender_private_key=private_key
                    )
                    
                    if deploy_result['success']:
                        st.success("🎉 Contract deployed successfully!")
                        st.json(deploy_result)
                        st.markdown(f"[View on AlgoExplorer]({deploy_result['explorer_url']})")
                    else:
                        st.error(f"Deployment failed: {deploy_result['error']}")

# TAB 4: History
with tab4:
    st.header("Generation History")
    
    if not st.session_state.generation_history:
        st.info("No contracts generated yet")
    else:
        for idx, entry in enumerate(reversed(st.session_state.generation_history)):
            with st.expander(f"Contract #{len(st.session_state.generation_history) - idx} - {entry['timestamp'][:19]}"):
                st.markdown(f"*Description:* {entry['description']}")
                st.code(entry['result']['code'], language='python')

# Footer
st.divider()
st.caption("Built with ❤ | Algorand + Perplexity AI | IEEE EAD & EU AI Act Compliant")