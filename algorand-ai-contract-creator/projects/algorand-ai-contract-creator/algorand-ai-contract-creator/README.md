# 🔗 AI-Powered Algorand Smart Contract Creator

Production-grade platform for generating, validating, and deploying Algorand PyTeal smart contracts using natural language and AI (Perplexity/GPT-4).

## 🎯 Features

- **Natural Language → PyTeal**: Describe contracts in plain English
- **Multi-Layer Validation**: Syntax, security, and compilation checks
- **Auto-Correction**: Self-healing generation with retry logic
- **TestNet Deployment**: One-click deployment to Algorand TestNet
- **Audit Trail**: Complete logging of all generations
- **Explainability**: AI-powered code explanations
- **Security-First**: OWASP-aligned, EU AI Act Tier 2 compliant
- **AlgoKit Integration**: Compatible with AlgoKit tooling

## 📁 Project Structure

```
algorand-ai-contract-creator/
├── smart_contracts/              # AlgoKit contracts folder
│   ├── __init__.py
│   ├── config.py                 # Contract schemas & settings
│   ├── ai_generated/             # AI-generated contracts
│   └── artifacts/                # Compiled TEAL files
├── tools/                        # AI generation tools
│   ├── ai_engine.py              # AI contract generator
│   ├── algorand_utils.py         # Deployment utilities
│   └── web_interface.py          # Streamlit UI
├── tests/                        # Test suite
│   └── test_contracts.py
├── .env                          # API keys (create this)
├── .algokit.toml                 # AlgoKit configuration
├── pyproject.toml                # Python project metadata
└── requirements.txt
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Perplexity API key OR OpenAI API key
- Algorand TestNet account (for deployment)

### Installation

**Windows (Easy Mode):**

```bash
# Run the installation script
install.bat

# Follow the prompts
```

**Manual Installation:**

```bash
$ git clone <repository-url>
$ cd algorand-ai-contract-creator
$ python -m venv venv
$ venv\Scripts\activate  # On Windows
$ source venv/bin/activate  # On macOS/Linux

# Install package in editable mode
$ pip install -e .

# Install dependencies
$ pip install -r requirements.txt
```

### Configuration

1. **Copy environment template:**

   ```bash
   copy .env.example .env  # Windows
   cp .env.example .env    # macOS/Linux
   ```

2. **Add your API key to `.env`:**

   ```bash
   AI_PROVIDER=perplexity
   PERPLEXITY_API_KEY=pplx-your-actual-key-here
   ```

3. **Get API key from:** https://www.perplexity.ai/settings/api

### Run Application

**Windows (Easy Mode):**

```bash
run.bat
```

**Manual:**

```bash
# Activate virtual environment
venv\Scripts\activate

# Run Streamlit
streamlit run tools\web_interface.py
```

Navigate to `http://localhost:8501`

## 📖 Usage

### 1. Generate Contract

- Enter natural language description
- Click "Generate Contract"
- Review code, explanation, and audit summary
- Download or deploy to TestNet

### 2. Deploy to TestNet

- Compile generated contract to TEAL
- Fund a TestNet account via [dispenser](https://testnet.algoexplorer.io/dispenser)
- Deploy with your private key
- View on AlgoExplorer

### 3. Explain Existing Code

- Paste PyTeal code
- Get human-readable explanation

## 🧪 Testing

```bash
# Run all tests
$ pytest tests/ -v

# Or using AlgoKit
$ algokit project run test
```

## 🔒 Security

- Temperature capped at 0.2 for deterministic output
- Automatic detection of dangerous patterns (`eval()`, `exec()`)
- No hardcoded keys in generated contracts
- All deployments logged for audit trail
- Private keys never stored
- Multi-layer validation pipeline

## 🤖 AI Providers

### Perplexity (Recommended)

- **Models**: `llama-3.1-sonar-small-128k-online`, `llama-3.1-sonar-large-128k-online`
- **Pros**: More affordable, online search capability, good for latest docs
- **Get Key**: https://www.perplexity.ai/settings/api

### OpenAI

- **Models**: `gpt-4`, `gpt-4-turbo`
- **Pros**: Industry standard, highly reliable
- **Get Key**: https://platform.openai.com/api-keys

## 📊 Performance Metrics

- **Compilation Success Rate**: TBC
- **Average Generation Time**: TBC
- **Retry Rate**: TBC
- **Security Compliance**: EU AI Act Tier 2, IEEE EAD

## 🛠️ Architecture

```
User Input (Natural Language)
        ↓
AI Engine (Perplexity/GPT-4 with safety prompts)
        ↓
Validation Pipeline (Syntax + Security)
        ↓
PyTeal → TEAL Compilation
        ↓
Algorand TestNet Deployment
        ↓
AlgoExplorer Integration
```

## 📝 Example Prompts

**Escrow Contract:**

> "Create an escrow that holds 10 ALGO until both buyer and seller call approve()"

**Time-Lock Vault:**

> "Design a vault that releases funds to address X after Unix timestamp Y"

**Voting System:**

> "Build a voting contract where each address can vote once on a yes/no proposal"

**Multi-Sig Wallet:**

> "Create a 2-of-3 multi-signature wallet for secure fund management"

## 🤝 Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new features
4. Run tests (`pytest tests/ -v`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Submit Pull Request

## 📜 License

MIT License - See LICENSE file

## 🆘 Support

- **Issues**: GitHub Issues
- **Documentation**: `/docs` folder
- **Algorand SDK**: https://developer.algorand.org
- **PyTeal Docs**: https://pyteal.readthedocs.io
- **AlgoKit**: https://github.com/algorandfoundation/algokit-cli

## 🔗 Resources

- [AlgoKit Documentation](https://github.com/algorandfoundation/algokit-cli)
- [Perplexity API Docs](https://docs.perplexity.ai/)
- [OpenAI API Docs](https://platform.openai.com/docs)
- [PyTeal Examples](https://github.com/algorand/pyteal/tree/master/examples)

---

**⚠️ DISCLAIMER**: This tool generates smart contracts for educational and testing purposes. Always conduct thorough security audits before deploying to MainNet. AI-generated code should be reviewed by experienced developers.
