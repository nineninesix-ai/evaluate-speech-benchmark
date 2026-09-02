.PHONY: help setup login validate discover normalize evaluate report publish publish-dry clean clean-cache

# Colors for terminal output
BLUE   := \033[1;34m
GREEN  := \033[1;32m
YELLOW := \033[1;33m
RED    := \033[1;31m
DIM    := \033[2m
RESET  := \033[0m

# Environment is managed by uv. Point VENV at an existing environment to reuse it:
#   make setup VENV=venv_eval
VENV   ?= .venv
PYTHON := $(VENV)/bin/python
CONFIG ?= config/eval.yaml

# Torch must match the GPU architecture, so it is installed from an explicit
# index before anything else resolves against it. Blackwell (RTX 50xx, RTX PRO,
# B200) needs a CUDA >= 12.8 build; override for older cards.
TORCH_INDEX ?= https://download.pytorch.org/whl/cu128

help: ## Show this help message
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(RESET)"
	@echo "$(BLUE)║        speecheval — zero-shot TTS evaluation suite         ║$(RESET)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(RESET)"
	@echo ""
	@echo "$(GREEN)make setup$(RESET)      - Create the environment with uv and install everything"
	@echo "$(GREEN)make login$(RESET)      - Authenticate with Hugging Face"
	@echo "$(GREEN)make validate$(RESET)   - Parse and check $(CONFIG)"
	@echo "$(GREEN)make discover$(RESET)   - List synthesis subsets and verify the join"
	@echo "$(GREEN)make normalize$(RESET)  - Verify text normalisation against the benchmark"
	@echo "$(GREEN)make evaluate$(RESET)   - Run the full evaluation pipeline"
	@echo "$(GREEN)make report$(RESET)     - Rebuild reports from cached metrics (no GPU)"
	@echo "$(GREEN)make publish$(RESET)    - Upload this run's metrics to the Hub"
	@echo "$(GREEN)make publish-dry$(RESET)- Assemble the upload and list it, no Hub access"
	@echo "$(GREEN)make clean$(RESET)      - Remove the environment"
	@echo "$(GREEN)make clean-cache$(RESET)- Drop the stage cache, keeping reports"
	@echo ""
	@echo "$(YELLOW)💡 First run:$(RESET)"
	@echo "   1. $(GREEN)make setup$(RESET)     - Install dependencies"
	@echo "   2. $(GREEN)make login$(RESET)     - Authenticate with Hugging Face"
	@echo "   3. Put ELEVENLABS_KEY in $(GREEN).env$(RESET) (or disable scribe in the config)"
	@echo "   4. $(GREEN)make discover$(RESET)  - Confirm the data lines up"
	@echo "   5. $(GREEN)make evaluate$(RESET)  - Measure"
	@echo ""
	@echo "$(DIM)   Reuse an existing environment: make setup VENV=venv_eval$(RESET)"
	@echo "$(DIM)   Use another config:            make evaluate CONFIG=config/mine.yaml$(RESET)"
	@echo ""

setup: ## Create virtual environment and install dependencies
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(RESET)"
	@echo "$(BLUE)║              Setting up the evaluation env                 ║$(RESET)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(RESET)"
	@echo ""
	@command -v uv >/dev/null 2>&1 || { \
		echo "$(RED)✗ uv is not installed$(RESET)"; \
		echo "$(YELLOW)  curl -LsSf https://astral.sh/uv/install.sh | sh$(RESET)"; \
		exit 1; }
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(YELLOW)→ Creating environment at $(VENV)...$(RESET)"; \
		uv venv --python 3.12 $(VENV); \
		echo "$(GREEN)✓ Environment created$(RESET)"; \
	else \
		echo "$(GREEN)✓ Reusing existing environment at $(VENV)$(RESET)"; \
	fi
	@echo ""
	@echo "$(YELLOW)→ Installing torch for this GPU ($(TORCH_INDEX))...$(RESET)"
	@echo "$(DIM)   torchvision/torchcodec come from the same index on purpose: timm pulls$(RESET)"
	@echo "$(DIM)   torchvision in, and PyPI's torchcodec is a CUDA-13 build that needs$(RESET)"
	@echo "$(DIM)   libnvrtc.so.13 and won't load against this CUDA-12 (cu128) torch.$(RESET)"
	@uv pip install --python $(PYTHON) torch torchvision torchaudio torchcodec --index-url $(TORCH_INDEX)
	@echo "$(GREEN)✓ torch installed$(RESET)"
	@echo ""
	@echo "$(YELLOW)→ Installing speecheval and its dependencies...$(RESET)"
	@echo "$(DIM)   (the measurement engine comes from a private repo over SSH)$(RESET)"
	@uv pip install --python $(PYTHON) -e . || { \
		echo "$(RED)✗ Install failed$(RESET)"; \
		echo "$(YELLOW)  If it stopped on make-speech-benchmark, this machine needs an$(RESET)"; \
		echo "$(YELLOW)  SSH key with access to nineninesix-ai/make-speech-benchmark.$(RESET)"; \
		exit 1; }
	@echo "$(GREEN)✓ Dependencies installed$(RESET)"
	@echo ""
	@echo "$(YELLOW)→ Checking the GPU is usable...$(RESET)"
	@$(PYTHON) -c "import torch; print('  torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available()); \
		print('  device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
	@echo ""
	@echo "$(GREEN)╔════════════════════════════════════════════════════════════╗$(RESET)"
	@echo "$(GREEN)║                     Setup complete ✓                       ║$(RESET)"
	@echo "$(GREEN)╚════════════════════════════════════════════════════════════╝$(RESET)"
	@echo ""
	@echo "$(YELLOW)Next:$(RESET) $(GREEN)make login$(RESET) then $(GREEN)make discover$(RESET)"
	@echo ""

login: ## Login to Hugging Face
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(RESET)"
	@echo "$(BLUE)║              Hugging Face authentication                    ║$(RESET)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(RESET)"
	@echo ""
	@echo "$(YELLOW)→ Configuring git credential helper...$(RESET)"
	@git config --global credential.helper store
	@echo "$(YELLOW)→ Starting authentication (you will be asked for a token)...$(RESET)"
	@$(VENV)/bin/hf auth login
	@echo ""
	@echo "$(GREEN)✓ Authenticated$(RESET)"
	@echo ""

validate: ## Parse and check the configuration
	@$(PYTHON) -m speecheval.cli validate --config $(CONFIG)

discover: ## List synthesis subsets and verify the join onto the benchmark
	@$(PYTHON) -m speecheval.cli discover --config $(CONFIG)

normalize: ## Verify text normalisation against the benchmark's text_norm
	@$(PYTHON) -m speecheval.cli normalize --config $(CONFIG)

evaluate: ## Run the full evaluation pipeline
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(RESET)"
	@echo "$(BLUE)║                  Running the evaluation                    ║$(RESET)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(RESET)"
	@echo ""
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(RED)✗ Environment not found at $(VENV)$(RESET)"; \
		echo "$(YELLOW)  Run 'make setup' first.$(RESET)"; \
		exit 1; fi
	@$(PYTHON) -m speecheval.cli run --config $(CONFIG)

report: ## Rebuild reports from cached metrics, without a GPU
	@$(PYTHON) -m speecheval.cli report --config $(CONFIG)

publish: ## Upload this run's metrics to the Hub (DATE=YYYY-MM-DD to restamp)
	@echo "$(BLUE)╔════════════════════════════════════════════════════════════╗$(RESET)"
	@echo "$(BLUE)║              Publishing metrics to the Hub                 ║$(RESET)"
	@echo "$(BLUE)╚════════════════════════════════════════════════════════════╝$(RESET)"
	@echo ""
	@$(PYTHON) -m speecheval.cli publish --config $(CONFIG) $(if $(DATE),--date $(DATE),)

publish-dry: ## Assemble the upload and list it, without touching the Hub
	@$(PYTHON) -m speecheval.cli publish --config $(CONFIG) --dry-run \
		$(if $(DATE),--date $(DATE),)

clean: ## Remove the virtual environment
	@echo "$(YELLOW)→ Removing $(VENV)...$(RESET)"
	@rm -rf $(VENV)
	@find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned$(RESET)"

clean-cache: ## Drop the stage cache, keeping the reports
	@echo "$(YELLOW)→ Removing the stage cache...$(RESET)"
	@rm -rf .cache/speecheval
	@echo "$(GREEN)✓ Cache dropped — the next run recomputes every stage$(RESET)"
