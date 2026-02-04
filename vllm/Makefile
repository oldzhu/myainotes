.PHONY: help vllm-last50 vllm-last200 vllm-since-tag clean-reports

# Default repo to analyze.
VLLM_REPO ?= $(HOME)/vllm

# Output directory (relative to this Makefile).
OUT_DIR ?= reports

# Optional git log range.
REV ?= HEAD

# Optional tag to compare from: make vllm-since-tag TAG=v0.6.0
TAG ?=

help:
	@echo "Targets:"
	@echo "  make vllm-last50      # snapshot last 50 commits report into $(OUT_DIR)/"
	@echo "  make vllm-last200     # snapshot last 200 commits report into $(OUT_DIR)/"
	@echo "  make vllm-since-tag   # snapshot changes since TAG (requires TAG=...)"
	@echo "  make clean-reports    # delete generated reports in $(OUT_DIR)/"
	@echo ""
	@echo "Variables: VLLM_REPO=$(VLLM_REPO) OUT_DIR=$(OUT_DIR) REV=$(REV)"

vllm-last50:
	python3 scripts/snapshot_git_activity.py --repo "$(VLLM_REPO)" -n 50 --rev "$(REV)" --out-dir "$(OUT_DIR)" --latest

vllm-last200:
	python3 scripts/snapshot_git_activity.py --repo "$(VLLM_REPO)" -n 200 --rev "$(REV)" --out-dir "$(OUT_DIR)" --latest

vllm-since-tag:
	@if [ -z "$(TAG)" ]; then echo "ERROR: TAG is required (e.g. make vllm-since-tag TAG=v0.6.0)"; exit 2; fi
	python3 scripts/snapshot_git_activity.py --repo "$(VLLM_REPO)" -n 200 --rev "$(TAG)..HEAD" --out-dir "$(OUT_DIR)" --latest

clean-reports:
	@mkdir -p "$(OUT_DIR)"
	@rm -f "$(OUT_DIR)"/*.md
