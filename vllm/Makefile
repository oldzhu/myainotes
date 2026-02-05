.PHONY: help vllm-last50 vllm-last200 vllm-since-tag vllm-kernels-last10 clean-reports

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
	@echo "  make vllm-kernels-last10 # snapshot last 10 commits touching custom ops/native kernels"
	@echo "  make clean-reports    # delete generated reports in $(OUT_DIR)/"
	@echo ""
	@echo "Variables: VLLM_REPO=$(VLLM_REPO) OUT_DIR=$(OUT_DIR) REV=$(REV)"
	@echo "           KERNEL_WATCH_PATHS=$(KERNEL_WATCH_PATHS)"

# Paths that roughly correspond to “custom ops + native kernels” (GPU+CPU).
# Override to tune scope.
KERNEL_WATCH_PATHS ?= vllm/_custom_ops.py vllm/_aiter_ops.py csrc vllm/v1/attention

vllm-last50:
	python3 scripts/snapshot_git_activity.py --repo "$(VLLM_REPO)" -n 50 --rev "$(REV)" --out-dir "$(OUT_DIR)" --latest

vllm-last200:
	python3 scripts/snapshot_git_activity.py --repo "$(VLLM_REPO)" -n 200 --rev "$(REV)" --out-dir "$(OUT_DIR)" --latest

vllm-since-tag:
	@if [ -z "$(TAG)" ]; then echo "ERROR: TAG is required (e.g. make vllm-since-tag TAG=v0.6.0)"; exit 2; fi
	python3 scripts/snapshot_git_activity.py --repo "$(VLLM_REPO)" -n 200 --rev "$(TAG)..HEAD" --out-dir "$(OUT_DIR)" --latest

vllm-kernels-last10:
	python3 scripts/snapshot_git_activity.py --repo "$(VLLM_REPO)" -n 10 --rev "$(REV)" --out-dir "$(OUT_DIR)" --latest --show-commits 10 $(foreach p,$(KERNEL_WATCH_PATHS),--path "$(p)")

clean-reports:
	@mkdir -p "$(OUT_DIR)"
	@rm -f "$(OUT_DIR)"/*.md
