# Pallet packer — convenience targets.
#
# Everything runs inside the project Docker image to keep the toolchain
# deterministic across dev machines + CI. The image is built once with
# `make image` then reused for every command.

IMAGE := pallet-packer:dev
RUN   := docker run --rm -v $(PWD):/app -w /app $(IMAGE)

.PHONY: help image build clean shell test br-smoke industry-smoke \
        cython-clean cython-build

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

image:  ## Build the project Docker image (~3 min first time)
	docker build -t $(IMAGE) .

build:  ## Build/install the package + Cython extensions (editable)
	$(RUN) pip install --no-build-isolation -e . --quiet

cython-clean:  ## Remove generated .c / .so / .html files
	find pallet_packer -name '*.c' -delete
	find pallet_packer -name '*.so' -delete
	find pallet_packer -name '*.html' -delete

cython-build: cython-clean build  ## Fresh Cython rebuild

shell:  ## Drop into a bash shell inside the container
	docker run --rm -it -v $(PWD):/app -w /app $(IMAGE) bash

test:  ## Cython hello-world smoke + v3.12 import + IND2 round-trip
	$(RUN) python -c "from pallet_packer._brkga_core._cyhello import add; \
		assert add(2,3) == 5; print('cython OK')"
	$(RUN) python -c "from pallet_packer.brkga_v3_5 import warmup_jit; \
		warmup_jit(); print('numba OK')"

br-smoke:  ## Quick BR n=2 sanity check (BR1 + BR3)
	$(RUN) python scripts/run_v310_br_regression.py --n 2 --time 20 \
		--sets thpack1 thpack3 \
		--out results/checkpoints/smoke_br.json

industry-smoke:  ## Quick industry n=3 sanity check
	$(RUN) python scripts/run_v310_industry_eval.py --time 15 \
		--out results/checkpoints/smoke_industry.json
