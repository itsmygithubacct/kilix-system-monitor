UV ?= uv

.PHONY: check contracts-check telemetry-check hardware-check package-check model-sizer-blocked

check: contracts-check telemetry-check hardware-check package-check model-sizer-blocked

contracts-check:
	$(UV) run --locked --offline python contracts/p1-candidate/tools/validate.py --self-test

telemetry-check:
	cd components/kilix-telemetry && PYTHONDONTWRITEBYTECODE=1 $(UV) run --locked --offline python -m unittest discover -s tests -v

hardware-check:
	cd components/plebian-hardware && PYTHONDONTWRITEBYTECODE=1 $(UV) run --locked --offline python -m unittest discover -s tests -v
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --locked --offline python tools/validate_live_hardware.py

package-check:
	UV=$(UV) PYTHONDONTWRITEBYTECODE=1 $(UV) run --locked --offline python tools/check_distributions.py

model-sizer-blocked:
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --locked --offline python tools/check_model_sizer_block.py
