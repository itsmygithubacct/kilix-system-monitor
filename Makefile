UV ?= uv

.PHONY: check contracts-check telemetry-check hardware-check package-check launcher-consumer-readiness model-sizer-blocked

check: contracts-check telemetry-check hardware-check package-check launcher-consumer-readiness model-sizer-blocked

contracts-check:
	UV="$(UV)" /bin/sh tools/validate_candidate

telemetry-check:
	cd components/kilix-telemetry && PYTHONDONTWRITEBYTECODE=1 $(UV) run --locked --offline python -m unittest discover -s tests -v

hardware-check:
	cd components/plebian-hardware && PYTHONDONTWRITEBYTECODE=1 $(UV) run --locked --offline python -m unittest discover -s tests -v
	UV="$(UV)" /bin/sh tools/validate_candidate --live-hardware

package-check:
	UV=$(UV) PYTHONDONTWRITEBYTECODE=1 $(UV) run --locked --offline python tools/check_distributions.py

launcher-consumer-readiness:
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --locked --offline python tools/check_trusted_launcher_consumer_readiness.py --self-test

model-sizer-blocked:
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --locked --offline python tools/check_model_sizer_block.py
