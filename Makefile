UV ?= uv

.PHONY: check contracts-check telemetry-check lease-check hardware-check package-check profile-measure-check capacity-evidence-check launcher-consumer-readiness model-sizer-blocked prefetch

check: contracts-check telemetry-check lease-check hardware-check package-check profile-measure-check capacity-evidence-check launcher-consumer-readiness model-sizer-blocked
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --locked --offline python tools/check_gate_wiring.py

contracts-check:
	UV="$(UV)" /bin/sh tools/validate_candidate
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --locked --offline python tools/validate_frozen_contracts.py
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --locked --offline python -m unittest discover -s tools/tests -v

telemetry-check:
	cd components/kilix-telemetry && PYTHONDONTWRITEBYTECODE=1 $(UV) run --locked --offline python -m unittest discover -s tests -v

lease-check:
	cd components/kilix-device-lease && PYTHONDONTWRITEBYTECODE=1 $(UV) run --locked --offline python -m unittest discover -s tests -v

hardware-check:
	cd components/plebian-hardware && PYTHONDONTWRITEBYTECODE=1 $(UV) run --locked --offline python -m unittest discover -s tests -v
	UV="$(UV)" /bin/sh tools/validate_candidate --live-hardware

package-check:
	UV=$(UV) PYTHONDONTWRITEBYTECODE=1 $(UV) run --locked --offline python tools/check_distributions.py

profile-measure-check:
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --locked --offline python -m unittest discover -s tools/measure/tests -v

capacity-evidence-check:
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --locked --offline python tools/validate_h2_capacity_evidence.py

launcher-consumer-readiness:
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --locked --offline python tools/check_trusted_launcher_consumer_readiness.py --self-test

model-sizer-blocked:
	PYTHONDONTWRITEBYTECODE=1 $(UV) run --locked --offline python tools/check_model_sizer_block.py

# Online, once: fill the uv cache that make check reads offline. uv verifies every
# download against uv.lock, and the root's build backend against tools/build-constraints.txt.
prefetch:
	$(UV) sync --locked --no-install-project --managed-python --no-python-downloads --python 3.12.8
	cd components/kilix-telemetry && $(UV) sync --locked
	cd components/kilix-device-lease && $(UV) sync --locked
	cd components/plebian-hardware && $(UV) sync --locked
	UV=$(UV) PYTHONDONTWRITEBYTECODE=1 $(UV) run --locked python tools/check_distributions.py --prefetch
