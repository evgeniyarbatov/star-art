VENV_PATH := .venv

PYTHON := $(VENV_PATH)/bin/python
PIP := $(VENV_PATH)/bin/pip
REQUIREMENTS := requirements.txt

default: art

venv:
	@python3 -m venv $(VENV_PATH)

install: venv
	@$(PIP) install --disable-pip-version-check -q --upgrade pip
	@$(PIP) install --disable-pip-version-check -q -r $(REQUIREMENTS)

art:
	@$(PYTHON) scripts/star-art.py

stars:
	@$(PYTHON) scripts/star-art-names.py

galaxies:
	@$(PYTHON) scripts/star-art-galaxies.py

planets:
	@$(PYTHON) scripts/star-art-planets.py

nebulae:
	@$(PYTHON) scripts/star-art-nebulae.py

clusters:
	@$(PYTHON) scripts/star-art-star-clusters.py

exotic:
	@$(PYTHON) scripts/star-art-exotic-objects.py

clean:
	find images -mindepth 1 ! -name '.gitignore' -exec rm -rf {} +

cleanvenv:
	@rm -rf $(VENV_PATH)
