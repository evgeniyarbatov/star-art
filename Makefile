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

art: clean
	@$(PYTHON) scripts/star-art.py

clean:
	find images -mindepth 1 ! -name '.gitignore' -exec rm -rf {} +
	@rm -rf $(VENV_PATH)
