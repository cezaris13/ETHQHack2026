PYTHON ?= python3
VENV ?= .venv
VENV_PY := $(VENV)/bin/python
VENV_PIP := $(VENV_PY) -m pip

.PHONY: install env-file clean

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)

env-file:
	@if [ ! -s mermin/.env ]; then \
		cp mermin/.env_template mermin/.env; \
		echo "Created mermin/.env from mermin/.env_template — fill in the values."; \
	else \
		echo "mermin/.env already exists, leaving it untouched."; \
	fi

install: $(VENV)/bin/activate env-file
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	@echo
	@echo "Activate the environment with: source $(VENV)/bin/activate"

clean:
	rm -rf $(VENV)
