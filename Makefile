SHELL := /bin/bash

.PHONY: env
env:  ## Setup python virtual environment
	python3 -m venv venv

.PHONY: activate
activate:  ## Activate python virtual environment
	@echo "run: source venv/bin/activate"

.PHONY: deactivate
deactivate: ## Deactivate python virtual environment
	@echo "run: deactivate"

.PHONY: install_deps
install_deps:  ## Install dependencies
	pip install -r requirements.txt

.PHONY: run
run: ## Run the application
	python3 main.py