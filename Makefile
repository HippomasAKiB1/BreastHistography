.PHONY: install test verify eval figures package clean

install:
	pip install -e .

test:
	pytest tests/ -v

verify:
	python scripts/verify_dataset.py

eval:
	python scripts/evaluate_checkpoints.py

figures:
	python scripts/generate_figures.py

package:
	python scripts/package_manuscript.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
