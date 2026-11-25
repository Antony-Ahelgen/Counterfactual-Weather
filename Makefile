PY ?= .venv/Scripts/python
PRESET ?= cpu

.PHONY: test smoke data train eval paper-assets audit

test:
	$(PY) tasks.py test
smoke:
	$(PY) tasks.py smoke
data:
	$(PY) tasks.py data --preset $(PRESET)
train:
	$(PY) tasks.py train --preset $(PRESET)
eval:
	$(PY) tasks.py eval --preset $(PRESET)
paper-assets:
	$(PY) tasks.py paper-assets --preset $(PRESET)
audit:
	$(PY) tasks.py audit
