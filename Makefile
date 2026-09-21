ifeq ($(OS),Windows_NT)
    PYTHON ?= .venv/Scripts/python.exe
else
    PYTHON ?= .venv/bin/python
endif

.PHONY: dataset bench figures article check test smoke all clean

dataset:
	$(PYTHON) -m data.generate_dataset

bench:
	$(PYTHON) -m src.runner

figures:
	$(PYTHON) -m analysis.plot_results

article:
	$(PYTHON) -m analysis.render_article_tables

check:
	$(PYTHON) -m analysis.render_article_tables --check

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) -m src.runner --limit 20

all: bench figures article

clean:
	rm -f results/records.jsonl results/summary.json
	rm -rf figures/

