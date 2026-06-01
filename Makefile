.PHONY: run install clean lint

install:
	pip install -r requirements.txt

run:
	python run.py

clean:
	rm -rf catboost_info/ __pycache__/ src/__pycache__/ config/__pycache__/
	rm -f outputs/submission.csv

lint:
	python -m py_compile run.py
	python -m py_compile src/data_loader.py
	python -m py_compile src/features.py
	python -m py_compile src/models.py
	python -m py_compile src/ensemble.py
	python -m py_compile src/utils.py
	python -m py_compile config/settings.py
