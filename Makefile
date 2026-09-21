.PHONY: install test eval api ui docker-up lint

install:
	pip install -r requirements.txt && pip install -e .

test:
	PYTHONPATH=src FINRAG_PROVIDER=offline pytest -q

eval:
	PYTHONPATH=src FINRAG_PROVIDER=offline python -m finrag.eval.run_eval

api:
	PYTHONPATH=src uvicorn finrag.api.main:app --reload --port 8000

ui:
	PYTHONPATH=src streamlit run app/streamlit_app.py

lint:
	ruff check src app tests

docker-up:
	docker compose up --build
