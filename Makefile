.PHONY: run test docker docker-run-quick

run:
	uvicorn src.api.main:app --reload

test:
	pytest -q

docker:
	docker build -t fraud-api .

docker-run-quick:
	docker run --rm -p 8000:8000 fraud-api