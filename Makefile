export PYTHONPATH := src

.PHONY: install data pipeline dashboard test lint docker

install:    ## install dependencies
	pip install -r requirements-dev.txt
data:       ## generate 20,000 loyalty members, 2 years of baskets and an A/B coupon test (~3 min)
	python -m cust360.data
pipeline:   ## segments, churn, CLV, recommendations, uplift -> outputs/report.md + customer_scores.csv
	python -m cust360.pipeline
dashboard:  ## http://localhost:8501
	streamlit run dashboard.py
test:
	pytest -q
lint:
	ruff check .
docker:
	docker build -t uae-customer-360 . && docker run -p 8501:8501 uae-customer-360
