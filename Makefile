bench:
	python bench/run_bench.py --url http://localhost:8001 --concurrency 1 4 16 --tag hf
	python bench/run_bench.py --url http://localhost:8000 --concurrency 1 16 64 128 300 --stream --tag vllm
	python bench/report.py results/
test:
	pytest -q tests/
