install:
	@pip install -e .

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -f */.ipynb_checkpoints
	@rm -Rf build
	@rm -Rf */__pycache__
	@rm -Rf */*.pyc

all: install clean

run_api:
	uvicorn p7_global.api.app:app --reload

test_api_root:
	@echo "Test API endpoints"
	pytest \
	tests/integration/test_api.py::test_root_is_up --asyncio-mode=strict -W "ignore" \
	tests/integration/test_api.py::test_root_returns_greeting --asyncio-mode=strict -W "ignore"

test_api_predict:
	@echo "Test API predictions results"
	pytest \
	tests/integration/test_api.py::test_predict_is_up --asyncio-mode=strict -W "ignore" \
	tests/integration/test_api.py::test_predict_is_positive --asyncio-mode=strict -W "ignore" \

test_embedding:
	@echo "Test dictionnaire embedding"
	pytest \
	tests/ml/test_ml.py::test_vocab_exist \
	tests/ml/test_ml.py::test_vocab_size \
	tests/ml/test_ml.py::test_vector_size


test_all:	test_api_root test_api_predict test_embedding
