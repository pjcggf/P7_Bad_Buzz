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
	pytest \
	tests/api_test.py::test_root_is_up --asyncio-mode=strict -W "ignore" \
	tests/api_test.py::test_root_returns_greeting --asyncio-mode=strict -W "ignore"

test_api_predict:
	pytest \
	tests/api_test.py::test_predict_is_up --asyncio-mode=strict -W "ignore" \
	tests/api_test.py::test_predict_is_positive --asyncio-mode=strict -W "ignore" \

test_embedding:
	pytest \
	tests/ml_test.py::check_vocab_size

test_all: test_api_root test_api_predict
