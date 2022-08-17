
all: all_tests

run:
	npx light-server -s . -p 8152 --no-reload

test:
	cd tests && node test.js

all_tests: tests/t5-base-tests.json

tests/t5-base-tests.json: tests/make_tests.py
	cd tests && python make_tests.py


