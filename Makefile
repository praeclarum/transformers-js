
all: all_tests

build:
	rm -rf dist
	mkdir -p dist
	mkdir -p dist/src
	cp -a src/*.js dist/src
	cp -a index.html dist/
	cp -a test.html dist/
	@echo "Built!"

run:
	npx light-server -s . -p 8152 --no-reload

test:
	cd tests && node test.js

all_tests: tests/t5-base-tests.json

tests/t5-base-tests.json: tests/make_tests.py
	cd tests && python make_tests.py


