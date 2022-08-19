DEMODIR=dist

all: all_tests

demo:
	rm -rf $(DEMODIR)
	mkdir -p $(DEMODIR)
	mkdir -p $(DEMODIR)/src
	cp -a src/*.js $(DEMODIR)/src
	cp -a index.html $(DEMODIR)/
	cp -a test.html $(DEMODIR)/
	cp -a favicon.png $(DEMODIR)/favicon.ico
	@echo "Built Demo!"

run:
	npx light-server -s . -p 8152 --no-reload

test:
	cd tests && node test.js

all_tests: tests/t5-base-tests.json

tests/t5-base-tests.json: tests/make_tests.py
	cd tests && python make_tests.py


