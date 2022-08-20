DEMODIR=dist

all: all_tests

clean:
	rm -rf $(DEMODIR)
	rm -rf buildmodel

run:
	npx light-server -s . -p 8152 --no-reload

demo: demomodel
	mkdir -p $(DEMODIR)
	mkdir -p $(DEMODIR)/src
	cp -a src/*.js $(DEMODIR)/src
	mkdir -p $(DEMODIR)/tests
	cp -a tests/*.html $(DEMODIR)/tests
	cp -a tests/*.json $(DEMODIR)/tests
	cp -a index.html $(DEMODIR)/
	cp -a icon.png $(DEMODIR)/favicon.ico
	cp -a icon.png $(DEMODIR)/icon.png
	@echo "Built Demo!"

demomodel: $(DEMODIR)/models/t5-small-decoder-quantized.onnx
	@echo "Built Demo Model!"

$(DEMODIR)/models/t5-small-decoder-quantized.onnx: tools/convert_model.py
	pip3 install -r requirements.txt
	python3 tools/convert_model.py t5-small $(DEMODIR)/models

test:
	cd tests && node test.js

all_tests: tests/t5-base-tests.json

tests/t5-base-tests.json: tests/make_tests.py
	cd tests && python make_tests.py


