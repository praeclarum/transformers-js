# Huggingface Transformers Running in the Browser

Browser-compatible, js-only, huggingface transformer support.


## Live Demo

[https://transformers-js.praeclarum.org](https://transformers-js.praeclarum.org)

This demo is a static website hosted on [Azure Static Web Apps](https://azure.microsoft.com/en-us/services/app-service/static/).
No code is executed on the server. Instead, the neural network is downloaded and executed in the browser.

See the [Makefile](Makefile) `demo` rule to see how the demo is built.


## Usage

This usage example shows how to use the library to load the T5 model to translate from English to French.

```js
// Load the tokenizer and model.
const tokenizer = await AutoTokenizer.fromPretrained("t5-small", models_path="/models");
const model = await AutoModelForSeq2SeqLM.fromPretrained("t5-small", models_path="/models");

// Translate "Hello, world!"
const english = "Hello, world!";
const inputTokenIds = tokenizer.encode("translate English to French: " + english);
const outputTokenIds = await model.generate(inputTokenIds, 140);
const french = tokenizer.decode(outputTokenIds, true);
console.log(french); // "Bonjour monde!"
```


## Library

The library contains several components:

1. [Tokenizers](#tokenizers) to load and execute pretrained tokenizers from their huggingface JSON representation.
2. [Transformer Models](#transformer-models) to load and execute pretrained models from their ONNX representation.
3. [Model Converter](#model-converter) to convert huggingface models to ONNX to be served by your static web server.


### Tokenizers

[tokenizers.js](src/tokenizers.js)


### Transformer Models

[transformers.js](src/transformers.js)


### Model Converter

The ONNX Runtime for the Web is used to run models in the browser.

You can run the conversion from the command line:

```bash
python3 tools/convert_model.py <modelid> <outputdir> <quantize> <testinput>
```

For example:

```bash
python3 tools/convert_model.py praeclarum/cuneiform ./models true "Translate Akkadian to English: lugal"
```

Or you can run it from Python:

```python
from convert_model import t5_to_onnx

onnx_model = t5_to_onnx("t5-small", output_dir="./models", quantized=True)
```
