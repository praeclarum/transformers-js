# Huggingface Transformers Running in the Browser

[![Azure Static Web Apps CI/CD](https://github.com/praeclarum/transformers-js/actions/workflows/azure-static-web-apps-gentle-desert-0ddc8ce10.yml/badge.svg)](https://github.com/praeclarum/transformers-js/actions/workflows/azure-static-web-apps-gentle-desert-0ddc8ce10.yml)

This library enables you to run huggingface transformer models directly in the browser.
It accomplishes this by running the models using the
[ONNX Runtime JavaScript API](https://github.com/microsoft/onnxruntime/tree/main/js)
and by implementing its own JavaScript-only tokenization library.

At the moment, it is compatible with Google's T5 models, but it was designed to be expanded.
I hope to support GPT2, Roberta, and InCoder in the future.


## Live Demo

[https://transformers-js.praeclarum.org](https://transformers-js.praeclarum.org)

This demo is a static website hosted on [Azure Static Web Apps](https://azure.microsoft.com/en-us/services/app-service/static/).
No code is executed on the server. Instead, the neural network is downloaded and executed in the browser.

See the [Makefile](Makefile) `demo` rule to see how the demo is built.


## Usage

This example shows how to use the library to load the T5 neural network to translate from English to French.

```js
// Load the tokenizer and model.
const tokenizer = await AutoTokenizer.fromPretrained("t5-small", "/models");
const model = await AutoModelForSeq2SeqLM.fromPretrained("t5-small", "/models", "wasm", null);

// Translate "Hello, world!"
const english = "Hello, world!";
const inputTokenIds = tokenizer.encode("translate English to French: " + english);
const outputTokenIds = await model.generate(inputTokenIds, {maxLength:50});
const french = tokenizer.decode(outputTokenIds, true);
console.log(french); // "Bonjour monde!"
```

To run this demo, you need to have converted the model to ONNX format using the [Model Conversion Tool](#model-converter).

```bash
python3 tools/convert_model.py t5-small models
```


## Library

The library contains several components:

1. [Tokenizers](#tokenizers) to load and execute pretrained tokenizers from their huggingface JSON representation.
2. [Transformers](#transformers) to load and execute pretrained models from their ONNX representation.
3. [Model Converter](#model-converter) to convert huggingface models to ONNX to be served by your static web server.


### Tokenizers

[tokenizers.js](src/tokenizers.js)


### Transformers

[transformers.js](src/transformers.js)


#### Models

Currently only the *T5* network is supported.


#### Sampling

The neural network outputs the logarithm of the probability of each token.
In order to get a token, a probabilistic sample has to be taken.
The following algorithms are implemented:

* *Greedy*: Take the token with the highest probability.
* *Top-k*: Take the top-k tokens with the highest probability.


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

**Developer Note:** The model conversion script is a thin wrapper over the amazing
[fastT5](https://github.com/Ki6an/fastT5) library by @Ki6an.
The wrapper exists because I hope to support more model types in the future.
