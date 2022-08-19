# Huggingface Transformers Running in the Browser

Browser-compatible, js-only, huggingface transformer support.


## Live Demo

[https://transformers-js.praeclarum.org](https://transformers-js.praeclarum.org)

This demo is a static website hosted on [Azure Static Web Sites](https://azure.microsoft.com/en-us/services/static-websites/).
No code is executed on the server. Instead, the neural network is downloaded and executed in the browser.

See the [Makefile](Makefile) `demo` rule to see how the demo is built.


## Usage

```js
const tokenizer = await AutoTokenizer.fromPretrained("t5-small", models_path="/models");
const model = await AutoModelForSeq2SeqLM.fromPretrained("t5-small", models_path="/models");
```


## Library

### Tokenizers

[tokenizers.js](src/tokenizers.js)

### Transformers

[transformers.js](src/transformers.js)

### Transformers

## Tools

### Convert T5 Models to ONNX Model Format

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
