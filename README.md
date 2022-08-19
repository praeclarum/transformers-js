# Huggingface Transformers Running in the Browser

Browser-compatible, js-only, huggingface transformer support.

## Library

### Tokenizers

[tokenizers.js](src/tokenizers.js)

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

onnx_model = t5_to_onnx(model_id, output_dir, quantized)
```
