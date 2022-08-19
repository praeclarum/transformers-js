# Huggingface Transformers Running in the Browser

[tokenizers.js](src/tokenizers.js)

Browser-compatible, js-only, huggingface transformer support.


## Tools

### Convert T5 Models to ONNX Model Format

The ONNX Runtime for the Web is used to run models in the browser.

```bash
python3 tools/convert_model.py <model_id>
```