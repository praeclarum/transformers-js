from transformers import AutoTokenizer


def t5_to_onnx(model_id, output_dir, quantized):
    import fastT5
    model = fastT5.export_and_get_onnx_model(model_id, custom_output_path=output_dir, quantized=quantized)
    return model


def onnx_generate(input, model_id, onnx_model):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    token = tokenizer(input, return_tensors='pt')
    tokens = onnx_model.generate(input_ids=token['input_ids'],
                                 attention_mask=token['attention_mask'],
                                 num_beams=2)
    output = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)
    return output


if __name__ == '__main__':
    model_id = "t5-small"
    test_input = "translate English to French: The universe is a dark forest."
    output_dir = "./models"
    quantized = True
    onnx_model = t5_to_onnx(model_id, output_dir, quantized)
    test_output = onnx_generate(test_input, model_id, onnx_model)
    print(f"> {test_input}")
    print(f"< {test_output}")
