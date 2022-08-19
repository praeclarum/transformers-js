import sys, os, shutil
from transformers import AutoTokenizer


def t5_to_onnx(model_id, output_dir, quantized):
    import fastT5
    model = fastT5.export_and_get_onnx_model(model_id, custom_output_path=output_dir, quantized=quantized)
    return model


def onnx_generate(input, onnx_model, tokenizer):
    token = tokenizer(input, return_tensors='pt')
    tokens = onnx_model.generate(input_ids=token['input_ids'],
                                 attention_mask=token['attention_mask'],
                                 num_beams=2)
    output = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)
    return output


if __name__ == '__main__':
    args = sys.argv
    model_id = "t5-small"
    output_dir = "./models"
    quantized = True
    test_input = "translate English to French: The universe is a dark forest."

    if len(args) > 1:
        model_id = args[1]
    if len(args) > 2:
        output_dir = args[2]
    if len(args) > 3:
        quantized = args[3].lower() == "true" or args[3].lower() == "1" or args[3].lower() == "yes"
    if len(args) > 4:
        test_input = args[4]
    model_name = model_id.split("/")[-1]

    print(f"model_name: {model_name}")
    print(f"  model_id: {model_id}")
    print(f"output_dir: {output_dir}")
    print(f" quantized: {quantized}")

    os.makedirs(output_dir, exist_ok=True)

    build_dir = os.path.abspath(f"buildmodel")
    os.makedirs(build_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(build_dir)
    shutil.copyfile(os.path.join(build_dir, "tokenizer.json"), os.path.join(output_dir, f"{model_name}-tokenizer.json"))

    onnx_model = t5_to_onnx(model_id, build_dir, quantized)
    msuffix = "-quantized" if quantized else ""
    for session in ["encoder", "init-decoder", "decoder"]:
        shutil.copyfile(os.path.join(build_dir, f"{model_name}-{session}{msuffix}.onnx"), os.path.join(output_dir, f"{model_name}-{session}{msuffix}.onnx"))

    test_output = onnx_generate(test_input, onnx_model, tokenizer)
    print(f"> {test_input}")
    print(f"< {test_output}")
