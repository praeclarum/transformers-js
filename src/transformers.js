class PretrainedModel {

}


class AutoModelForSeq2SeqLM {
    constructor(encoderSource, initDecoderSource, decoderSource) {
        this.encoderSource = encoderSource;
        this.initDecoderSource = initDecoderSource;
        this.decoderSource = decoderSource;
        this.encoderSession = null;
        this.initDecoderSession = null;
        this.decoderSession = null;
    }
    static async fromPretrained(modelId, modelsPath) {
        const modelIdParts = modelId.split('/');
        const modelName = modelIdParts[modelIdParts.length - 1];
        const initDecoderUrl = `${modelsPath}/${modelName}-init-decoder-quantized.onnx`;
        const decoderUrl = `${modelsPath}/${modelName}-decoder-quantized.onnx`;
        const encoderUrl = `${modelsPath}/${modelName}-encoder-quantized.onnx`;
        return new T5Model(encoderUrl, initDecoderUrl, decoderUrl);
    }
    async ensureLoaded() {
        if (this.encoderSession === null) {
            this.encoderSession = new onnx.InferenceSession(this.encoderSource);
            await this.encoderSession.loadModel();
        }
    }
}


class T5Model extends AutoModelForSeq2SeqLM {
    constructor(encoderSource, initDecoderSource, decoderSource) {
        super(encoderSource, initDecoderSource, decoderSource);
    }
    async generate(inputTokens) {
        this.ensureLoaded();
    }
}


