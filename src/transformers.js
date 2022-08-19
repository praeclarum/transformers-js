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
        const suffix = "-quantized";
        const initDecoderUrl = `${modelsPath}/${modelName}-init-decoder${suffix}.onnx`;
        const decoderUrl = `${modelsPath}/${modelName}-decoder${suffix}.onnx`;
        const encoderUrl = `${modelsPath}/${modelName}-encoder${suffix}.onnx`;
        return new T5ForConditionalGeneration(encoderUrl, initDecoderUrl, decoderUrl);
    }
    async ensureLoaded() {
        if (this.encoderSession === null && this.encoderSource) {
            console.log('Loading encoder...');
            this.encoderSession = await ort.InferenceSession.create(this.encoderSource);
        }
        if (this.initDecoderSession === null && this.initDecoderSource) {
            console.log('Loading init decoder...');
            this.initDecoderSession = await ort.InferenceSession.create(this.initDecoderSource);
        }
        if (this.decoderSession === null && this.decoderSource) {
            console.log('Loading decoder...');
            this.decoderSession = await ort.InferenceSession.create(this.decoderSource);
            console.log('Done loading decoder.');
        }
    }
}


class T5ForConditionalGeneration extends AutoModelForSeq2SeqLM {
    constructor(encoderSource, initDecoderSource, decoderSource) {
        super(encoderSource, initDecoderSource, decoderSource);
    }
    async generate(inputTokens) {
        await this.ensureLoaded();

        console.log("Encoding...");
        const inputIds = new ort.Tensor("int64", new BigInt64Array(inputTokens.map(x => BigInt(x))), [1, inputTokens.length]);
        const encoderAttentionMask = new ort.Tensor("int64", new BigInt64Array(inputTokens.length).fill(1n), [1, inputTokens.length]);
        const encoderFeeds = {
            "input_ids": inputIds,
            "attention_mask": encoderAttentionMask,
        }
        const encoderResults = await this.encoderSession.run(encoderFeeds);
        const encoderHiddenStates = encoderResults.hidden_states;
        console.log("Encoding done.", encoderResults);

        console.log("Init Decoding...");
        const decoderInputIds = new ort.Tensor("int64", new BigInt64Array(inputTokens.map(x => BigInt(x))), [1, inputTokens.length]);
        const decoderAttentionMask = new ort.Tensor("int64", new BigInt64Array(inputTokens.length).fill(1n), [1, inputTokens.length]);
        const initDecoderFeeds = {
            "input_ids": decoderInputIds,
            "encoder_attention_mask": decoderAttentionMask,
            "encoder_hidden_states": encoderHiddenStates,
        };
        const initDecoderResults = await this.initDecoderSession.run(initDecoderFeeds);
        const initDecoderPastKeyValues = this.getPastKeyValues(initDecoderResults);
        console.log("Init Decoding done.", initDecoderResults, initDecoderPastKeyValues);

        console.log("Decoding...");
        const decoderFeeds = {
            "input_ids": decoderInputIds,
            "encoder_attention_mask": decoderAttentionMask,
            "encoder_hidden_states": encoderHiddenStates,
        };
        for (const [k, v] of initDecoderPastKeyValues) {
            decoderFeeds[k] = v;
        }
        const decoderResults = await this.decoderSession.run(decoderFeeds);
        const logits = decoderResults.logits;
        const decoderPastKeyValues = this.getPastKeyValues(decoderResults);
        console.log("Decoding done.", logits, decoderPastKeyValues);
        return inputTokens;
    }

    getPastKeyValues(decoderResults) {
        const orig_past_key_values = [];
        for (const k in decoderResults) {
            if (decoderResults.hasOwnProperty(k)) {
                let newK = k;
                if (newK === "logits") continue;
                if (newK === "past_key_values" || newK === "output_past_key_values") {
                    newK = 0;
                }
                else {
                    newK = parseInt(/\d+/.exec(k)[0]);
                }
                const v = decoderResults[k];
                orig_past_key_values.push([newK, v])
            }
        }
        const past_key_values = orig_past_key_values.sort((a, b) => a[0] - b[0]).map((x, i) => ["pkv_" + i, x[1]]);
        return past_key_values;
    }
}


