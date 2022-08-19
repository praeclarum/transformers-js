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

class Seq2SeqLMOutput {
    constructor(logits, pastKeyValues, encoderOutputs) {
        this.logits = logits;
        this.pastKeyValues = pastKeyValues;
        this.encoderOutputs = encoderOutputs;
    }
}

class T5ForConditionalGeneration extends AutoModelForSeq2SeqLM {
    constructor(encoderSource, initDecoderSource, decoderSource) {
        super(encoderSource, initDecoderSource, decoderSource);
    }

    async generate(inputTokenIds, initialOutputTokenIds) {
        let encoderOutputs = null;
        let pastKeyValues = null;
        let outputTokenIds = [initialOutputTokenIds[0]];
        let numOutputTokens = 1;
        const maxOutputTokens = numOutputTokens + 2;
        while (numOutputTokens < maxOutputTokens) {
            let output = await this.forward(inputTokenIds, outputTokenIds, encoderOutputs, pastKeyValues);
            pastKeyValues = output.pastKeyValues;
            encoderOutputs = output.encoderOutputs;
            let newTokenId = this.sample(output.logits, numOutputTokens);
            outputTokenIds.push(newTokenId);
            numOutputTokens++;
        }
        return outputTokenIds;
    }

    sample(logits, index) {
        return 42;
    }

    async forward(inputIds, decoderInputIds, encoderOutputs, pastKeyValues) {
        await this.ensureLoaded();

        const inputIdsTensor = new ort.Tensor("int64", new BigInt64Array(inputIds.map(x => BigInt(x))), [1, inputIds.length]);
        const encoderAttentionMaskTensor = new ort.Tensor("int64", new BigInt64Array(inputIds.length).fill(1n), [1, inputIds.length]);
        if (encoderOutputs === null) {
            console.log("Encoding...");
            const encoderFeeds = {
                "input_ids": inputIdsTensor,
                "attention_mask": encoderAttentionMaskTensor,
            }
            const encoderResults = await this.encoderSession.run(encoderFeeds);
            const encoderHiddenStates = encoderResults.hidden_states;
            encoderOutputs = encoderHiddenStates;
            console.log("Encoding done.", encoderOutputs);
        }

        const decoderInputIdsTensor = new ort.Tensor("int64", new BigInt64Array(decoderInputIds.map(x => BigInt(x))), [1, decoderInputIds.length]);
        const decoderAttentionMaskTensor = new ort.Tensor("int64", new BigInt64Array(decoderInputIds.length).fill(1n), [1, decoderInputIds.length]);
        const decoderFeeds = {
            // "input_ids": decoderInputIdsTensor,
            // "encoder_attention_mask": decoderAttentionMaskTensor,
            "input_ids": decoderInputIdsTensor,
            "encoder_attention_mask": encoderAttentionMaskTensor,
            "encoder_hidden_states": encoderOutputs,
        };
        let logits = null;

        if (pastKeyValues === null) {
            console.log("Init Decoding...");
            const initDecoderResults = await this.initDecoderSession.run(decoderFeeds);
            logits = initDecoderResults.logits;
            pastKeyValues = this.getPastKeyValues(initDecoderResults);
            console.log("Init Decoding done.", logits, pastKeyValues);
        }
        else {
            console.log("Decoding...");
            for (const [k, v] of pastKeyValues) {
                decoderFeeds[k] = v;
            }
            const decoderResults = await this.decoderSession.run(decoderFeeds);
            logits = decoderResults.logits;
            pastKeyValues = this.getPastKeyValues(decoderResults);
            console.log("Decoding done.", logits, pastKeyValues);
        }
        return new Seq2SeqLMOutput(logits, pastKeyValues, encoderOutputs);
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


