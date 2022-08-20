class PretrainedModel {
    static async loadSession(modelSource) {
        console.log('Loading session from', modelSource);
        const response = await fetch(modelSource, { cache: 'force-cache' });
        const modelBuffer = await response.arrayBuffer();
        const session = await ort.InferenceSession.create(modelBuffer, { executionProviders: ["wasm"] });
        console.log('Session loaded from', modelSource);
        return session;
    }
}


class AutoModelForSeq2SeqLM extends PretrainedModel {
    constructor(encoderSession, initDecoderSession, decoderSession) {
        super();
        this.encoderSession = encoderSession;
        this.initDecoderSession = initDecoderSession;
        this.decoderSession = decoderSession;
    }

    static async fromPretrained(modelId, modelsPath, progressAsyncCallback) {
        // TODO: This should load different model types. Right now it's hardcoded to T5.

        const modelIdParts = modelId.split('/');
        const modelName = modelIdParts[modelIdParts.length - 1];
        const suffix = "-quantized";
        const encoderUrl = `${modelsPath}/${modelName}-encoder${suffix}.onnx`;
        const initDecoderUrl = `${modelsPath}/${modelName}-init-decoder${suffix}.onnx`;
        const decoderUrl = `${modelsPath}/${modelName}-decoder${suffix}.onnx`;

        const progressMax = 4;
        let progress = 0;
        async function incrementProgress() {
            progress++;
            const p = progress / progressMax;
            console.log(`Loading model ${modelId}... ${p * 100}%`);
            if (progressAsyncCallback) {
                await progressAsyncCallback(p);
            }
        }
        await incrementProgress();
        const encoderSessionPromise = this.loadSession(encoderUrl);
        const initDecoderSessionPromise = this.loadSession(initDecoderUrl);
        const decoderSessionPromise = this.loadSession(decoderUrl);
        const encoderSession = await encoderSessionPromise;
        await incrementProgress();
        const initDecoderSession = await initDecoderSessionPromise;
        await incrementProgress();
        const decoderSession = await decoderSessionPromise;
        await incrementProgress();

        return new T5ForConditionalGeneration(encoderSession, initDecoderSession, decoderSession);
    }

    /**
     * Generate a sequence of tokens.
     * 
     * @param {Array} inputTokenIds
     * @param {Object} options Properties:
     *  `maxLength` for the maximum generated sequence length,
     *  `topK` for the number of logits to consider when sampling.
     * @param {Promise} progressAsyncCallback 
     * @returns The generated sequence of tokens.
     */
    async generate(inputTokenIds, options, progressAsyncCallback) {
        const maxLength = options.maxLength || 100;
        const topK = options.topK || 0;
        const topP = options.topP || 0;
        const numBeams = options.numBeams || 0;
        // attention_mask=token['attention_mask'], num_beams=2
        const startOfDecoderTokenId = 0;
        const endOfDecoderTokenId = 1;
        let encoderOutputs = null;
        let pastKeyValues = null;
        let outputTokenIds = [startOfDecoderTokenId];
        let numOutputTokens = 1;
        let shouldContinue = true;
        const maxOutputTokens = numOutputTokens + maxLength;
        async function progress() {
            if (progressAsyncCallback) {
                shouldContinue = await progressAsyncCallback(outputTokenIds, inputTokenIds);
            }
        }
        let sampler = x => this.sampleLogitsGreedily(x);
        if (topK > 0) {
            sampler = x => this.sampleLogitsTopK(x, topK);
        }
        while (shouldContinue && numOutputTokens < maxOutputTokens) {
            let output = await this.forward(inputTokenIds, outputTokenIds, encoderOutputs, pastKeyValues);
            pastKeyValues = output.pastKeyValues;
            encoderOutputs = output.encoderOutputs;
            let newTokenId = sampler(output.logits);
            outputTokenIds.push(newTokenId);
            numOutputTokens++;
            await progress(outputTokenIds);
            if (newTokenId === endOfDecoderTokenId) {
                break;
            }
        }
        return outputTokenIds;
    }

    sampleLogitsGreedily(logits) {
        let shape = logits.dims;
        let [batchSize, seqLength, vocabSize] = shape;
        let n = batchSize * seqLength * vocabSize;
        let startIndex = n - vocabSize;
        let argmaxi = 0;
        let argmax = logits.data[startIndex + argmaxi];
        for (let i = 1; i < vocabSize; i++) {
            let l = logits.data[startIndex + i];
            if (l > argmax) {
                argmaxi = i;
                argmax = l;
            }
        }
        return argmaxi;
    }
    sampleLogitsTopK(logits, k) {
        let shape = logits.dims;
        let [batchSize, seqLength, vocabSize] = shape;
        let n = batchSize * seqLength * vocabSize;
        let startIndex = n - vocabSize;
        let logs = logits.data.slice(startIndex);
        k = Math.min(k, vocabSize);
        let logitAndId = Array.from(logs).map((x, i) => [x, i])
            .sort((a, b) => b[0] - a[0]);
        const sMin = Math.exp(-100.0);
        let sumS = 0.0;
        for (let i = 0; i < logitAndId.length; i++) {
            const s = i < k ? Math.exp(logitAndId[i][0]) : sMin;
            sumS += s;
            logitAndId[i][0] = s;
        }
        let r = Math.random() * sumS;
        for (let i = 0; i < logitAndId.length; i++) {
            r -= logitAndId[i][0];
            if (r <= 0) {
                return logitAndId[i][1];
            }
        }
        return logitAndId[0][1];
    }
}

class T5ForConditionalGeneration extends AutoModelForSeq2SeqLM {
    constructor(encoderSession, initDecoderSession, decoderSession) {
        super(encoderSession, initDecoderSession, decoderSession);
    }

    async forward(inputIds, decoderInputIds, encoderOutputs, pastKeyValues) {
        const inputIdsTensor = new ort.Tensor("int64", new BigInt64Array(inputIds.map(x => BigInt(x))), [1, inputIds.length]);
        const encoderAttentionMaskTensor = new ort.Tensor("int64", new BigInt64Array(inputIds.length).fill(1n), [1, inputIds.length]);
        if (encoderOutputs === null) {
            // console.log("Encoding...");
            const encoderFeeds = {
                "input_ids": inputIdsTensor,
                "attention_mask": encoderAttentionMaskTensor,
            }
            const encoderResults = await this.encoderSession.run(encoderFeeds);
            const encoderHiddenStates = encoderResults.hidden_states;
            encoderOutputs = encoderHiddenStates;
            // console.log("Encoding done.", encoderOutputs);
        }

        const decoderInputIdsTensor = new ort.Tensor("int64", new BigInt64Array(decoderInputIds.map(x => BigInt(x))), [1, decoderInputIds.length]);
        // const decoderAttentionMaskTensor = new ort.Tensor("int64", new BigInt64Array(decoderInputIds.length).fill(1n), [1, decoderInputIds.length]);
        const decoderFeeds = {
            "input_ids": decoderInputIdsTensor,
            "encoder_attention_mask": encoderAttentionMaskTensor,
            "encoder_hidden_states": encoderOutputs,
        };
        let logits = null;

        if (pastKeyValues === null) {
            // console.log("Init Decoding...");
            const initDecoderResults = await this.initDecoderSession.run(decoderFeeds);
            logits = initDecoderResults.logits;
            pastKeyValues = this.getPastKeyValues(this.initDecoderSession.outputNames.slice(1), initDecoderResults);
            // console.log("Init Decoding done.", logits, pastKeyValues);
        }
        else {
            // console.log("Decoding...");
            for (const [k, v] of pastKeyValues) {
                decoderFeeds[k] = v;
            }
            const decoderResults = await this.decoderSession.run(decoderFeeds);
            logits = decoderResults.logits;
            pastKeyValues = this.getPastKeyValues(this.decoderSession.outputNames.slice(1), decoderResults);
            // console.log("Decoding done.", logits, pastKeyValues);
        }
        return new Seq2SeqLMOutput(logits, pastKeyValues, encoderOutputs);
    }

    getPastKeyValues(pkvNames, decoderResults) {
        const pkvs = [];
        for (const i in pkvNames) {
            const k = pkvNames[i];
            const v = decoderResults[k];
            pkvs.push([`pkv_${i}`, v]);
        }
        return pkvs;
    }
}


class Seq2SeqLMOutput {
    constructor(logits, pastKeyValues, encoderOutputs) {
        this.logits = logits;
        this.pastKeyValues = pastKeyValues;
        this.encoderOutputs = encoderOutputs;
    }
}

