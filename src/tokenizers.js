"use strict";

console.log("Tokenizers");

class Tokenizer {
    constructor(vocab, normalizer, decoder) {
        this.vocab = vocab;
        this.normalizer = normalizer;
        this.decoder = decoder;
        this.vocabIndex = new Map(vocab.map((x, i) => [this.normalize(x[0]), i]));
        this.starts = {};
        this.eos = "</s>";
        this.unk = "<unk>";
    }
    preprocess(text) {
        return " " + text + this.eos;
    }
    normalize(text) {
        return text.replace(/\s+/g, this.decoder.replacement);
    }
    denormalize(text) {
        return text.replaceAll(this.decoder.replacement, " ");
    }
    getStarts(start) {
        if (start.length <= 0) {
            return this.vocab;
        }
        if (this.starts[start]) {
            return this.starts[start];
        }
        const parentTokens = this.getStarts(start.slice(0, -1));
        const starts = [];
        parentTokens.forEach(token => {
            if (token[0].startsWith(start)) {
                starts.push(token);
            }
        });
        this.starts[start] = starts;
        return starts;
    }
    tokenize(normalized) {
        let b = 0;
        let e = normalized.length;
        let p = 0;
        let bestMatchToken = null;
        let bestMatchScore = 1.0e6;
        let bestMatchP = 0;
        let bestMatchB = 0;
        const tokens = [];
        while (p < e) {
            const maybeToken = normalized.slice(b, p);
            if (maybeToken.length == 0) {
                p++;
                continue;
            }
            const starts = this.getStarts(maybeToken);
            const match = starts.find(x => x[0] == maybeToken);
            if (match !== undefined) {
                const matchScore = match[1];
                if (matchScore + bestMatchScore < bestMatchScore) {
                    bestMatchScore = matchScore + bestMatchScore;
                    bestMatchToken = match[0];
                    bestMatchP = p;
                    bestMatchB = b;
                }
            }

            if (starts.length == 0) {
                if (bestMatchToken) {
                    tokens.push(bestMatchToken);
                    b = bestMatchP;
                    p = b + 1;
                }
                else {
                    tokens.push(this.unk);
                    b = p;
                }
                bestMatchToken = null;
                bestMatchScore = 1.0e6;
            } else {
                p++;
            }
        }
        const lastToken = normalized.slice(b);
        if (lastToken.length > 0) {
            tokens.push(lastToken);
        }
        return tokens.map(x => this.vocabIndex.get(x));
    }
    encode(text) {
        const pp = this.preprocess(text);
        const normalized = this.normalize(pp);
        const tokenized = this.tokenize(normalized);
        return tokenized;
    }
    decode(tokens) {
        const normalized = tokens.map(x => {
            if (x in this.vocab) {
                return this.vocab[x][0];
            }
            else {
                return '[' + x + ']';
            }
        }).join('');
        return this.denormalize(normalized).slice(1);
    }
}

async function loadTokenizer(url) {
    function loadNormalizer(jnormalizer) {
        switch (jnormalizer.type) {
            case "Precompiled":
                return jnormalizer;
            default:
                throw new Error("Unknown normalizer type: " + jnormalizer.type);
        }
    }
    const response = await fetch(url);
    const jtokenizer = await response.json();
    console.log(jtokenizer);
    return new Tokenizer(jtokenizer.model.vocab, loadNormalizer(jtokenizer.normalizer), jtokenizer.decoder);
}

