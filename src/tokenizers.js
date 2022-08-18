

console.log("Tokenizers");

class Tokenizer {
    constructor(vocab, normalizer, decoder) {
        this.vocab = vocab;
        this.normalizer = normalizer;
        this.decoder = decoder;
        this.vocabIndex = new Map(vocab.map((x, i) => [this.normalize(x), i]));
        this.starts = {};
    }
    preprocess(text) {
        return " " + text + "</s>";
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
            if (token.startsWith(start)) {
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
        let prevToken = null;
        const tokens = [];
        while (p < e) {
            const maybeToken = normalized.slice(b, p);
            if (maybeToken.length == 0) {
                p++;
                continue;
            }
            const starts = this.getStarts(maybeToken);
            if (starts.length == 0) {
                if (prevToken) {
                    tokens.push(prevToken);
                    prevToken = null;
                    b = p - 1;
                }
                else {
                    throw new Error("No token found for " + maybeToken);
                }
            } else if (starts.length == 1) {
                const start = starts[0];
                if (start.length == maybeToken.length) {
                    tokens.push(starts[0]);
                    prevToken = null;
                    b = p;
                    p++;
                }
                else {
                    p++;
                }
            } else {
                prevToken = maybeToken;
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
        const normalized = tokens.map(x => this.vocab[x]).join("");
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
    return new Tokenizer(jtokenizer.model.vocab.map(x => x[0]), loadNormalizer(jtokenizer.normalizer), jtokenizer.decoder);
}

