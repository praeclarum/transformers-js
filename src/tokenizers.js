

console.log("Tokenizers");

class Tokenizer {
    constructor(vocab, normalizer, decoder) {
        this.vocab = vocab;
        this.normalizer = normalizer;
        this.decoder = decoder;
        this.vocabIndex = {};
        this.vocab.forEach((token, index) => {
            this.vocabIndex[String(token[0])] = index;
        });
        this.starts = {};
    }
    encode(text) {
        return text.split(" ");
    }
    decode(tokens) {
        return tokens.join(" ");
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

