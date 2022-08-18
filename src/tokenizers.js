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

class SentenceLattice {
    constructor(sentence, bos_id, eos_id) {
        this.sentence = sentence;
        this.len = sentence.length;
        this.bos_id = bos_id;
        this.eos_id = eos_id;
        this.nodes = [];
        this.begin_nodes = new Array(len + 1);
        this.end_nodes = new Array(len + 1);
        for (let i = 0; i < len + 1; i++) {
            this.begin_nodes[i] = [];
            this.end_nodes[i] = [];
        }
        const bos = new SentenceNode(this.bos_id, 0, 0, 0, 0.0);
        const eos = new SentenceNode(this.eos_id, 1, this.len, 0, 0.0);
        this.nodes.push(bos.clone());
        this.nodes.push(eos.clone());
        this.begin_nodes[len].push(eos);
        this.end_nodes[0].push(bos);
    }

    insert(pos, length, score, id) {
        const node_id = this.nodes.length;
        const node = new SentenceNode(id, node_id, pos, length, score);
        this.begin_nodes[pos].push(node.clone());
        this.end_nodes[pos + length].push(node.clone());
        this.nodes.push(node);
    }

    viterbi() {
        const len = this.len;
        let pos = 0;
        while (pos <= len) {
            if (this.begin_nodes[pos].length == 0) {
                return [];
            }
            for (let rnode of this.begin_nodes[pos]) {
                rnode.prev = null;
                let best_score = 0.0;
                let best_node = null;
                for (let lnode of this.end_nodes[pos]) {
                    const score = lnode.backtrace_score + rnode.score;
                    if (best_node === null || score > best_score) {
                        best_node = lnode.clone();
                        best_score = score;
                    }
                }
                if (best_node !== null) {
                    rnode.prev = best_node.clone();
                    rnode.backtrace_score = best_score;
                }
                else {
                    return [];
                }
            }
            pos++;
        }
        const results = [];
        const root = this.begin_nodes[len][0];
        const prev = root.prev;
        if (prev === null) {
            return [];
        }
        const node = prev.clone();
        while (node.prev !== null) {
            results.push(node.clone());
            const n = node.clone();
            node = n.prev.clone();
        }
        results.reverse();
        return results;
    }

    piece(node) {
        return this.sentence.slice(node.pos, node.pos + node.length);
    }

    tokens() {
        const nodes = this.viterbi();
        return nodes.map(this.piece);
    }
}

class SentenceNode {
    constructor(id, node_id, pos, length, score) {
        this.id = id;
        this.node_id = node_id;
        this.pos = pos;
        this.length = length;
        this.score = score;
        this.prev = null;
        this.backtrace_score = 0.0;
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

