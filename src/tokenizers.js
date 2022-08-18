//"use strict";

console.log("Tokenizers");

class Tokenizer {
    constructor(vocab, normalizer, decoder) {
        this.vocab = vocab;
        this.normalizer = normalizer;
        this.decoder = decoder;
        this.tokenToIds = new Map(vocab.map((x, i) => [this.normalize(x[0]), i]));
        this.bosToken = this.normalize(" ");
        this.bosTokenId = this.getTokenId(this.bosToken);
        this.eosToken = "</s>";
        this.unkToken = "<unk>";
        this.trie = new CharTrie();
        vocab.forEach(x => this.trie.push(x[0]));
    }
    getTokenId(normalizedToken) {
        return this.tokenToIds.get(normalizedToken);
    }
    preprocess(text) {
        return " " + text + this.eosToken;
    }
    normalize(text) {
        return text.replace(/\s+/g, this.decoder.replacement);
    }
    denormalize(normalized) {
        return normalized.replaceAll(this.decoder.replacement, " ");
    }
    populateNodes(lattice) {
        const unkScore = this.minScore - 10.0;

        const sentence = lattice.sentence;
        const len = sentence.length;

        let beginPos = 0;
        while (beginPos < len) {
            const mblen = 1;
            let hasSingleNode = false;
            // console.log("PREFIX SEARCH", sentence.slice(beginPos));
            const tokens = [];
            for (let token of this.trie.commonPrefixSearch(sentence.slice(beginPos))) {
                tokens.push(token);
                const tokenId = this.getTokenId(token);
                const tokenScore = this.vocab[tokenId][1];
                const n = token.length;
                lattice.insert(beginPos, n, tokenScore, tokenId);
                if (!hasSingleNode && n == mblen) {
                    hasSingleNode = true;
                }
            }
            // console.log("SEARCH RESULTS", tokens);
            if (!hasSingleNode) {
                lattice.insert(beginPos, mblen, unkScore, this.unkTokenId);
            }
            beginPos += mblen;
        }
    }
    tokenize(normalized) {
        const lattice = new TokenLattice(normalized, this.bosTokenId, this.eosTokenId);
        this.populateNodes(lattice);
        const tokens = lattice.tokens();
        return tokens.map(x => this.tokenToIds.get(x));
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

class CharTrie {
    constructor() {
        this.root = CharTrieNode.default();
    }
    push(text) {
        let node = this.root;
        for (let i = 0; i < text.length; i++) {
            const ch = text[i];
            let child = node.children.get(ch);
            if (child === undefined) {
                child = CharTrieNode.default();
                node.children.set(ch,  child);
            }
            node = child;
        }
        node.isLeaf = true;
    }
    *commonPrefixSearch(text) {
        let node = this.root;
        let prefix = "";
        for (let i = 0; i < text.length && node !== undefined; i++) {
            const ch = text[i];
            prefix += ch;
            node = node.children.get(ch);
            if (node !== undefined && node.isLeaf) {
                yield prefix;
            }
        }
    }
}

class CharTrieNode {
    constructor(isLeaf, children) {
        this.isLeaf = isLeaf;
        this.children = children;
    }
    static default() {
        return new CharTrieNode(false, new Map());
    }
}

class TokenLattice {
    constructor(sentence, bosTokenId, eosTokenId) {
        this.sentence = sentence;
        this.len = sentence.length;
        this.bosTokenId = bosTokenId;
        this.eosTokenId = eosTokenId;
        this.nodes = [];
        this.beginNodes = new Array(this.len + 1);
        this.endNodes = new Array(this.len + 1);
        for (let i = 0; i < this.len + 1; i++) {
            this.beginNodes[i] = [];
            this.endNodes[i] = [];
        }
        const bos = new SentenceLatticeNode(this.bosTokenId, 0, 0, 0, 0.0);
        const eos = new SentenceLatticeNode(this.eosTokenId, 1, this.len, 0, 0.0);
        this.nodes.push(bos.clone());
        this.nodes.push(eos.clone());
        this.beginNodes[this.len].push(eos);
        this.endNodes[0].push(bos);
    }

    insert(pos, length, score, tokenId) {
        const nodeId = this.nodes.length;
        const node = new SentenceLatticeNode(tokenId, nodeId, pos, length, score);
        this.beginNodes[pos].push(node);
        this.endNodes[pos + length].push(node);
        this.nodes.push(node);
    }

    viterbi() {
        const len = this.len;
        let pos = 0;
        while (pos <= len) {
            if (this.beginNodes[pos].length == 0) {
                return [];
            }
            for (let rnode of this.beginNodes[pos]) {
                rnode.prev = null;
                let bestScore = 0.0;
                let bestNode = null;
                for (let lnode of this.endNodes[pos]) {
                    const score = lnode.backtraceScore + rnode.score;
                    if (bestNode === null || score > bestScore) {
                        bestNode = lnode.clone();
                        bestScore = score;
                    }
                }
                if (bestNode !== null) {
                    rnode.prev = bestNode;
                    rnode.backtraceScore = bestScore;
                }
                else {
                    return [];
                }
            }
            pos++;
        }
        const results = [];
        const root = this.beginNodes[len][0];
        const prev = root.prev;
        if (prev === null) {
            return [];
        }
        let node = prev.clone();
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
        return nodes.map(x => this.piece(x));
    }
}

class SentenceLatticeNode {
    constructor(tokenId, nodeId, pos, length, score) {
        this.tokenId = tokenId;
        this.nodeId = nodeId;
        this.pos = pos;
        this.length = length;
        this.score = score;
        this.prev = null;
        this.backtraceScore = 0.0;
    }
    clone() {
        const n = new SentenceLatticeNode(this.tokenId, this.nodeId, this.pos, this.length, this.score);
        n.prev = this.prev;
        n.backtraceScore = this.backtraceScore;
        return n;
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

