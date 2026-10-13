package com.zrahay.restrainedml

import org.json.JSONObject

/**
 * Kotlin port of CLIP's BPE tokenizer (open_clip SimpleTokenizer).
 *
 * The algorithm is reimplemented here; the data (vocabulary and ordered
 * merge rules) comes from clip_tokenizer.json, exported from Python by
 * ml/export/export_tokenizer_assets.py. Must produce token IDs identical
 * to mobileclip.get_tokenizer(...) — verified against tokenizer_golden.json.
 */
class ClipTokenizer(json: String) {

    val contextLength: Int
    private val sot: Int
    private val eot: Int
    private val vocab = HashMap<String, Int>(65536)
    private val ranks = HashMap<String, Int>(65536) // key: "left right"
    private val byteEncoder = arrayOfNulls<String>(256)
    private val cache = HashMap<String, List<String>>()

    // CLIP's splitting regex: contractions, letter runs, single digits,
    // punctuation runs (case-insensitive; input is lowercased anyway)
    private val pat = Regex(
        """'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
        RegexOption.IGNORE_CASE
    )

    init {
        val root = JSONObject(json)
        contextLength = root.getInt("context_length")
        sot = root.getInt("sot")
        eot = root.getInt("eot")

        val v = root.getJSONObject("vocab")
        for (key in v.keys()) vocab[key] = v.getInt(key)

        val merges = root.getJSONArray("merges")
        for (i in 0 until merges.length()) ranks[merges.getString(i)] = i

        buildByteEncoder()
    }

    /** CLIP's bytes_to_unicode: maps every byte to a printable unicode char. */
    private fun buildByteEncoder() {
        val bs = ArrayList<Int>()
        for (b in '!'.code..'~'.code) bs.add(b)
        for (b in 0xA1..0xAC) bs.add(b)
        for (b in 0xAE..0xFF) bs.add(b)
        val cs = ArrayList<Int>(bs)
        var n = 0
        for (b in 0..255) {
            if (b !in bs) {
                bs.add(b)
                cs.add(256 + n)
                n++
            }
        }
        for (i in bs.indices) byteEncoder[bs[i]] = cs[i].toChar().toString()
    }

    /** Apply BPE merges to one byte-encoded word. */
    private fun bpe(token: String): List<String> {
        cache[token]?.let { return it }

        var word = ArrayList<String>(token.length)
        for (i in token.indices) word.add(token[i].toString())
        word[word.size - 1] = word[word.size - 1] + "</w>"

        while (word.size > 1) {
            var bestRank = Int.MAX_VALUE
            var bestIdx = -1
            for (i in 0 until word.size - 1) {
                val rank = ranks["${word[i]} ${word[i + 1]}"] ?: continue
                if (rank < bestRank) {
                    bestRank = rank
                    bestIdx = i
                }
            }
            if (bestIdx == -1) break

            val first = word[bestIdx]
            val second = word[bestIdx + 1]
            val merged = ArrayList<String>(word.size - 1)
            var i = 0
            while (i < word.size) {
                if (i < word.size - 1 && word[i] == first && word[i + 1] == second) {
                    merged.add(first + second)
                    i += 2
                } else {
                    merged.add(word[i])
                    i += 1
                }
            }
            word = merged
        }

        cache[token] = word
        return word
    }

    /** Text -> BPE token IDs (no special tokens, no padding). */
    fun encode(text: String): List<Int> {
        val cleaned = text.replace(Regex("\\s+"), " ").trim().lowercase()
        val ids = ArrayList<Int>()
        for (match in pat.findAll(cleaned)) {
            val bytes = match.value.toByteArray(Charsets.UTF_8)
            val sb = StringBuilder(bytes.size)
            for (b in bytes) sb.append(byteEncoder[b.toInt() and 0xFF])
            for (sub in bpe(sb.toString())) {
                ids.add(vocab[sub] ?: error("BPE token not in vocab: $sub"))
            }
        }
        return ids
    }

    /** Text -> fixed [contextLength] array: [sot, ...ids, eot, 0, 0, ...]. */
    fun tokenize(text: String): LongArray {
        val ids = encode(text)
        val out = LongArray(contextLength) // zero-padded
        out[0] = sot.toLong()
        var i = 1
        for (id in ids) {
            if (i >= contextLength - 1) break
            out[i++] = id.toLong()
        }
        out[i] = eot.toLong()
        return out
    }
}
