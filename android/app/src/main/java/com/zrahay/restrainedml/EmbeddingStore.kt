package com.zrahay.restrainedml

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream

/**
 * On-phone equivalent of embeddings.pt: MediaStore photo ID -> L2-normalized
 * embedding, held in memory and appended to a binary file so indexing
 * survives app restarts and resumes incrementally.
 *
 * File format: [magic int][dim int] then per record: [id long][dim floats].
 */
class EmbeddingStore(private val file: File, private val dim: Int) {

    private val ids = ArrayList<Long>()
    private val embeddings = ArrayList<FloatArray>()
    private val known = HashSet<Long>()

    companion object {
        private const val MAGIC = 0x524D4C31 // "RML1"
    }

    init {
        if (file.exists() && file.length() > 8) {
            DataInputStream(file.inputStream().buffered()).use { input ->
                val magic = input.readInt()
                val fileDim = input.readInt()
                if (magic == MAGIC && fileDim == dim) {
                    val recordBytes = 8 + 4 * dim
                    val records = ((file.length() - 8) / recordBytes).toInt()
                    repeat(records) {
                        val id = input.readLong()
                        val emb = FloatArray(dim) { input.readFloat() }
                        ids.add(id)
                        embeddings.add(emb)
                        known.add(id)
                    }
                }
            }
        }
    }

    @Synchronized fun size(): Int = ids.size

    @Synchronized fun contains(id: Long): Boolean = id in known

    @Synchronized
    fun add(id: Long, embedding: FloatArray) {
        require(embedding.size == dim)
        val newFile = !file.exists() || file.length() < 8
        DataOutputStream(FileOutputStream(file, !newFile).buffered()).use { out ->
            if (newFile) {
                out.writeInt(MAGIC)
                out.writeInt(dim)
            }
            out.writeLong(id)
            for (v in embedding) out.writeFloat(v)
        }
        ids.add(id)
        embeddings.add(embedding)
        known.add(id)
    }

    /** Top-k photo IDs by dot product against a normalized query embedding. */
    @Synchronized
    fun search(query: FloatArray, k: Int): List<Pair<Long, Float>> {
        val scored = ArrayList<Pair<Long, Float>>(ids.size)
        for (i in ids.indices) {
            val emb = embeddings[i]
            var dot = 0f
            for (j in 0 until dim) dot += query[j] * emb[j]
            scored.add(ids[i] to dot)
        }
        scored.sortByDescending { it.second }
        return scored.take(k)
    }
}
