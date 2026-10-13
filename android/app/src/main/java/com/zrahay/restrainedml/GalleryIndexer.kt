package com.zrahay.restrainedml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import kotlin.math.sqrt

/**
 * On-phone equivalent of indexer.py: enumerate gallery photos via
 * MediaStore, embed each new one with the image encoder, store the
 * L2-normalized embedding. Skips photos already in the store, logs and
 * counts failures instead of dying on them.
 */
class GalleryIndexer(
    private val context: Context,
    private val store: EmbeddingStore,
    private val imageModule: Module,
) {
    var failed = 0
        private set

    fun galleryIds(): List<Long> {
        val ids = ArrayList<Long>()
        context.contentResolver.query(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            arrayOf(MediaStore.Images.Media._ID),
            null, null,
            "${MediaStore.Images.Media.DATE_ADDED} DESC"
        )?.use { cursor ->
            val col = cursor.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
            while (cursor.moveToNext()) ids.add(cursor.getLong(col))
        }
        return ids
    }

    /** Embeds every gallery photo not yet in the store. Reports progress. */
    fun indexAll(onProgress: (done: Int, total: Int) -> Unit) {
        val pending = galleryIds().filterNot { store.contains(it) }
        val total = pending.size
        for ((i, id) in pending.withIndex()) {
            try {
                val bitmap = loadBitmap(id) ?: throw IllegalStateException("decode failed")
                val data = ImagePreprocessor.toTensorData(bitmap)
                val input = Tensor.fromBlob(
                    data,
                    longArrayOf(1, 3, ImagePreprocessor.SIZE.toLong(), ImagePreprocessor.SIZE.toLong())
                )
                val raw = imageModule.forward(EValue.from(input))[0].toTensor().dataAsFloatArray
                store.add(id, normalize(raw))
            } catch (e: Exception) {
                failed++
                Log.w("RestrainedML", "Skipped photo $id: $e")
            }
            onProgress(i + 1, total)
        }
    }

    /** Decode downsampled so the shorter side stays >= 2x the model input. */
    private fun loadBitmap(id: Long): Bitmap? {
        val uri = Uri.withAppendedPath(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI, id.toString()
        )
        // Bounds probe: with inJustDecodeBounds the decode itself always
        // returns null — success is signaled through bounds.outWidth/Height
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        val probe = context.contentResolver.openInputStream(uri) ?: return null
        probe.use { BitmapFactory.decodeStream(it, null, bounds) }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null

        var sample = 1
        val minSide = minOf(bounds.outWidth, bounds.outHeight)
        while (minSide / (sample * 2) >= ImagePreprocessor.SIZE * 2) sample *= 2

        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        return context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, opts)
        }
    }

    private fun normalize(v: FloatArray): FloatArray {
        var norm = 0f
        for (x in v) norm += x * x
        norm = sqrt(norm)
        return FloatArray(v.size) { v[it] / norm }
    }
}
