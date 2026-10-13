package com.zrahay.restrainedml

import android.graphics.Bitmap

/**
 * Kotlin port of the MobileCLIP preprocess pipeline:
 *   Resize(256, bilinear, antialias) -> CenterCrop(256) -> ToTensor()
 *
 * Output layout matches ToTensor: CHW float array, RGB, values in [0, 1].
 * No mean/std normalization — MobileCLIP takes raw 0-1 pixels.
 */
object ImagePreprocessor {

    const val SIZE = 256

    fun toTensorData(bitmap: Bitmap): FloatArray {
        val w = bitmap.width
        val h = bitmap.height

        // Shorter side -> SIZE, other side scaled proportionally
        // (truncation matches torchvision's int() rounding)
        val newW: Int
        val newH: Int
        if (w <= h) {
            newW = SIZE
            newH = (h.toLong() * SIZE / w).toInt()
        } else {
            newW = (w.toLong() * SIZE / h).toInt()
            newH = SIZE
        }
        val scaled = Bitmap.createScaledBitmap(bitmap, newW, newH, true)

        val left = (newW - SIZE) / 2
        val top = (newH - SIZE) / 2
        val cropped = Bitmap.createBitmap(scaled, left, top, SIZE, SIZE)

        val pixels = IntArray(SIZE * SIZE)
        cropped.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)

        val plane = SIZE * SIZE
        val out = FloatArray(3 * plane)
        for (i in pixels.indices) {
            val p = pixels[i]
            out[i] = ((p shr 16) and 0xFF) / 255f           // R
            out[plane + i] = ((p shr 8) and 0xFF) / 255f    // G
            out[2 * plane + i] = (p and 0xFF) / 255f        // B
        }
        return out
    }
}
