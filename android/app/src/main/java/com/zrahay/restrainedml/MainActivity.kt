package com.zrahay.restrainedml

import android.app.Activity
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.widget.ScrollView
import android.widget.TextView
import org.json.JSONObject
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import kotlin.math.sqrt
import kotlin.system.measureTimeMillis

private const val TAG = "RestrainedML"

class MainActivity : Activity() {

    private lateinit var statusView: TextView
    private val report = StringBuilder()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        statusView = TextView(this).apply {
            text = "Running Phase D checks…"
            textSize = 14f
            gravity = Gravity.START
            setPadding(32, 64, 32, 64)
        }
        setContentView(ScrollView(this).apply { addView(statusView) })

        Thread { runPhaseD() }.start()
    }

    private fun runPhaseD() {
        try {
            // --- Suite 1: tokenizer IDs vs Python (50 cases) ---
            line("Loading tokenizer…")
            val tokenizer: ClipTokenizer
            val tokLoadMs = measureTimeMillis {
                tokenizer = ClipTokenizer(readAsset("clip_tokenizer.json"))
            }
            line("Tokenizer loaded in $tokLoadMs ms")

            val tokGolden = JSONObject(readAsset("tokenizer_golden.json"))
                .getJSONArray("cases")
            var tokFails = 0
            for (i in 0 until tokGolden.length()) {
                val case = tokGolden.getJSONObject(i)
                val query = case.getString("query")
                val expectedJson = case.getJSONArray("tokens")
                val expected = LongArray(expectedJson.length()) { expectedJson.getLong(it) }
                val actual = tokenizer.tokenize(query)
                if (!expected.contentEquals(actual)) {
                    tokFails++
                    line("TOKENIZER MISMATCH: \"$query\"")
                    line("  expected: ${expected.take(12)}…")
                    line("  actual:   ${actual.take(12)}…")
                }
            }
            line("Suite 1 tokenizer: ${tokGolden.length() - tokFails}/${tokGolden.length()} exact matches")

            // --- Suite 2: string -> Kotlin tokenizer -> text encoder (end to end) ---
            val textModule = loadModule("mobileclip_text.pte")
            line("Text encoder loaded")
            val textGolden = JSONObject(readAsset("golden_text.json")).getJSONArray("cases")
            var textFails = 0
            for (i in 0 until textGolden.length()) {
                val case = textGolden.getJSONObject(i)
                val query = case.getString("query")
                val expected = floatArray(case, "embedding")
                val tokens = tokenizer.tokenize(query)
                val input = Tensor.fromBlob(tokens, longArrayOf(1, tokens.size.toLong()))
                val output = textModule.forward(EValue.from(input))[0]
                    .toTensor().dataAsFloatArray
                val cos = cosine(expected, output)
                val ok = cos >= 0.999f
                if (!ok) textFails++
                line("text \"$query\": cosine=%.6f %s".format(cos, if (ok) "OK" else "FAIL"))
            }

            // --- Suite 3: jpg -> Kotlin preprocess -> image encoder (end to end) ---
            val imageModule = loadModule("mobileclip_image.pte")
            line("Image encoder loaded")
            val imgGolden = JSONObject(readAsset("golden_images.json")).getJSONArray("cases")
            var imgFails = 0
            for (i in 0 until imgGolden.length()) {
                val case = imgGolden.getJSONObject(i)
                val file = case.getString("file")
                val expected = floatArray(case, "embedding")
                val bitmap = assets.open(file).use { BitmapFactory.decodeStream(it) }
                val data = ImagePreprocessor.toTensorData(bitmap)
                val input = Tensor.fromBlob(
                    data, longArrayOf(1, 3, ImagePreprocessor.SIZE.toLong(), ImagePreprocessor.SIZE.toLong())
                )
                var output: FloatArray
                val ms = measureTimeMillis {
                    output = imageModule.forward(EValue.from(input))[0]
                        .toTensor().dataAsFloatArray
                }
                val cos = cosine(expected, output)
                val ok = cos >= 0.99f // preprocessing is approximate (resize filters differ)
                if (!ok) imgFails++
                line("image $file: cosine=%.6f  %d ms %s".format(cos, ms, if (ok) "OK" else "FAIL"))
            }

            val total = tokFails + textFails + imgFails
            line(
                if (total == 0) "\nPHASE D PASSED — tokenizer + preprocessing match desktop"
                else "\nPHASE D FAILED: $tokFails tokenizer, $textFails text, $imgFails image"
            )
        } catch (e: Throwable) {
            Log.e(TAG, "Phase D crashed", e)
            line("ERROR: $e")
        }
    }

    private fun readAsset(name: String): String =
        assets.open(name).bufferedReader().readText()

    private fun loadModule(assetName: String): Module {
        val file = File(filesDir, assetName)
        if (!file.exists() || file.length() == 0L) {
            assets.open(assetName).use { input ->
                file.outputStream().use { input.copyTo(it) }
            }
        }
        return Module.load(file.absolutePath)
    }

    private fun floatArray(obj: JSONObject, key: String): FloatArray {
        val arr = obj.getJSONArray(key)
        return FloatArray(arr.length()) { arr.getDouble(it).toFloat() }
    }

    private fun cosine(a: FloatArray, b: FloatArray): Float {
        var dot = 0f; var na = 0f; var nb = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
        }
        return dot / (sqrt(na) * sqrt(nb))
    }

    private fun line(text: String) {
        Log.i(TAG, text)
        report.appendLine(text)
        runOnUiThread { statusView.text = report.toString() }
    }
}
