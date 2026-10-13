package com.zrahay.restrainedml

import android.app.Activity
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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        statusView = TextView(this).apply {
            text = "Loading text encoder…"
            textSize = 16f
            gravity = Gravity.START
            setPadding(32, 64, 32, 64)
        }
        setContentView(ScrollView(this).apply { addView(statusView) })

        Thread { runParityCheck() }.start()
    }

    private fun runParityCheck() {
        try {
            val report = StringBuilder()

            // Module.load needs a real file path, so copy the asset out once
            val pteFile = File(filesDir, "mobileclip_text.pte")
            if (!pteFile.exists() || pteFile.length() == 0L) {
                assets.open("mobileclip_text.pte").use { input ->
                    pteFile.outputStream().use { input.copyTo(it) }
                }
            }

            var module: Module
            val loadMs = measureTimeMillis { module = Module.load(pteFile.absolutePath) }
            report.appendLine("Text encoder loaded in ${loadMs} ms")
            post(report)

            val golden = JSONObject(
                assets.open("golden_text.json").bufferedReader().readText()
            ).getJSONArray("cases")

            var failures = 0
            for (i in 0 until golden.length()) {
                val case = golden.getJSONObject(i)
                val query = case.getString("query")

                val tokensJson = case.getJSONArray("tokens")
                val tokens = LongArray(tokensJson.length()) { tokensJson.getLong(it) }

                val expectedJson = case.getJSONArray("embedding")
                val expected = FloatArray(expectedJson.length()) {
                    expectedJson.getDouble(it).toFloat()
                }

                val input = Tensor.fromBlob(tokens, longArrayOf(1, tokens.size.toLong()))
                var output: FloatArray
                val inferMs = measureTimeMillis {
                    output = module.forward(EValue.from(input))[0]
                        .toTensor().dataAsFloatArray
                }

                val cos = cosine(expected, output)
                val ok = cos >= 0.999f
                if (!ok) failures++
                val line = "\"$query\": cosine=%.6f  %d ms  %s"
                    .format(cos, inferMs, if (ok) "OK" else "FAIL")
                Log.i(TAG, line)
                report.appendLine(line)
                post(report)
            }

            val verdict = if (failures == 0) {
                "\nON-DEVICE PARITY PASSED — the model runs correctly on this phone"
            } else {
                "\nON-DEVICE PARITY FAILED: $failures case(s)"
            }
            Log.i(TAG, verdict.trim())
            report.appendLine(verdict)
            post(report)
        } catch (e: Throwable) {
            Log.e(TAG, "Parity check crashed", e)
            post(StringBuilder("ERROR: $e"))
        }
    }

    private fun cosine(a: FloatArray, b: FloatArray): Float {
        var dot = 0f; var na = 0f; var nb = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
        }
        return dot / (sqrt(na) * sqrt(nb))
    }

    private fun post(report: StringBuilder) {
        runOnUiThread { statusView.text = report.toString() }
    }
}
