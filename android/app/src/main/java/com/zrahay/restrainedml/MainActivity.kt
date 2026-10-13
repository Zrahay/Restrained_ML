package com.zrahay.restrainedml

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.LruCache
import android.util.Size
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.widget.BaseAdapter
import android.widget.EditText
import android.widget.GridView
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.util.concurrent.Executors
import kotlin.math.sqrt

private const val PERMISSION_REQUEST = 1
private const val EMBEDDING_DIM = 512
private const val TOP_K = 30

class MainActivity : Activity() {

    private lateinit var searchBox: EditText
    private lateinit var statusView: TextView
    private lateinit var grid: GridView
    private val gridAdapter = ResultsAdapter()

    private var tokenizer: ClipTokenizer? = null
    private var textModule: Module? = null
    private var store: EmbeddingStore? = null

    private val thumbExecutor = Executors.newFixedThreadPool(3)
    private val thumbCache = LruCache<Long, Bitmap>(120)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildUi()
        if (checkSelfPermission(Manifest.permission.READ_MEDIA_IMAGES)
            == PackageManager.PERMISSION_GRANTED
        ) {
            start()
        } else {
            requestPermissions(arrayOf(Manifest.permission.READ_MEDIA_IMAGES), PERMISSION_REQUEST)
        }
    }

    override fun onRequestPermissionsResult(
        code: Int, permissions: Array<String>, results: IntArray
    ) {
        if (code == PERMISSION_REQUEST && results.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            start()
        } else {
            status("Photo permission denied — nothing to search. Grant it in Settings.")
        }
    }

    private fun buildUi() {
        // Explicit colors: with no declared theme, dark mode gives light text
        // on a light window — an invisible UI. Pin them instead.
        searchBox = EditText(this).apply {
            hint = "Search your photos…  e.g. man in black shirt"
            setTextColor(0xFF111111.toInt())
            setHintTextColor(0xFF888888.toInt())
            isSingleLine = true
            imeOptions = EditorInfo.IME_ACTION_SEARCH
            setOnEditorActionListener { _, action, _ ->
                if (action == EditorInfo.IME_ACTION_SEARCH) {
                    search(text.toString()); true
                } else false
            }
            isEnabled = false
        }
        statusView = TextView(this).apply {
            text = "Loading…"
            setTextColor(0xFF444444.toInt())
            setPadding(16, 8, 16, 8)
        }
        grid = GridView(this).apply {
            numColumns = 3
            horizontalSpacing = 8
            verticalSpacing = 8
            adapter = gridAdapter
            setOnItemClickListener { _, _, pos, _ -> openPhoto(gridAdapter.items[pos].first) }
        }
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(0xFFF5F5F5.toInt())
            setPadding(16, 32, 16, 0)
            addView(searchBox)
            addView(statusView)
            addView(grid, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))
        }
        // targetSdk 35 forces edge-to-edge: pad content below the system bars
        root.setOnApplyWindowInsetsListener { v, insets ->
            val bars = insets.getInsets(android.view.WindowInsets.Type.systemBars())
            v.setPadding(16, bars.top + 16, 16, bars.bottom)
            insets
        }
        setContentView(root)
    }

    private fun start() {
        status("Loading models…")
        Thread {
            val tok = ClipTokenizer(assets.open("clip_tokenizer.json").bufferedReader().readText())
            val text = Module.load(extractAsset("mobileclip_text.pte"))
            val st = EmbeddingStore(File(filesDir, "index.bin"), EMBEDDING_DIM)
            tokenizer = tok; textModule = text; store = st

            runOnUiThread {
                searchBox.isEnabled = true
                status("Ready — ${st.size()} photos indexed. Indexing new ones…")
            }

            // Index in the background; search works on whatever is ready
            val image = Module.load(extractAsset("mobileclip_image.pte"))
            val indexer = GalleryIndexer(this, st, image)
            var lastShown = -1
            indexer.indexAll { done, total ->
                if (done == total || done - lastShown >= 5) {
                    lastShown = done
                    runOnUiThread {
                        status(
                            if (done == total)
                                "Indexed ${st.size()} photos" +
                                    (if (indexer.failed > 0) " (${indexer.failed} failed)" else "")
                            else
                                "Indexing… $done/$total new photos (searchable: ${st.size()})"
                        )
                    }
                }
            }
            if (indexer.galleryIds().isEmpty()) runOnUiThread { status("No photos found in gallery") }
        }.start()
    }

    private fun search(query: String) {
        val tok = tokenizer ?: return
        val text = textModule ?: return
        val st = store ?: return
        if (query.isBlank() || st.size() == 0) return
        Thread {
            val tokens = tok.tokenize(query)
            val input = Tensor.fromBlob(tokens, longArrayOf(1, tokens.size.toLong()))
            val raw = text.forward(EValue.from(input))[0].toTensor().dataAsFloatArray
            var norm = 0f
            for (v in raw) norm += v * v
            norm = sqrt(norm)
            val queryEmb = FloatArray(raw.size) { raw[it] / norm }

            val results = st.search(queryEmb, TOP_K)
            runOnUiThread {
                gridAdapter.items = results
                gridAdapter.notifyDataSetChanged()
                status("\"$query\" — top ${results.size} of ${st.size()} photos")
            }
        }.start()
    }

    private fun openPhoto(id: Long) {
        val uri = Uri.withAppendedPath(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, id.toString())
        startActivity(Intent(Intent.ACTION_VIEW, uri).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION))
    }

    private fun extractAsset(name: String): String {
        val file = File(filesDir, name)
        if (!file.exists() || file.length() == 0L) {
            assets.open(name).use { input -> file.outputStream().use { input.copyTo(it) } }
        }
        return file.absolutePath
    }

    private fun status(text: String) {
        runOnUiThread { statusView.text = text }
    }

    private inner class ResultsAdapter : BaseAdapter() {
        var items: List<Pair<Long, Float>> = emptyList()

        override fun getCount() = items.size
        override fun getItem(position: Int) = items[position]
        override fun getItemId(position: Int) = items[position].first

        override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
            val size = parent.width / 3 - 8
            val view = (convertView as? ImageView) ?: ImageView(this@MainActivity).apply {
                scaleType = ImageView.ScaleType.CENTER_CROP
            }
            view.layoutParams = ViewGroup.LayoutParams(size, size)
            val id = items[position].first
            view.tag = id
            val cached = thumbCache.get(id)
            if (cached != null) {
                view.setImageBitmap(cached)
            } else {
                view.setImageDrawable(null)
                thumbExecutor.execute {
                    try {
                        val uri = Uri.withAppendedPath(
                            MediaStore.Images.Media.EXTERNAL_CONTENT_URI, id.toString()
                        )
                        val thumb = contentResolver.loadThumbnail(uri, Size(256, 256), null)
                        thumbCache.put(id, thumb)
                        runOnUiThread { if (view.tag == id) view.setImageBitmap(thumb) }
                    } catch (_: Exception) {
                    }
                }
            }
            return view
        }
    }
}
