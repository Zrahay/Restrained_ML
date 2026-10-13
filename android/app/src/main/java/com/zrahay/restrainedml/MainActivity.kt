package com.zrahay.restrainedml

import android.app.Activity
import android.os.Bundle
import android.view.Gravity
import android.widget.TextView

class MainActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val view = TextView(this).apply {
            text = "Restrained ML — hello from the A34!"
            textSize = 20f
            gravity = Gravity.CENTER
        }
        setContentView(view)
    }
}
