plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.zrahay.restrainedml"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.zrahay.restrainedml"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    androidResources {
        // Keep .pte uncompressed in the APK so the runtime can map it directly
        noCompress += "pte"
    }
}

dependencies {
    implementation("org.pytorch:executorch-android:1.5.1")
}
