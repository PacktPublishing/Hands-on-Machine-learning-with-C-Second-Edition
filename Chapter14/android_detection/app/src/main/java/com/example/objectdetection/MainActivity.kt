package com.example.objectdetection

import android.app.NativeActivity
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY
import android.hardware.camera2.CameraMetadata.LENS_FACING_BACK
import android.os.Bundle
import android.view.WindowManager
import android.widget.Toast
import androidx.core.app.ActivityCompat

class MainActivity : NativeActivity(), ActivityCompat.OnRequestPermissionsResultCallback {

//    private lateinit var camId: String
//
//    override fun onCreate(savedInstanceState: Bundle?) {
//        super.onCreate(savedInstanceState)
//
//    }

    override fun onResume() {
        super.onResume()

        // Ask for camera permission if necessary
        val cameraPermission = android.Manifest.permission.CAMERA
        if (checkSelfPermission(cameraPermission) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(arrayOf(cameraPermission), CAM_PERMISSION_CODE)
        } else {
            val camId = getCameraBackCameraId()
            if (camId.isEmpty()) {
                Toast.makeText(
                    this, "Camera probably won't work on this device!",
                    Toast.LENGTH_LONG
                ).show()
                finish()
            }

            initObjectDetection(camId)
        }
    }

    override fun onPause() {
        super.onPause()
        stopObjectDetection()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAM_PERMISSION_CODE
            && grantResults[0] != PackageManager.PERMISSION_GRANTED
        ) {
            Toast.makeText(this, "This app requires camera permission", Toast.LENGTH_SHORT).show()
            finish()
        }
    }

    private fun getCameraBackCameraId(): String {
        val camManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager

        for (camId in camManager.cameraIdList) {
            val characteristics = camManager.getCameraCharacteristics(camId)
            val hwLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL)
            val facing = characteristics.get(CameraCharacteristics.LENS_FACING)
            if (hwLevel != INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY && facing == LENS_FACING_BACK) {
                return camId
            }
        }

        return ""
    }


    private external fun initObjectDetection(camId: String)
    private external fun stopObjectDetection()

    companion object {
        const val CAM_PERMISSION_CODE = 1

        init {
            System.loadLibrary("object-detection")
        }
    }
}