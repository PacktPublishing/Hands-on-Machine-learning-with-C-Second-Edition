#include "log.h"
#include "detector.h"

#include <jni.h>
#include <string>

#include <android_native_app_glue.h>
#include <android/native_window.h>
#include <camera/NdkCameraDevice.h>
#include <camera/NdkCameraError.h>
#include <camera/NdkCameraManager.h>
#include <camera/NdkCameraMetadataTags.h>
#include <android/looper.h>

std::shared_ptr<ObjectDetector> object_detector_;

extern "C" JNIEXPORT void JNICALL
Java_com_example_objectdetection_MainActivity_initObjectDetection(
        JNIEnv *env,
        jobject /* this */,
        jstring camId) {
    auto camera_id = env->GetStringUTFChars(camId, nullptr);
    LOGI("Camera ID: %s", camera_id);
    if (object_detector_) {
        object_detector_->allow_camera_session(camera_id);
        object_detector_->configure_resources();
    } else
        LOGE("Object Detector object is missed!");
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_objectdetection_MainActivity_stopObjectDetection(
        JNIEnv *,
        jobject /* this */) {
    object_detector_->release_resources();
}

static void ProcessAndroidCmd(struct android_app */*app*/, int32_t cmd) {
    if (object_detector_) {
        switch (cmd) {
            case APP_CMD_INIT_WINDOW:
                object_detector_->configure_resources();
                break;
            case APP_CMD_TERM_WINDOW:
                object_detector_->release_resources();
                break;
        }
    }
}

extern "C" void android_main(struct android_app *app) {
    LOGI("Native entry point");
    object_detector_ = std::make_shared<ObjectDetector>(app);
    app->onAppCmd = ProcessAndroidCmd;

    while (!app->destroyRequested) {
        struct android_poll_source *source = nullptr;
        auto result = ALooper_pollOnce(0, nullptr, nullptr, (void **) &source);
        ASSERT(result != ALOOPER_POLL_ERROR, "ALooper_pollOnce returned an error");
        if (source != nullptr) {
            source->process(app, source);
        }
        if (object_detector_)
            object_detector_->draw_frame();
    }

    // application is closing ...
    object_detector_.reset();
}