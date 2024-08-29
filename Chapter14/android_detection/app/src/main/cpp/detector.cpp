#include "detector.h"
#include "log.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <android/native_window.h>

ObjectDetector::ObjectDetector(android_app *app) : android_app_(app) {
    yolo_ = std::make_shared<YOLO>(app->activity->assetManager);
}

ObjectDetector::~ObjectDetector() {
    release_resources();
    LOGI("Object Detector was destroyed!");
}

void ObjectDetector::release_resources() {
    delete_camera();
    delete_image_reader();
    delete_session();
}

void ObjectDetector::configure_resources() {
    if (!is_session_allowed() || !android_app_ || !android_app_->window) {
        LOGE("Can't configure output window!");
        return;
    }

    if (!camera_device_)
        create_camera();

    // configure output window size and format
    ACameraMetadata *metadata_obj{nullptr};
    ACameraManager_getCameraCharacteristics(camera_mgr_, camera_id_.c_str(), &metadata_obj);

    ACameraMetadata_const_entry entry;
    ACameraMetadata_getConstEntry(metadata_obj,
                                  ACAMERA_SENSOR_ORIENTATION, &entry);

    orientation_ = entry.data.i32[0];

    bool is_horizontal = orientation_ == 0 || orientation_ == 270;
    auto out_width = is_horizontal ? width_ : height_;
    auto out_height = is_horizontal ? height_ : width_;

    auto status = ANativeWindow_setBuffersGeometry(
            android_app_->window, out_width, out_height, WINDOW_FORMAT_RGBA_8888);
    if (status < 0) {
        LOGE("Can't configure output window, failed to set output format!");
        return;
    }

    if (!image_reader_ && !session_output_) {
        create_image_reader();
        create_session();
    }
}

void ObjectDetector::allow_camera_session(std::string_view camera_id) {
    camera_id_ = camera_id;
}

bool ObjectDetector::is_session_allowed() const {
    return !camera_id_.empty();
}

namespace {
    void onDisconnected([[maybe_unused]]void *context, [[maybe_unused]]ACameraDevice *device) {
        LOGI("Camera onDisconnected");
    }

    void onError([[maybe_unused]]void *context, [[maybe_unused]]ACameraDevice *device, int error) {
        LOGE("Camera error %d", error);
    }

    ACameraDevice_stateCallbacks camera_device_callbacks = {
            .context = nullptr,
            .onDisconnected = onDisconnected,
            .onError = onError,
    };
}

void ObjectDetector::create_camera() {
    camera_mgr_ = ACameraManager_create();
    ASSERT(camera_mgr_, "Failed to create Camera Manager");

    ACameraManager_openCamera(camera_mgr_, camera_id_.c_str(), &camera_device_callbacks,
                              &camera_device_);
    ASSERT(camera_device_, "Failed to open camera");
}

void ObjectDetector::delete_camera() {
    if (camera_device_) {
        ACameraDevice_close(camera_device_);
        camera_device_ = nullptr;
    }
    if (camera_mgr_) {
        ACameraManager_delete(camera_mgr_);
        camera_mgr_ = nullptr;
    }
}

void ObjectDetector::create_image_reader() {
    constexpr int32_t MAX_BUF_COUNT = 4;
    auto status = AImageReader_new(width_, height_, AIMAGE_FORMAT_YUV_420_888,
                                   MAX_BUF_COUNT, &image_reader_);
    ASSERT(image_reader_ && status == AMEDIA_OK, "Failed to create AImageReader");
}

void ObjectDetector::delete_image_reader() {
    if (image_reader_) {
        AImageReader_delete(image_reader_);
        image_reader_ = nullptr;
    }
}

namespace {
    void
    onSessionActive([[maybe_unused]]void *context, [[maybe_unused]]ACameraCaptureSession *session) {
        LOGI("onSessionActive()");
    }

    void
    onSessionReady([[maybe_unused]]void *context, [[maybe_unused]]ACameraCaptureSession *session) {
        LOGI("onSessionReady()");
    }

    void
    onSessionClosed([[maybe_unused]]void *context, [[maybe_unused]]ACameraCaptureSession *session) {
        LOGI("onSessionClosed()");
    }

    ACameraCaptureSession_stateCallbacks session_callbacks{
            .context = nullptr,
            .onClosed = onSessionClosed,
            .onReady = onSessionReady,
            .onActive = onSessionActive
    };
}

void ObjectDetector::create_session() {
    ANativeWindow *output_native_window;
    auto status = AImageReader_getWindow(image_reader_, &output_native_window);
    ASSERT(status == AMEDIA_OK, "Could not get ANativeWindow");

    auto cam_status = ACaptureSessionOutputContainer_create(&output_container_);
    ASSERT(cam_status == ACAMERA_OK, "Could not create ACaptureSessionOutputContainer");

    ANativeWindow_acquire(output_native_window);
    cam_status = ACaptureSessionOutput_create(output_native_window, &session_output_);
    ASSERT(cam_status == ACAMERA_OK, "Could not create ACaptureSessionOutput");

    cam_status = ACaptureSessionOutputContainer_add(output_container_, session_output_);
    ASSERT(cam_status == ACAMERA_OK, "Could add ACaptureSessionOutput to a container");

    cam_status = ACameraOutputTarget_create(output_native_window, &output_target_);
    ASSERT(cam_status == ACAMERA_OK, "Could not create ACameraOutputTarget");

    cam_status = ACameraDevice_createCaptureRequest(camera_device_,
                                                    TEMPLATE_PREVIEW, &capture_request_);
    ASSERT(cam_status == ACAMERA_OK, "Could not create ACaptureRequest");

    cam_status = ACaptureRequest_addTarget(capture_request_, output_target_);
    ASSERT(cam_status == ACAMERA_OK, "Could not add target for a ACaptureRequest");

    cam_status = ACameraDevice_createCaptureSession(camera_device_,
                                                    output_container_,
                                                    &session_callbacks,
                                                    &capture_session_);
    ASSERT(cam_status == ACAMERA_OK, "Could not create ACaptureSession");

    // Start capturing continuously
    cam_status = ACameraCaptureSession_setRepeatingRequest(capture_session_,
                                                                nullptr,
                                                                1,
                                                                &capture_request_,
                                                                nullptr);
    ASSERT(cam_status == ACAMERA_OK, "Could not start capturing session");
}

void ObjectDetector::delete_session() {
    if (capture_session_) {
        ACameraCaptureSession_stopRepeating(capture_session_);
        capture_session_ = nullptr;
    }
    if (output_container_) {
        ACaptureSessionOutputContainer_free(output_container_);
        output_container_ = nullptr;
    }
    if (session_output_) {
        ACaptureSessionOutput_free(session_output_);
        session_output_ = nullptr;
    }
    if (output_target_) {
        ACameraOutputTarget_free(output_target_);
        output_target_ = nullptr;
    }
    if (capture_request_) {
        ACaptureRequest_free(capture_request_);
        capture_request_ = nullptr;
    }
    if (capture_session_) {
        ACameraCaptureSession_close(capture_session_);
        capture_session_ = nullptr;
    }
}


void ObjectDetector::draw_frame() {
    if (image_reader_ == nullptr)
        return;

    AImage *image = nullptr;
    auto status = AImageReader_acquireNextImage(image_reader_, &image);
    if (status != AMEDIA_OK) {
        return;
    }

    ANativeWindow_acquire(android_app_->window);
    ANativeWindow_Buffer buf;
    if (ANativeWindow_lock(android_app_->window, &buf, nullptr) < 0) {
        AImage_delete(image);
        return;
    }

    process_image(&buf, image);
    AImage_delete(image);
    ANativeWindow_unlockAndPost(android_app_->window);
    ANativeWindow_release(android_app_->window);
}

void ObjectDetector::process_image(ANativeWindow_Buffer *buf, AImage *image) {
    ASSERT(buf->format == WINDOW_FORMAT_RGBX_8888 ||
           buf->format == WINDOW_FORMAT_RGBA_8888,
           "Not supported buffer format");

    int32_t src_format = -1;
    AImage_getFormat(image, &src_format);
    ASSERT(AIMAGE_FORMAT_YUV_420_888 == src_format, "Unsupported image format for displaying");
    int32_t num_src_planes = 0;
    AImage_getNumberOfPlanes(image, &num_src_planes);
    ASSERT(num_src_planes == 3, "Image for display has unsupported number of planes");

    int32_t src_height;
    AImage_getHeight(image, &src_height);

    int32_t src_width;
    AImage_getWidth(image, &src_width);

    int32_t y_stride{0};
    AImage_getPlaneRowStride(image, 0, &y_stride);
    int32_t uv_stride1{0};
    AImage_getPlaneRowStride(image, 1, &uv_stride1);
    int32_t uv_stride2{0};
    AImage_getPlaneRowStride(image, 1, &uv_stride2);

    uint8_t *y_pixel{nullptr}, *uv_pixel1{nullptr}, *uv_pixel2{nullptr};
    int32_t y_len{0}, uv_len1{0}, uv_len2{0};
    AImage_getPlaneData(image, 0, &y_pixel, &y_len);
    AImage_getPlaneData(image, 1, &uv_pixel1, &uv_len1);
    AImage_getPlaneData(image, 2, &uv_pixel2, &uv_len2);
    int32_t uv_pixel_stride{0};
    AImage_getPlanePixelStride(image, 1, &uv_pixel_stride);

    // if chroma channels are interleaved
    if (orientation_ == 90 && uv_pixel_stride == 2) {
        cv::Size actual_size(src_width, src_height);
        cv::Size half_size(src_width / 2, src_height / 2);

        cv::Mat y(actual_size, CV_8UC1, y_pixel, y_stride);
        cv::Mat uv1(half_size, CV_8UC2, uv_pixel1, uv_stride1);
        cv::Mat uv2(half_size, CV_8UC2, uv_pixel2, uv_stride2);

        long addr_diff = uv2.data - uv1.data;
        if (addr_diff > 0) {
            cvtColorTwoPlane(y, uv1, rgba_img_, cv::COLOR_YUV2RGBA_NV12);
        } else {
            cvtColorTwoPlane(y, uv2, rgba_img_, cv::COLOR_YUV2RGBA_NV21);
        }
        cv::rotate(rgba_img_, rgba_img_, cv::ROTATE_90_CLOCKWISE);

        auto results = yolo_->detect(rgba_img_);

        for (auto &result: results) {
            int thickness = 2;
            rectangle(rgba_img_, result.rect.tl(), result.rect.br(),
                      cv::Scalar(255, 0, 0, 255),
                      thickness, cv::LINE_4);

            cv::putText(rgba_img_,
                        result.class_name,
                        result.rect.tl(),
                        cv::FONT_HERSHEY_DUPLEX,
                        1.0,
                        CV_RGB(0, 255, 0),
                        2);
        }
        cv::Mat buffer_mat(src_width, src_height, CV_8UC4, buf->bits, buf->stride * 4);
        rgba_img_.copyTo(buffer_mat);
    }
    // other cases should be processed in different way
}
