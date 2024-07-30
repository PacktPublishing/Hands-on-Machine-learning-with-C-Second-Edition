#pragma once

#include <string_view>
#include <string>

#include "yolo.h"

#include <android_native_app_glue.h>
#include <camera/NdkCameraManager.h>
#include <media/NdkImageReader.h>

#include <opencv2/core/core.hpp>

class ObjectDetector {
public:
    explicit ObjectDetector(android_app *app);

    ~ObjectDetector();

    ObjectDetector(const ObjectDetector &) = delete;

    ObjectDetector &operator=(const ObjectDetector &) = delete;

    ObjectDetector(ObjectDetector &&) = delete;

    ObjectDetector &operator=(ObjectDetector &&) = delete;

    void configure_resources();

    // should be called when permissions were granted and there is suitable camera device
    void allow_camera_session(std::string_view camera_id);

    void release_resources();

    void draw_frame();

private:
    [[nodiscard]] bool is_session_allowed() const;

    void create_camera();

    void delete_camera();

    // camera capture properties are configured here
    void create_image_reader();

    void delete_image_reader();

    void create_session();

    void delete_session();

    void process_image(ANativeWindow_Buffer *buf, AImage *image);

private:
    android_app *android_app_{nullptr};
    std::string camera_id_;
    ACameraManager *camera_mgr_{nullptr};
    ACameraDevice *camera_device_{nullptr};
    AImageReader *image_reader_{nullptr};

    ACaptureSessionOutput *session_output_{nullptr};
    ACaptureSessionOutputContainer *output_container_{nullptr};
    ACameraOutputTarget *output_target_{nullptr};
    ACaptureRequest *capture_request_{nullptr};
    ACameraCaptureSession *capture_session_{nullptr};

    int32_t width_{800};
    int32_t height_{600};
    int32_t orientation_{0};

    cv::Mat rgba_img_;
    std::shared_ptr<YOLO> yolo_;
};
