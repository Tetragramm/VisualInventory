#pragma once

namespace dist
{
    cv::Mat cameraMatrix;
    cv::Mat distortion;
    cv::Mat map1, map2;
    void init(int cols, int rows)
    {
        using namespace cv;
        cameraMatrix.create(3, 3, CV_64F);
        setIdentity(cameraMatrix);
        cameraMatrix.at<double>(0, 0) = 1.0834372011947185e3;
        cameraMatrix.at<double>(0, 1) = 0;
        cameraMatrix.at<double>(0, 2) = 6.4509475040540178e2;
        cameraMatrix.at<double>(1, 0) = 0;
        cameraMatrix.at<double>(1, 1) = 1.0873977971533066e3;
        cameraMatrix.at<double>(1, 2) = 3.8727637732726322e2;

        distortion.create(5, 1, CV_64F);
        distortion.at<double>(0) = 1.4984357248456412e-2;
        distortion.at<double>(1) = 2.1695557983590708e-1;
        distortion.at<double>(2) = 4.3339098266878149e-3;
        distortion.at<double>(3) = -3.5623639211147516e-3;
        distortion.at<double>(4) = -9.3552831102265499e-1;

        initUndistortRectifyMap(cameraMatrix, distortion, Mat(),
            getOptimalNewCameraMatrix(cameraMatrix, distortion, cv::Size(cols, rows), 1),
            cv::Size(cols, rows), CV_16SC2, map1, map2);
    }

    void undistort(cv::Mat& src, cv::Mat& dst)
    {
        using namespace cv;
        remap(src, dst, map1, map2, INTER_LINEAR);
    }
}
