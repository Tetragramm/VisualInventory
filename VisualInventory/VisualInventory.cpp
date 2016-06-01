// VisualInventory.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <windows.h>
#include "opencv2\adas.hpp"
#include "opencv2\aruco.hpp"
#include "opencv2\aruco\charuco.hpp"
#include "opencv2\bgsegm.hpp"
#include "opencv2\bioinspired.hpp"
#include "opencv2\calib3d.hpp"
#include "opencv2\core.hpp"
#include "opencv2\face.hpp"
#include "opencv2\features2d.hpp"
#include "opencv2\flann.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\imgcodecs.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\mapping3d.hpp"
#include "opencv2\ml.hpp"
#include "opencv2\objdetect.hpp"
#include "opencv2\opencv.hpp"
#include "opencv2\opencv_modules.hpp"
#include "opencv2\optflow.hpp"
#include "opencv2\photo.hpp"
#include "opencv2\rgbd.hpp"
#include "opencv2\saliency.hpp"
#include "opencv2\shape.hpp"
#include "opencv2\stitching.hpp"
#include "opencv2\superres.hpp"
#include "opencv2\surface_matching.hpp"
#include "opencv2\text.hpp"
#include "opencv2\tracking.hpp"
#include "opencv2\video.hpp"
#include "opencv2\videoio.hpp"
#include "opencv2\videostab.hpp"
#include "opencv2\ximgproc.hpp"
#include "opencv2\xobjdetect.hpp"
#include "opencv2\xphoto.hpp"

#undef sign
#include "CameraDistortion.h"

int thresh = 150, N = 3;

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
static void findSquares(const cv::Mat& image, std::vector<std::vector<cv::Point> >& squares)
{
    using namespace cv;
    squares.clear();

    Mat pyr, timg, temp(image.size(), CV_8U), gray, gray0;

    // down-scale and upscale the image to filter out the noise
    pyrDown(image, pyr, Size(image.cols / 2, image.rows / 2));
    pyrUp(pyr, timg, image.size());
    std::vector<std::vector<Point> > contours;

    // find squares in every color plane of the image
    for (int c = 0; c < 3; c++)
    {
        temp.create(image.size(), CV_8U);
        int ch[] = { c, 0 };
        mixChannels(&timg, 1, &gray0, 1, ch, 1);
        //imshow("gray0", gray0);

        // try several threshold levels
        for (int l = 0; l < N; l++)
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if (l == 0)
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1, -1));
                //imshow("canny", gray);
                //waitKey();
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l + 1) * 255 / N;
            }

            // find contours and store them all as a list
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            std::vector<Point> approx;

            // test each contour
            for (size_t i = 0; i < contours.size(); i++)
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if (approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > 1000 &&
                    isContourConvex(Mat(approx)))
                {
                    double maxCosine = 0;

                    for (int j = 2; j < 5; j++)
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if (maxCosine < 0.1)
                        squares.push_back(approx);
                }
            }
        }
    }
}

// the function draws all the squares in the image
static void drawSquares(cv::Mat& image, const std::vector<std::vector<cv::Point> >& squares)
{
    using namespace cv;
    for (size_t i = 0; i < squares.size(); i++)
    {
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        polylines(image, &p, &n, 1, true, Scalar(0, 255, 0), 3, LINE_AA);
    }

    imshow("Squares", image);
}

cv::Mat video2Panorama(cv::String filename)
{
    using namespace cv;
    //Create vectors and Mats
    Mat image;
    std::vector<Mat> images;
    std::vector<Mat> transforms;
    Mat panorama;
    Mat affine(2, 3, CV_64F);

    //Create ORB descriptors, memory, and matcher
    std::vector<KeyPoint> pKps, iKps;
    Mat pDesc, iDesc;
    Ptr<ORB> pORB = ORB::create(10000);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    std::vector<DMatch> matches;

    //Init Video Reader
    VideoCapture vidIn(filename);
    //Error Checking
    if (!vidIn.isOpened())
        return Mat();

    //Init First Image
    vidIn.read(image);

    pORB->detectAndCompute(image, noArray(), pKps, pDesc);

    affine.setTo(0);
    affine.at<double>(0, 1) = -1;
    affine.at<double>(1, 0) = 1;
    affine.at<double>(0, 2) = 720;

    images.push_back(image.clone());
    transforms.push_back(affine.clone());


    while (vidIn.read(image))
    {
        //Detect Keypoints and find matches
        pORB->detectAndCompute(image, noArray(), iKps, iDesc);
        matcher->match(iDesc, pDesc, matches);

        double max_dist = 0; double min_dist = 100;
        //-- Quick calculation of max and min distances between keypoints
        for (int i = 0; i < iDesc.rows; i++)
        {
            double dist = matches[i].distance;
            if (dist < min_dist)
                min_dist = dist;
            if (dist > max_dist)
                max_dist = dist;
        }

        //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
        std::vector< Point2f > pPts, iPts;
        for (int i = 0; i < iDesc.rows; i++)
        {
            if (matches[i].distance < 5 * min_dist)
            {
                pPts.push_back(pKps[matches[i].trainIdx].pt);
                iPts.push_back(iKps[matches[i].queryIdx].pt);
            }
        }

        //If there are enough points
        if (pPts.size() > 20)
        {
            //We want to match to the warped locations
            transform(pPts, pPts, affine);

            //Estimate transform from current image to warped previous
            Mat temp = estimateRigidTransform(iPts, pPts, false);
            //If it worked
            if (!temp.empty())
            {
                //Save the image, transform, and set the new set of keypoints and descriptors
                images.push_back(image.clone());
                transforms.push_back(temp.clone());
                pKps = iKps;
                iDesc.copyTo(pDesc);
                temp.copyTo(affine);
            }
        }
    }

    //Find the size of the panorama by transforming each of the four corners of the image and finding the bounds
    std::vector<Point2f> testPoints, transformedPoints;
    testPoints.push_back(Point2f(0, 0));
    testPoints.push_back(Point2f(images[1].cols, 0));
    testPoints.push_back(Point2f(0, images[1].rows));
    testPoints.push_back(Point2f(images[1].cols, images[1].rows));
    Size panoSize(0, 0), orig(10000, 10000);
    for (int i = 0; i < images.size(); ++i)
    {
        transform(testPoints, transformedPoints, transforms[i]);
        panoSize.width = MAX(cvCeil(transformedPoints[0].x), panoSize.width);
        panoSize.width = MAX(cvCeil(transformedPoints[1].x), panoSize.width);
        panoSize.width = MAX(cvCeil(transformedPoints[2].x), panoSize.width);
        panoSize.width = MAX(cvCeil(transformedPoints[3].x), panoSize.width);
        panoSize.height = MAX(cvCeil(transformedPoints[0].y), panoSize.height);
        panoSize.height = MAX(cvCeil(transformedPoints[1].y), panoSize.height);
        panoSize.height = MAX(cvCeil(transformedPoints[2].y), panoSize.height);
        panoSize.height = MAX(cvCeil(transformedPoints[3].y), panoSize.height);

        orig.width = MIN(cvFloor(transformedPoints[0].x), orig.width);
        orig.width = MIN(cvFloor(transformedPoints[1].x), orig.width);
        orig.width = MIN(cvFloor(transformedPoints[2].x), orig.width);
        orig.width = MIN(cvFloor(transformedPoints[3].x), orig.width);
        orig.height = MIN(cvFloor(transformedPoints[0].y), orig.height);
        orig.height = MIN(cvFloor(transformedPoints[1].y), orig.height);
        orig.height = MIN(cvFloor(transformedPoints[2].y), orig.height);
        orig.height = MIN(cvFloor(transformedPoints[3].y), orig.height);
    }

    panoSize.height -= orig.height;
    panoSize.width -= orig.width;


    //Smooth the vertical direction
    Mat avg(1, images.size(), CV_64F), runningAvg, rot;

    for (int i = 0; i < images.size(); ++i)
        avg.at<double>(i) = transforms[i].at<double>(1, 2);
    blur(avg, runningAvg, Size(500, 1), Point(-1, -1), BORDER_REPLICATE);

    //Adjust the size by that smoothing
    double min, max;
    minMaxIdx(runningAvg, &min, &max);
    panoSize.height -= max;

    subtract(avg, runningAvg, avg);
    add(avg, -orig.height, avg);

    //Smooth the rotation so it doesn't fall off the way it did before.
    for (int i = 0; i < images.size(); ++i)
    {
        transforms[i].at<double>(1, 2) = avg.at<double>(i);
        double angle = atan2(-transforms[i].at<double>(0, 1), transforms[i].at<double>(0, 0));
        avg.at<double>(i) = angle * 180 / CV_PI;
    }

    blur(avg, runningAvg, Size(300, 1), Point(-1, -1), BORDER_REFLECT);
    for (int i = 0; i < images.size(); ++i)
    {
        rot = getRotationMatrix2D(Point2f(images[i].cols / 2, images[i].rows / 2), runningAvg.at<double>(i) - avg.at<double>(0), 1);
        transforms[i](Rect(0, 0, 2, 2)) = rot(Rect(0, 0, 2, 2)) * transforms[i](Rect(0, 0, 2, 2));
        transforms[i].at<double>(0, 2) -= rot.at<double>(0, 2);
        transforms[i].at<double>(1, 2) -= rot.at<double>(1, 2);
    }

    //Create the panorama
    panorama.create(panoSize, CV_8UC3);
    panorama.setTo(0);

    //Warp the images to the panorama
    for (int i = 0; i < images.size(); ++i)
    {
        warpAffine(images[i], panorama, transforms[i], Size(panorama.cols, panorama.rows), 1, BORDER_TRANSPARENT);
    }

    return panorama;
}

int main()
{
    LARGE_INTEGER start, stop, freq;
    using namespace cv;
    dist::init(1280, 720);

    Mat pano, pano2, image;
    VideoCapture vidIn("Gallery4.mp4");
    Ptr<ORB> pORB = ORB::create(10000);
    Ptr<ORB> iORB = ORB::create(2000);
    std::vector<KeyPoint> pKps, iKps;
    Mat pDesc, iDesc;

    for (int i = 0; i < 250; ++i)
        vidIn.read(image);

    pano = imread("Panorama.png");
    pano2 = imread("Panorama_30.png");

    pORB->detectAndCompute(pano, noArray(), pKps, pDesc);
    pORB->detectAndCompute(pano2, noArray(), iKps, iDesc);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    std::vector< DMatch > matches;
    matcher->match(pDesc, iDesc, matches);

    double max_dist = 0; double min_dist = 100;
    //-- Quick calculation of max and min distances between keypoints
    for (int i = 0; i < pDesc.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);
    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //-- small)
    //-- PS.- radiusMatch can also be used here.
    std::vector< DMatch > good_matches;
    std::vector<Point2f> pPts, iPts;
    for (int i = 0; i < pDesc.rows; i++)
    {
        if (matches[i].distance <= max(min_dist, 0.02))
        {
            good_matches.push_back(matches[i]);
            pPts.push_back(pKps[matches[i].queryIdx].pt);
            iPts.push_back(iKps[matches[i].trainIdx].pt);
        }
    }

    Mat img_matches;
    drawMatches(pano, pKps, pano2, iKps,
        good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
        std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    imwrite("Matches.png", img_matches);
    imshow("Panorama", pano);

    std::vector<uchar> mask;
    Mat homography = findHomography(pPts, iPts, RANSAC, 3.0, mask, 2000, 0.995);
    std::cout << pPts.size() << "\n";
    std::cout << homography << " " << homography.cols << "\n";
    Mat warp;
    warpPerspective(pano2, warp, homography, Size(pano.cols, pano.rows), INTER_CUBIC | WARP_INVERSE_MAP, BORDER_CONSTANT, 0);
    absdiff(pano, warp, pano);
    int removed = 0;
    for (int i = 0, maskIdx = 0; i < pPts.size(); ++i, ++maskIdx)
    {
        if (mask[maskIdx] == 1)
        {
            pPts.erase(pPts.begin() + i);
            iPts.erase(iPts.begin() + i);
            --i;
        }
    }
    homography = findHomography(pPts, iPts, RANSAC, 3.0, mask, 2000, 0.995);
    warpPerspective(pano2, warp, homography, Size(pano.cols, pano.rows), INTER_CUBIC | WARP_INVERSE_MAP, BORDER_CONSTANT, 0);
    absdiff(pano, warp, pano);

    imwrite("Diff.png", pano);
    imshow("Diff", pano);

    waitKey();

    return 0;
}
