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
#include "opencv2\reg\mapaffine.hpp"
#include "opencv2\reg\mapshift.hpp"
#include "opencv2\reg\mapprojec.hpp"
#include "opencv2\reg\mappergradshift.hpp"
#include "opencv2\reg\mappergradeuclid.hpp"
#include "opencv2\reg\mappergradsimilar.hpp"
#include "opencv2\reg\mappergradaffine.hpp"
#include "opencv2\reg\mappergradproj.hpp"
#include "opencv2\reg\mapperpyramid.hpp"
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

int threshHigh = 255, N = 1;
int threshLow = 150;


static double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

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
        gray0.create(image.size(), CV_8U);
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
                gray = gray0 >= 50;
                Canny(gray, gray, threshLow, threshHigh, 5);
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
                    if (maxCosine < 0.2)
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
void bubbleSort(std::vector<cv::Point2f>& input1, std::vector<cv::Point2f>& output1, std::vector<cv::Point2f>& input2, std::vector<cv::Point2f>& output2) {
    for (int i = 0; i < input1.size(); i++) {
        output1.push_back(input1[i]);
        output2.push_back(input2[i]);
    }
    for (int i = 0; i < output1.size(); i++) {
        bool isSorted = true;
        for (int i = 0; i < output1.size() - 1; i++) {
            if (output1[i].x > output1[i + 1].x) {
                isSorted = false;
                cv::Point2f temp = output1[i];
                output1[i] = output1[i + 1];
                output1[i + 1] = temp;
                temp = output2[i];
                output2[i] = output2[i + 1];
                output2[i + 1] = temp;
            }
        }

        if (isSorted) break;
    }
    int turnoverIndex = -1;
    for (int i = 0; i < output2.size() - 1; i++) {
        if (output2[i].x > output2[i + 1].x) {
            if (turnoverIndex == -1) {
                turnoverIndex = i + 1;
            }
            else {
                output1.erase(output1.begin() + turnoverIndex);
                output2.erase(output2.begin() + turnoverIndex);
                i--;
                turnoverIndex = i + 1;
            }
        }
    }
}

int main() {

    using namespace cv;
    //dist::init(1280, 720);

    Mat pano, pano2, image;
    Ptr<ORB> pORB = ORB::create(10000);
    Ptr<ORB> iORB = ORB::create(2000);
    std::vector<KeyPoint> pKps, iKps;
    Mat pDesc, iDesc;

    //for (int i = 0; i < 250; ++i)
    //vidIn.read(image);
    //read the panoramas in
    pano = imread("Gallery_2_(2).png");
    pano2 = imread("Gallery_2_(3).png");

    //std::cout << pano.cols << std::endl;
    //compute keypoints
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

    std::vector< DMatch > good_matches;
    std::vector<Point2f> pPts, iPts;

    //find the "good matches"
    for (int i = 0; i < pDesc.rows; i++)
    {
        if (matches[i].distance <= max(10 * min_dist, 0.02))
        {
            good_matches.push_back(matches[i]);
            pPts.push_back(pKps[matches[i].queryIdx].pt);
            iPts.push_back(iKps[matches[i].trainIdx].pt);
        }
    }

    std::vector<Point2f> sortedPPts, sortedIPts;
    bubbleSort(pPts, sortedPPts, iPts, sortedIPts);
    std::vector<Mat> subImages, subImages2;

    int index = 0;
    for (int i = 0; i < sortedPPts.size() - 1; i++) {
        if (sortedPPts[i].x - sortedPPts[index].x < 500) continue;
        subImages.push_back(pano(Rect(sortedPPts[index].x, 0, sortedPPts[i].x - sortedPPts[index].x, pano.rows)));
        if (sortedIPts[i].x < sortedIPts[index].x) {
            Mat image1 = pano2(Rect(sortedIPts[index].x, 0, pano2.cols - sortedIPts[index].x, pano2.rows));
            Mat image2 = pano2(Rect(0, 0, sortedIPts[i].x, pano2.rows));
            hconcat(image1, image2, image1);
            subImages2.push_back(image1);
        }
        else {
            subImages2.push_back(pano2(Rect(sortedIPts[index].x, 0, sortedIPts[i].x - sortedIPts[index].x, pano2.rows)));
        }
        index = i;
    }
    Mat image1 = pano(Rect(sortedPPts[index].x, 0, pano.cols - sortedPPts[index].x, pano.rows));
    Mat image2 = pano(Rect(0, 0, sortedPPts[0].x, pano.rows));
    hconcat(image1, image2);
    subImages.push_back(image1);
    if (sortedIPts[index].x > sortedIPts[0].x) {
        image1 = pano2(Rect(sortedIPts[index].x, 0, pano2.cols - sortedIPts[index].x, pano2.rows));
        image2 = pano2(Rect(0, 0, sortedIPts[0].x, pano2.rows));
        hconcat(image1, image2, image1);
        subImages2.push_back(image1);
    }
    else {
        subImages2.push_back(pano2(Rect(sortedIPts[index].x, 0, sortedIPts[0].x - sortedIPts[index].x, pano2.rows)));
    }


    std::vector<Mat> diffs;
    for (int i = 0; i < subImages.size(); i++) {
        std::vector<KeyPoint> pKps2, iKps2;
        Mat pDesc2, iDesc2;
        pORB->detectAndCompute(subImages[i], noArray(), pKps2, pDesc2);
        pORB->detectAndCompute(subImages2[i], noArray(), iKps2, iDesc2);

        std::vector<DMatch> matches2;
        matcher->match(pDesc2, iDesc2, matches2);

        double max_dist2 = 0; double min_dist2 = 100;

        for (int j = 0; j < pDesc2.rows; j++) {
            double dist = matches2[j].distance;
            if (dist < min_dist2) min_dist2 = dist;

            if (dist > max_dist2) max_dist2 = dist;

        }
        std::vector<DMatch> good_matches2;
        std::vector <Point2f> pPts2, iPts2;


        for (int j = 0; j < pDesc2.rows; j++) {
            if (matches2[j].distance <= max(10 * min_dist2, 0.02)) {
                good_matches2.push_back(matches2[j]);
                pPts2.push_back(pKps2[matches2[j].queryIdx].pt);
                iPts2.push_back(iKps2[matches2[j].trainIdx].pt);
            }
        }

        std::vector<uchar> mask;


        Mat transform = findHomography(pPts2, iPts2, RANSAC, 3.0, mask, 2000, 0.995);
        Mat proj;
        warpPerspective(subImages2[i], proj, transform, subImages[i].size(), INTER_CUBIC | WARP_INVERSE_MAP, BORDER_CONSTANT, 0);
        for (int j = 0, maskIdx = 0; j < pPts2.size(); ++j, ++maskIdx) {
            if (mask[maskIdx] == 1) {
                pPts2.erase(pPts2.begin() + j);
                iPts2.erase(iPts2.begin() + j);
                --j;
            }
        }
        /*
        transform = findHomography(pPts2,iPts2,RANSAC,3.0,mask,2000,0.995);
        warpPerspective(subImages2[i],proj,transform,subImages[i].size(),INTER_CUBIC | WARP_INVERSE_MAP, BORDER_CONSTANT,0);
        */

        // Register
        reg::MapperGradProj mapper;
        reg::MapperPyramid mappPyr(mapper);
        mappPyr.numLev_ = 5;
        mappPyr.numIterPerScale_ = 5;
        Ptr<reg::Map> mapPtr;
        resize(subImages2[i], subImages2[i], Size(subImages[i].cols, subImages[i].rows));
        subImages[i].convertTo(subImages[i], CV_64F);
        subImages2[i].convertTo(subImages2[i], CV_64F);
        mappPyr.calculate(subImages[i], subImages2[i], mapPtr);
        reg::MapProjec* mapProj = dynamic_cast<reg::MapProjec*>(mapPtr.get());
        mapProj->inverseWarp(subImages2[i], proj);

        Mat diff;
        absdiff(subImages[i], proj, diff);
        diff.convertTo(diff, CV_8U);
        Mat white(subImages2[i]);
        compare(subImages2[i], Scalar(0, 0, 0), white, CMP_GT);
        mapProj->inverseWarp(white, white);
        //warpPerspective(white, white, transform, diff.size(), INTER_CUBIC | WARP_INVERSE_MAP, BORDER_CONSTANT, 0);
        imshow("blargh", white);
        bitwise_and(diff, white, diff);
        compare(subImages[i], Scalar(0, 0, 0), white, CMP_GT);
        bitwise_and(diff, white, diff);
        diffs.push_back(diff);
    }
    int minheight = diffs[0].rows;
    for (int i = 0; i < diffs.size(); i++) {
        minheight = min(minheight, diffs[i].rows);
    }
    diffs[0] = diffs[0](Rect(0, 0, diffs[0].cols, minheight));
    Mat DiffPano = diffs[0];

    for (int i = 1; i < diffs.size(); i++) {
        diffs[i] = diffs[i](Rect(0, 0, diffs[i].cols, minheight));
        hconcat(DiffPano, diffs[i], DiffPano);
    }


    std::vector<std::vector<Point2i> > points;
    findSquares(DiffPano, points);
    std::cout << "MEEEEE" << std::endl;
    drawSquares(DiffPano, points);

    imwrite("ExperimentalDifference7.png", DiffPano);

    imshow("test", DiffPano);
    waitKey();




    return 0;
}
