#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <windows.h>
#include "opencv2\calib3d.hpp"
#include "opencv2\core.hpp"
#include "opencv2\features2d.hpp"
#include "opencv2\flann.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\imgcodecs.hpp"
#include "opencv2\imgproc.hpp"
#include "opencv2\ml.hpp"
#include "opencv2\objdetect.hpp"
#include "opencv2\opencv.hpp"
#include "opencv2\opencv_modules.hpp"
#include "opencv2\photo.hpp"
#include "opencv2\shape.hpp"
#include "opencv2\stitching.hpp"
#include "opencv2\superres.hpp"
#include "opencv2\video.hpp"
#include "opencv2\videoio.hpp"
#include "opencv2\videostab.hpp"
#include "Queue.h"
#include <string.h>

const int SECTION_HEIGHT = 10;

int impose(int x) {
	return x%SECTION_HEIGHT;
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
				output2.erase(output2.begin()+turnoverIndex);
				i--;
				turnoverIndex = i + 1;
			}
		}
	}
}

int main() {
	LARGE_INTEGER start, stop, freq;
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
	pano = imread("Panorama.png");
	pano2 = imread("Panorama_30.png");

	int height1 = pano.rows - SECTION_HEIGHT;
	int height2 = pano2.rows - SECTION_HEIGHT;

	//std::cout << pano.cols << std::endl;
	//compute keypoints
	pORB->detectAndCompute(pano, noArray(), pKps, pDesc);
	pORB->detectAndCompute(pano2, noArray(), iKps, iDesc);
	//compute the matches
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
	std::vector<cv::Point2f> sortedPPts, sortedIPts;
	//sort keypoints into the two sorted vectors above
	bubbleSort(pPts, sortedPPts, iPts, sortedIPts);
	//vectors of the various subimages of the panorama
	std::vector<Mat> subImages, subImages2;
	std::vector<Point2f> usedPPts, usedIPts;
	int index = 0;

	for (int i = 1; i < sortedPPts.size()-1; i++) {
		if (sortedPPts[i].x - sortedPPts[index].x < 250) continue;
		if (sortedIPts[index].x > sortedIPts[i].x) {
			subImages.push_back(pano(Rect(sortedPPts[index].x, impose((int)sortedPPts[index].y), sortedPPts[i].x - sortedPPts[index].x, height1)));
			Mat image1 = pano2(Rect(sortedIPts[index].x, impose((int)sortedIPts[index].y), pano2.cols - sortedIPts[index].x, height2));
			Mat image2 = pano2(Rect(0, impose((int)sortedIPts[index].y), sortedIPts[i].x, height2));
			hconcat(image1, image2, image1);
			subImages2.push_back(image1);
		}
		else {
			subImages.push_back(pano(Rect(sortedPPts[index].x, impose((int)sortedPPts[index].y), sortedPPts[i].x - sortedPPts[index].x, height1)));
			subImages2.push_back(pano2(Rect(sortedIPts[index].x, impose((int)sortedIPts[index].y), sortedIPts[i].x - sortedIPts[index].x, height2)));
		}
		usedPPts.push_back(sortedPPts[index]);
		usedIPts.push_back(sortedIPts[index]);
		index = i;
	}
	usedPPts.push_back(sortedPPts[index]);
	usedIPts.push_back(sortedIPts[index]);
	if (sortedIPts[index].x > sortedIPts[0].x) {
		Mat i1 = pano(Rect(sortedPPts[index].x, impose((int)sortedPPts[index].y), pano.cols - sortedPPts[index].x, height1));
		Mat i2 = pano(Rect(0, impose((int)sortedPPts[index].y), sortedPPts[0].x, height1));
		hconcat(i1, i2, i1);
		subImages.push_back(i1);
		Mat image1 = pano2(Rect(sortedIPts[index].x, impose((int)sortedIPts[index].y), pano2.cols - sortedIPts[index].x, height2));
		Mat image2 = pano2(Rect(0, impose((int)sortedIPts[index].y), sortedIPts[0].x, height2));
		hconcat(image1, image2, image1);
		subImages2.push_back(image1);
	}
	else {
		Mat i1 = pano(Rect(sortedPPts[index].x, impose((int)sortedPPts[index].y), pano.cols - sortedPPts[index].x, height1));
		Mat i2 = pano(Rect(0, impose((int)sortedPPts[index].y), sortedPPts[0].x, height1));
		hconcat(i1, i2, i1);
		subImages.push_back(i1);
		subImages2.push_back(pano2(Rect(sortedIPts[index].x, impose((int)sortedIPts[index].y), sortedIPts[0].x - sortedIPts[index].x, height2)));
	}
	int minY1 = 0;

	int minY2 = 0;

	int minHeight = min(height1, height2);

	for (int i = 0; i < usedPPts.size(); i++) {
		minY1 = min(minY1, impose((int)usedPPts[i].y));
	}

	for (int i = 0; i < usedIPts.size(); i++) {
		minY2 = min(minY2, impose((int)usedIPts[i].y));
	}
	std::vector<Mat> subImageDiffs;
	for (int i = 0; i < subImages.size(); i++) {
		int minWdith = min(subImages[i].cols, subImages2[i].cols);
		Mat image1 = subImages[i](Rect(0, 0, minWdith, minHeight));
		Mat image2 = subImages2[i](Rect(0, 0, minWdith, minHeight));
		absdiff(image1, image2, image1);
		subImageDiffs.push_back(image1);
	}

	int newHeight = minHeight - (SECTION_HEIGHT - min(minY1,minY2));

	for (int i = 0; i < subImageDiffs.size(); i++) {
		int y = SECTION_HEIGHT - impose((int)min(usedPPts[i].y,usedIPts[i].y));

		subImageDiffs[i] = subImageDiffs[i](Rect(0, y, subImageDiffs[i].cols, newHeight));
	}

	Mat diffPano = subImageDiffs[0];

	for (int i = 1; i < subImageDiffs.size(); i++) {
		hconcat(diffPano, subImageDiffs[i], diffPano);

	}
	imshow("diff", diffPano);
	imwrite("difference.png", diffPano);
	waitKey();

	return 0;

}