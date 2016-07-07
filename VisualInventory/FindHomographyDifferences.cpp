#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/shape.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/superres.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/videostab.hpp>
#include <string.h>

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
	pano2 = imread("Gallery_2_(3).png");
	pano = imread("Gallery_2_(2).png");

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
	for (int i = 0; i < sortedPPts.size()-1; i++) {
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
		pORB->detectAndCompute(subImages[i],noArray(),pKps2,pDesc2);
		pORB->detectAndCompute(subImages2[i],noArray(),iKps2,iDesc2);

		std::vector<DMatch> matches2;
		matcher->match(pDesc2,iDesc2,matches2);

		double max_dist2 = 0; double min_dist2 = 100;

		for (int j = 0; j < pDesc2.rows; j++) {
			double dist = matches2[j].distance;
			if (dist < min_dist2) min_dist2 = dist;

			if (dist > max_dist2) max_dist2 = dist;

		}
		std::vector<DMatch> good_matches2;
		std::vector <Point2f> pPts2, iPts2;


		for (int j = 0; j < pDesc2.rows; j++) {
			if (matches2[j].distance <= max(10*min_dist2,0.02)) {
				good_matches2.push_back(matches2[j]);
				pPts2.push_back(pKps2[matches2[j].queryIdx].pt);
				iPts2.push_back(iKps2[matches2[j].trainIdx].pt);
			}
		}


		Mat transform = findHomography(pPts2,iPts2,RANSAC,3.0,noArray(),2000,0.995);
		Mat proj;
		warpPerspective(subImages2[i],proj,transform,subImages[i].size(),INTER_CUBIC | WARP_INVERSE_MAP, BORDER_CONSTANT, 0);
		Mat diff;
		absdiff(subImages[i],proj,diff);
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

		imwrite("ExperimentalDifference3.png", DiffPano);

		imshow("test", subImages[0]);
		waitKey();
	

	

	return 0;
}
