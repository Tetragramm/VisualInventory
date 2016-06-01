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

#undef sign
/*
 * Sorts input1 into output2 and sorts input2 relative to output 1
 *eg. input1:[3,1,2] input2:[1,2,3] goes to output1:[1,2,3] output2:[2,3,1]
*/
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
		if (isSorted) return;
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
	std::vector<Mat> subImages,subImages2;
	//std::cout << "TEST:" << std::endl;
	for (int i = 0; i < sortedPPts.size(); i++) {
		//std::cout << sortedPPts[i].y << "\t" << sortedIPts[i].y << std::endl;
	}
	//subdivide the image
	for (int i = 0; i < sortedPPts.size()-1; i++) {
		//if the second panorama is wraping around itself
		if (sortedIPts[i].x > sortedIPts[i + 1].x) {
			//divide the panoramas into 3 sections for y coordinates and create a rectangle based on which coordiate it is.
			if (sortedPPts[i].y < 250) {
				subImages.push_back(pano(Rect(sortedPPts[i].x, sortedPPts[i].y, sortedPPts[i + 1].x - sortedPPts[i].x, 500)));
				Mat image1 = pano2(Rect(sortedIPts[i].x, sortedIPts[i].y, pano2.cols - sortedIPts[i].x, 500));
				Mat image2 = pano2(Rect(0, sortedIPts[i].y, sortedIPts[i + 1].x, 500));
				Mat image3;
				hconcat(image1,image2,image3);
				subImages2.push_back(image3);
			}
			else if (sortedPPts[i].y > 250 && sortedPPts[i].y < 500) {
				subImages.push_back(pano(Rect(sortedPPts[i].x, sortedPPts[i].y - 250, sortedPPts[i + 1].x - sortedPPts[i].x, 500)));
				Mat image1 = pano2(Rect(sortedIPts[i].x, max(sortedIPts[i].y-250,0.0f), pano2.cols - sortedIPts[i].x, 500));
				Mat image2 = pano2(Rect(0, max(sortedIPts[i].y-250,0.0f), sortedIPts[i + 1].x, 500));
				Mat image3;
				hconcat(image1, image2, image3);
				subImages2.push_back(image3);
			}
			else if (sortedPPts[i].y > 500) {
				subImages.push_back(pano(Rect(sortedPPts[i].x, sortedPPts[i].y - 500, sortedPPts[i + 1].x - sortedPPts[i].x, 500)));
				Mat image1 = pano2(Rect(sortedIPts[i].x, max(sortedIPts[i].y - 500, 0.0f), pano2.cols - sortedIPts[i].x, 500));
				Mat image2 = pano2(Rect(0, max(sortedIPts[i].y - 500, 0.0f), sortedIPts[i + 1].x, 500));
				Mat image3;
				hconcat(image1, image2, image3);
				subImages2.push_back(image3);
			}
		}
		else {
			if (sortedPPts[i].y < 250) {
				subImages.push_back(pano(Rect(sortedPPts[i].x, sortedPPts[i].y, sortedPPts[i + 1].x - sortedPPts[i].x, 500)));
				subImages2.push_back(pano2(Rect(sortedIPts[i].x, sortedIPts[i].y, sortedIPts[i + 1].x - sortedIPts[i].x, 500)));
			}
			else if (sortedPPts[i].y > 250 && sortedPPts[i].y < 500) {
				subImages.push_back(pano(Rect(sortedPPts[i].x, sortedPPts[i].y-250, sortedPPts[i + 1].x - sortedPPts[i].x, 500)));
				subImages2.push_back(pano2(Rect(sortedIPts[i].x, max(sortedIPts[i].y-250,0.0f), sortedIPts[i + 1].x - sortedIPts[i].x, 500)));
			}
			else if (sortedPPts[i].y > 500) {
				subImages.push_back(pano(Rect(sortedPPts[i].x, sortedPPts[i].y - 500, sortedPPts[i + 1].x - sortedPPts[i].x, 500)));
				subImages2.push_back(pano2(Rect(sortedIPts[i].x, max(sortedIPts[i].y - 500, 0.0f), sortedIPts[i + 1].x - sortedIPts[i].x, 500)));
			}
			
		}

	}
	//Perform the last subimage operation(first panorama is wrapping around)
	Mat i1,i2,i3;
	if (sortedIPts[sortedIPts.size()-1].x > sortedIPts[0].x) {
		if (sortedPPts[sortedPPts.size()-1].y < 250) {
			i1 = pano(Rect(sortedPPts[sortedPPts.size()-1].x, sortedPPts[sortedPPts.size()-1].y, pano.cols - sortedPPts[sortedPPts.size()-1].x, 500));
			i2 = pano(Rect(0, sortedPPts[sortedPPts.size() - 1].y,  sortedPPts[0].x, 500));
			hconcat(i1, i2,i3);
			subImages.push_back(i3);
			Mat image1 = pano2(Rect(sortedIPts[sortedIPts.size()-1].x, sortedIPts[sortedIPts.size()-1].y, pano2.cols - sortedIPts[sortedIPts.size()-1].x, 500));
			Mat image2 = pano2(Rect(0, sortedIPts[sortedIPts.size()-1].y, sortedIPts[0].x, 500));
			Mat image3;
			hconcat(image1, image2, image3);
			subImages2.push_back(image3);
		}
		else if (sortedPPts[sortedPPts.size()-1].y > 250 && sortedPPts[sortedPPts.size()-1].y < 500) {
			i1 = (pano(Rect(sortedPPts[sortedPPts.size()-1].x, sortedPPts[sortedPPts.size()-1].y - 250, pano.cols - sortedPPts[sortedPPts.size()-1].x, 500)));
			i2 = pano(Rect(0, sortedPPts[sortedPPts.size() - 1].y - 250, sortedPPts[0].x, 500));
			hconcat(i1, i2, i3);
			subImages.push_back(i3);
			Mat image1 = pano2(Rect(sortedIPts[sortedIPts.size()-1].x, max(sortedIPts[sortedIPts.size()-1].y - 250, 0.0f), pano2.cols - sortedIPts[sortedIPts.size()-1].x, 500));
			Mat image2 = pano2(Rect(0, max(sortedIPts[sortedIPts.size()-1].y - 250, 0.0f), sortedIPts[0].x, 500));
			Mat image3;
			hconcat(image1, image2, image3);
			subImages2.push_back(image3);
		}
		else if (sortedPPts[sortedPPts.size()-1].y > 500) {
			i1 = pano(Rect(sortedPPts[sortedPPts.size()-1].x, sortedPPts[sortedPPts.size()-1].y - 500, pano.cols - sortedPPts[sortedPPts.size()-1].x, 500));
			i2 = pano(Rect(0, sortedPPts[sortedPPts.size() - 1].y - 500, sortedPPts[0].x, 500));
			hconcat(i1, i2, i3);
			Mat image1 = pano2(Rect(sortedIPts[sortedIPts.size()-1].x, max(sortedIPts[sortedIPts.size()-1].y - 500, 0.0f), pano2.cols - sortedIPts[sortedIPts.size()-1].x, 500));
			Mat image2 = pano2(Rect(0, max(sortedIPts[sortedIPts.size()-1].y - 500, 0.0f), sortedIPts[0].x, 500));
			Mat image3;
			hconcat(image1, image2, image3);
			subImages.push_back(i3);
			subImages2.push_back(image3);
		}
	}
	else {
		if (sortedPPts[sortedPPts.size()-1].y < 250) {
			i1 = pano(Rect(sortedPPts[sortedPPts.size()-1].x, sortedPPts[sortedPPts.size()-1].y, pano.cols - sortedPPts[sortedPPts.size()-1].x, 500));
			i2 = pano(Rect(0, sortedPPts[sortedPPts.size() - 1].y,sortedPPts[0].x, 500));
			hconcat(i1, i2, i3);
			subImages.push_back(i3);
			subImages2.push_back(pano2(Rect(sortedIPts[sortedIPts.size()-1].x, sortedIPts[sortedIPts.size()-1].y, sortedIPts[0].x - sortedIPts[sortedIPts.size()-1].x, 500)));
		}
		else if (sortedPPts[sortedPPts.size()-1].y > 250 && sortedPPts[sortedPPts.size()-1].y < 500) {
			i1 = (pano(Rect(sortedPPts[sortedPPts.size()-1].x, sortedPPts[sortedPPts.size()-1].y - 250, pano.cols - sortedPPts[sortedPPts.size()-1].x, 500)));
			i2 = pano(Rect(0, sortedPPts[sortedPPts.size() - 1].y - 250, sortedPPts[0].x, 500));
			hconcat(i1, i2, i3);
			subImages.push_back(i3);
			subImages2.push_back(pano2(Rect(sortedIPts[sortedIPts.size()-1].x, max(sortedIPts[sortedIPts.size()-1].y - 250, 0.0f), sortedIPts[0].x - sortedIPts[sortedIPts.size()-1].x, 500)));
		}
		else if (sortedPPts[sortedPPts.size()-1].y > 500) {
			i1 = pano(Rect(sortedPPts[sortedPPts.size()-1].x, sortedPPts[sortedPPts.size()-1].y - 500, pano.cols - sortedPPts[sortedPPts.size()-1].x, 500));
			i2 = pano(Rect(0, sortedPPts[sortedPPts.size() - 1].y - 500, sortedPPts[0].x, 500));
			hconcat(i1, i2, i3);
			subImages.push_back(i3);
			subImages2.push_back(pano2(Rect(sortedIPts[sortedIPts.size()-1].x, max(sortedIPts[sortedIPts.size()-1].y - 500, 0.0f), sortedIPts[0].x - sortedIPts[sortedIPts.size()-1].x, 500)));
		}

	}
	imshow("test", subImages[1]);
	imshow("test2", subImages2[1]);
	waitKey();

	return 0;
}