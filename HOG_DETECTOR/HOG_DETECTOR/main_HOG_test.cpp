/* The MIT License (MIT)

Copyright (c) [2015] [Nrupatunga]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. */

#include <stdio.h>
#include <ios>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <iterator>
#include "HOG.hpp"

#pragma comment(lib, "opencv_core2413d.lib")
#pragma comment(lib, "opencv_features2d2413d.lib")
#pragma comment(lib, "opencv_highgui2413d.lib")
#pragma comment(lib, "opencv_imgproc2413d.lib")
#pragma comment(lib, "opencv_ml2413d.lib")
#pragma comment(lib, "opencv_objdetect2413d.lib")


//#define WIN_WIDTH (96)
//#define WIN_HEIGHT (160)

#define WIN_WIDTH (64)
#define WIN_HEIGHT (128)

using namespace std;
using namespace cv;

//static string descriptorVectorFile = "descriptor_svm_model_jul09.dat";
static string descriptorVectorFile = "descriptor_svm_model_jul10_hard.dat";
static const Size trainingPadding  = Size(0, 0);
static const Size winStride        = Size(8, 8);
const double hitThreshold          = 1.4;

void nms(vector<Rect> vecRect, vector<Rect>& vecRectNew, float fThreshold)
{
	int s32NumBoxes = vecRect.size();
	if(s32NumBoxes == 0) 
		return;

	vector<int> vecInd;
	vector<int> vecArea;
	vector<int> vecBotRightY;

	//Find out the area of all the boxes, also find the bottom right y-co-ordinate
	for(int i = 0; i < s32NumBoxes; i++){
		Rect r = vecRect.at(i);
		int s32Area = r.width * r.height;
		vecArea.push_back(s32Area);
		vecBotRightY.push_back(r.y + r.height + 1);
	}
	vector<int> vecSortIndx(vecBotRightY.size());
	size_t n(0);
	std::generate(std::begin(vecSortIndx), std::end(vecSortIndx), [&]{ return n++; });
	std::sort(  std::begin(vecSortIndx), std::end(vecSortIndx), [&](int i1, int i2) { return vecBotRightY[i1] < vecBotRightY[i2]; } );
	vector<int> vecPick;
	while(vecSortIndx.size() > 0){
		int s32LastIndx = vecSortIndx.size() - 1;
		int i = vecSortIndx.at(s32LastIndx);
		vector<int> vecSuppress;
		vecPick.push_back(i);
		vecSuppress.push_back(s32LastIndx);
		for (int k = 0; k < s32LastIndx; k++){
			int j = vecSortIndx.at(k);
			Rect r1 = vecRect.at(j);
			Rect r2 = vecRect.at(i);
			int xx1 = max(r2.x, r1.x);
			int yy1 = max(r2.y, r1.y);
			int xx2 = min(r2.x + r2.width + 1, r1.x + r1.width + 1);
			int yy2 = min(r2.y + r2.height + 1, r1.y + r1.height + 1);

			int s32W = max(0, xx2 - xx1 + 1);
			int s32H = max(0, yy2 - yy1 + 1);
			float fOverlap = float(s32W*s32H)/(vecArea.at(j));
			if(fOverlap > fThreshold)
				vecSuppress.push_back(k);
		}
		sort(vecSuppress.begin(), vecSuppress.end(), [](int a, int b){ return a>b;});
		for(int m = 0; m < vecSuppress.size(); m++){
			vecSortIndx.erase(vecSortIndx.begin() + vecSuppress.at(m));
		}
	}

	for (int i = 0; i < vecPick.size(); i++){
		vecRectNew.push_back(vecRect.at(vecPick.at(i)));
	}
}

/**
* Shows the detections in the image
* @param found vector containing valid detection rectangles
* @param imageData the image in which the detections are drawn
*/
//#define DUMP_FP
int s32HardCount = 0;
static void showDetections(const vector<Rect>& found, Mat& imageData) {
	vector<Rect> found_filtered;
	size_t i, j;
	for (i = 0; i < found.size(); ++i) {
		Rect r = found[i];
		for (j = 0; j < found.size(); ++j)
				if (j != i && (r & found[j]) == r)
				break;
		if (j == found.size())
			found_filtered.push_back(r);
	}

	vector<Rect> vecRect;
	vector<Rect> vecRectNew;
#if 1
	for (i = 0; i < found_filtered.size(); i++) {
		Rect r = found_filtered[i];
		vecRect.push_back(r);
	}
#else
	vecRect.push_back(Rect(12, 84, 140-12+1, 212 - 84 + 1));
	vecRect.push_back(Rect(24, 84, 152-24+1, 212 - 84 + 1));
	vecRect.push_back(Rect(36, 84, 164-36+1, 212 - 84 + 1));
	vecRect.push_back(Rect(12, 96, 140-12+1, 224 - 96 + 1));
	vecRect.push_back(Rect(24, 96, 152-24+1, 224 - 96 + 1));
	vecRect.push_back(Rect(24, 108, 152-24+1, 236 - 108 + 1));
#endif

	nms(vecRect, vecRectNew, 0.3);

	for (i = 0; i < vecRectNew.size(); i++) {
		Rect r = vecRectNew.at(i);
#define DUMP_FP
#ifdef DUMP_FP
		static String strHardNegDir = "..\\hard_negs\\";
		Mat sMatCrop;
		imageData(r).copyTo(sMatCrop);
		String strFileName = strHardNegDir + to_string(s32HardCount) + "_hard.jpg";
		imwrite(strFileName, sMatCrop);
		s32HardCount++;
#else 
		rectangle(imageData, r.tl(), r.br(), Scalar(64, 255, 64), 3);
#endif
	}
}

/**
* Test detection with custom HOG description vector
* @param hog
* @param hitThreshold threshold value for detection
* @param imageData
*/
static void detectTest(const HOGDescriptor& hog, const double hitThreshold, Mat& imageData) {
	vector<Rect> found;
	Size padding(Size(0, 0));
	Size winDetStride = Size(8, 8);
	hog.detectMultiScale(imageData, found, hitThreshold, winDetStride, padding, 1.05, 2.0);
	showDetections(found, imageData);
}

/**
* Function to detect person in image, using your trained SVM model
*/
int main_test_person_detector()
{
	HOGDescriptor hog;
	hog.winSize = Size(WIN_WIDTH, WIN_HEIGHT);

	ifstream infstream(descriptorVectorFile.c_str());
	istream_iterator<float> itrStart(infstream), itrEnd;
	vector<float> descriptorVector(itrStart, itrEnd);
	cout << "Read " << descriptorVector.size() << " samples" << endl;
	//descriptorVector = HOGDescriptor::getDefaultPeopleDetector();
	hog.setSVMDetector(descriptorVector);

	vector<String> vecStrImg;
	String strInFolder = "..\\test\\test";
	String strOutFolder = "..\\test\\results-10thJul\\";

	Mat sMatInputBGR; Mat sMatInputBGRCrop; String strOutImg;
	glob(strInFolder, vecStrImg);
	for (int i = 0; i < vecStrImg.size(); i++) {
		cout << i << " of total " << vecStrImg.size() << endl;
		String strInImg = vecStrImg[i];
		sMatInputBGR = imread(strInImg, IMREAD_COLOR);
		detectTest(hog, hitThreshold, sMatInputBGR);
		strOutImg = strOutFolder + to_string(i + 1) + "_detect.jpg";
		imwrite(strOutImg, sMatInputBGR);
	}
	return EXIT_SUCCESS;
}

/**
* Function to extract HOG feature and dump to a features.dat
*/
int main_HOG_train()
{
#define TRAIN
#ifdef TRAIN
	// Directory containing positive sample images
	static String strPosSamplesDir = "..\\pos_train\\";
	// Directory containing negative sample images
	static String strNegSamplesDir = "..\\neg_train\\";
	// Features are dumped in the this file
	static String strFeaturesFile = "..\\features_train_hard_two.dat";
#else
	// Directory containing positive sample images
	static String strPosSamplesDir = "..\\pos_test\\";
	// Directory containing negative sample images
	static String strNegSamplesDir = "..\\neg_test\\";
	// Features are dumped in the this file
	static String strFeaturesFile = "..\\features_test.dat";
#endif
	HOG objHOG = HOG(strPosSamplesDir, strNegSamplesDir, Size(64, 128));
	//objHOG.HOG_SetParameter(Size(16, 16), Size(2, 2));
	HOG::enumHOGError eHOGError = objHOG.getFeatures(strFeaturesFile, HOG::SVM_LIGHT);
	return EXIT_SUCCESS;
}

int main_test_cam()
{
	HOGDescriptor hog;
	hog.winSize = Size(WIN_WIDTH, WIN_HEIGHT);

	ifstream infstream(descriptorVectorFile.c_str());
	istream_iterator<float> itrStart(infstream), itrEnd;
	vector<float> descriptorVector(itrStart, itrEnd);
	cout << "Read " << descriptorVector.size() << " samples" << endl;
	//hog.setSVMDetector(descriptorVector);
	descriptorVector = HOGDescriptor::getDefaultPeopleDetector();
	hog.setSVMDetector(descriptorVector);

	Mat testImage;
	VideoCapture cap(0);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	if (!cap.isOpened()){
		return EXIT_FAILURE;
	}

	while ((cvWaitKey(10) & 255) != 27) {
		cap >> testImage; // get a new frame from camera
		//cvtColor(testImage, testImage, CV_BGR2GRAY); // If you want to work on grayscale images
		detectTest(hog, hitThreshold, testImage);
		imshow("HOG custom detection", testImage);
	}

	return EXIT_SUCCESS;
}

/**
* This is main()
*/
int main()
{
	//int s32Ret = main_test_person_detector();
	//int s32Ret = main_test_cam();
	int s32Ret = main_HOG_train();
	return s32Ret;
}
