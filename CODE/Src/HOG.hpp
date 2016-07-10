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
SOFTWARE.*/

#include <stdio.h>
#include <ios>
#include <fstream>
#include <opencv2/opencv.hpp>

#pragma comment(lib, "opencv_core2413d.lib")
#pragma comment(lib, "opencv_features2d2413d.lib")
#pragma comment(lib, "opencv_highgui2413d.lib")
#pragma comment(lib, "opencv_imgproc2413d.lib")
#pragma comment(lib, "opencv_ml2413d.lib")
#pragma comment(lib, "opencv_objdetect2413d.lib")


using namespace std;
using namespace cv;

/*
 * =====================================================================================
 *        Class:  HOG
 *  Description:  Histogram of gradients, training and detection
 * =====================================================================================
 */
class HOG
{

	public:
		enum enumSVMLib{
			SVM_LIGHT  = 1, // Use SVM light
			SVM_LIBSVM = 2  // Use libSVM
		};

		enum enumHOGError{
			HOG_NO_ERROR = 0,
			HOG_INVALID_ARGUMENTS
		};

		/* ====================  LIFECYCLE     ======================================= */
		HOG(String &strPosSamples, String &strNegSamples, Size &sizeWin);
		void setParameters(Size &sizeWinStride, Size &sizeTrainPadding);
		enumHOGError getFeatures(String &strFeatureFileName, enumSVMLib eSVMLib);
		~HOG(){};

	private:
		Size    trainingPadding;
		Size    winStride;
		int     s32NumOfSamples;
		int     s32NumPosSamples;
		int     s32NumNegSamples;

		vector<String> vecStrPosSamples;
		vector<String> vecStrNegSamples;
		HOGDescriptor objHog;

		void calculateFeature(const Mat &sMatInput, vector<float> &vecDescriptor, HOGDescriptor &objHog);
		Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size);

}; /* -----  end of class HOG  ----- */

