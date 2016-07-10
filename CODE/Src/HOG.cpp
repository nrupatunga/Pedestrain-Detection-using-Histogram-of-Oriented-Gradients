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

#include "HOG.hpp"

HOG::HOG(String &strPosSamples, String &strNegSamples, Size &sizeWin = Size(64, 128))
{
	glob(strPosSamples, vecStrPosSamples);
	glob(strNegSamples, vecStrNegSamples);

	s32NumPosSamples = vecStrPosSamples.size();
	s32NumNegSamples = vecStrNegSamples.size();
	if (s32NumPosSamples <= 0 || s32NumNegSamples <= 0){
		cout << "Invalid Training Sample Path/ Please check if there are any training samples in the given path" << endl;
		return;
	}
	objHog.winSize = sizeWin;
	s32NumOfSamples = s32NumPosSamples + s32NumNegSamples;

	return;
}

HOG::enumHOGError HOG::getFeatures(String &strFeatureFileName, HOG::enumSVMLib eSVMLib)
{
	enumHOGError eHOGError   = HOG_NO_ERROR;
	Mat          sMatInImage;

	if (eSVMLib != SVM_LIGHT && eSVMLib != SVM_LIBSVM){
		cout << " Invalid SVM library specified";
		eSVMLib = SVM_LIBSVM;
	}

	if (s32NumOfSamples <= 0) {
		cout << " ------------------------------------------------------------" << endl;
		cout << " Please provide samples to train" << endl << " Refer to the function:: HOG(String &strPosSamples, String &strNegSamples, Size &sizeWin)" << endl;
		cout << " ------------------------------------------------------------" << endl;
		return HOG_INVALID_ARGUMENTS;
	}

	ofstream outFeatureFile;
	outFeatureFile.open(strFeatureFileName, std::ofstream::trunc);
	if (!(outFeatureFile.good() && outFeatureFile.is_open())) {
		cout << "File " << strFeatureFileName << " failed to open. Please check the path" << endl;
		return HOG_INVALID_ARGUMENTS;
	}

#if WRITE_FILE_NAMES
	ofstream outFileNames;
	outFileNames.open("filenames.txt", std::ofstream::trunc);
#endif

	// Remove following line for libsvm which does not support comment
	if (eSVMLib == SVM_LIGHT){
		outFeatureFile << "# Use this file to train, e.g. SVMlight by issuing $ svm_learn -i 1 -a weights.txt " << strFeatureFileName << endl;
	}
	cout << " ------------------------------------------------------------" << endl;
	cout << " This could take a while, based on the number of training samples " << endl;
	cout << " ------------------------------------------------------------" << endl;
	cout << " Calculating HOG Features ...  " << endl;

	vector<float> vecFeature;
	for (unsigned long i = 0; i < s32NumOfSamples; i++){
		int s32Progress = (i * 100 / s32NumOfSamples);
		const String strCurrentFile = (i < s32NumPosSamples) ? vecStrPosSamples.at(i) : vecStrNegSamples.at(i - s32NumPosSamples);
		sMatInImage = imread(strCurrentFile);
#if WRITE_FILE_NAMES
		outFileNames << strCurrentFile << endl;
#endif
		if (sMatInImage.empty()){
			cout << "Cannot open file " << strCurrentFile << " Please check the format of the image file" << endl;
			break;
		}
		calculateFeature(sMatInImage, vecFeature, objHog);
		Mat sMatHogVis = get_hogdescriptor_visu(sMatInImage, vecFeature, objHog.winSize);
		if (vecFeature.empty()){
			break;
		}
#if 1
		outFeatureFile << ((i < s32NumPosSamples) ? "+1" : "-1");
		for (int j = 0; j < vecFeature.size(); j++){
			int s32Count = j + 1;
			outFeatureFile << " " << s32Count << ":" << vecFeature.at(j) << " ";
		}
#else
		for (int j = 0; j < vecFeature.size(); j++){
			int s32Count = j + 1;
			outFeatureFile << " " << vecFeature.at(j) << " ";
		}
#endif
		outFeatureFile << endl;
		cout << " Progress :: " << s32Progress << "%\r";
		cout.flush();
	}
	cout << " Progress :: " << "DONE.." << endl;

	outFeatureFile.close();
#if WRITE_FILE_NAMES
	outFileNames.close();
#endif
	sMatInImage.release();
	return eHOGError;
}

void HOG::setParameters(Size &sizeWinStride, Size &sizeTrainPadding)
{
	if (sizeWinStride.width > 0 && sizeWinStride.height > 0)
		winStride = sizeWinStride;
	else
		cout << "Invalid Window Stride " << endl;

	if (sizeTrainPadding.width > 0 && sizeTrainPadding.height > 0)
		trainingPadding = sizeTrainPadding;
	else
		cout << "Invalid Training padding " << endl;

}

void HOG::calculateFeature(const Mat &sMatInput, vector<float> &vecDescriptor, HOGDescriptor &objHog)
{
	Size hogWinSize = objHog.winSize;
	if (sMatInput.cols != hogWinSize.width || sMatInput.rows != hogWinSize.height){
		vecDescriptor.clear();
		cout << " ------------------------------------------------------------" << endl;
		cout << "Input image dimensions " << sMatInput.cols << "x" << sMatInput.rows << \
			" do not match with HOG window size " << hogWinSize.width << "x" << hogWinSize.height << endl;
		cout << " ------------------------------------------------------------" << endl;
		return;
	}

	objHog.compute(sMatInput, vecDescriptor, winStride, trainingPadding);
	return;
}


//Source code from http://www.juergenwiki.de/work/wiki/doku.php?id=public:hog_descriptor_computation_and_visualization

Mat HOG::get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size)
{
	const int DIMX = size.width;
	const int DIMY = size.height;
	float zoomFac = 3;
	Mat visu;
	resize(color_origImg, visu, Size((int)(color_origImg.cols*zoomFac), (int)(color_origImg.rows*zoomFac)));

	int cellSize = 8;
	int gradientBinSize = 9;
	float radRangeForOneBin = (float)(CV_PI / (float)gradientBinSize); // dividing 180 into 9 bins, how large (in rad) is one bin?

																	   // prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = DIMX / cellSize;
	int cells_in_y_dir = DIMY / cellSize;
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter = new int*[cells_in_y_dir];
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;

			for (int bin = 0; bin<gradientBinSize; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}

	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;

	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;

	for (int blockx = 0; blockx<blocks_in_x_dir; blockx++)
	{
		for (int blocky = 0; blocky<blocks_in_y_dir; blocky++)
		{
			// 4 cells per block ...
			for (int cellNr = 0; cellNr<4; cellNr++)
			{
				// compute corresponding cell nr
				cellx = blockx;
				celly = blocky;
				if (cellNr == 1) celly++;
				if (cellNr == 2) cellx++;
				if (cellNr == 3)
				{
					cellx++;
					celly++;
				}

				for (int bin = 0; bin<gradientBinSize; bin++)
				{
					float gradientStrength = descriptorValues[descriptorDataIdx];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;

				} // for (all bins)


				  // note: overlapping blocks lead to multiple updates of this sum!
				  // we therefore keep track how often a cell was updated,
				  // to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;

			} // for (all cells)


		} // for (all block x pos)
	} // for (all block y pos)


	  // compute average gradient strengths
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}

	// draw cells
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{
			int drawX = cellx * cellSize;
			int drawY = celly * cellSize;

			int mx = drawX + cellSize / 2;
			int my = drawY + cellSize / 2;

			rectangle(visu, Point((int)(drawX*zoomFac), (int)(drawY*zoomFac)), Point((int)((drawX + cellSize)*zoomFac), (int)((drawY + cellSize)*zoomFac)), Scalar(100, 100, 100), 1);

			// draw in each cell all 9 gradient strengths
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				// no line to draw?
				if (currentGradStrength == 0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

				float dirVecX = cos(currRad);
				float dirVecY = sin(currRad);
				float maxVecLen = (float)(cellSize / 2.f);
				float scale = 2.5; // just a visualization scale, to see the lines better

								   // compute line coordinates
				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visualization
				line(visu, Point((int)(x1*zoomFac), (int)(y1*zoomFac)), Point((int)(x2*zoomFac), (int)(y2*zoomFac)), Scalar(0, 255, 0), 1);

			} // for (all bins)

		} // for (cellx)
	} // for (celly)


	  // don't forget to free memory allocated by helper data structures!
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;

	return visu;

} // get_hogdescriptor_visu