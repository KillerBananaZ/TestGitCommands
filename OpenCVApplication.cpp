// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <random>
#include <fstream>

Mat_<uchar> invert(Mat_<uchar> img)
{
	Mat_<uchar> output = img.clone();
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++)
		{
			output(i, j) = 255 - output(i, j);
		}
	}
	//imshow("negative image", img);
	return output;
}
void lab01_add_grey(char* path, int factor)
{
	Mat_<uchar> img = imread(path, IMREAD_GRAYSCALE);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) + factor > 255)
				img(i, j) = 255;
			else
				if (img(i, j) + factor < 0)
					img(i, j) = 0;
				else
					img(i, j) = img(i, j) + factor;
		}
	}
	imshow("additive", img);
}
void lab01_mul_grey(char* path, float factor)
{
	Mat_<uchar> img = imread(path, IMREAD_GRAYSCALE);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++)
		{
			if (img(i, j) * factor > 255)
				img(i, j) = 255;
			else
				if (img(i, j) * factor < 0)
					img(i, j) = 0;
				else
					img(i, j) = img(i, j) * factor;
		}
	}
	imshow("multiplicative", img);
	imwrite("images/multiplicative.bmp", img);
}
void lab01_create_color_image() {
	Mat_<Vec3b> img(256, 256);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++)
		{
			
			if (i < img.rows / 2 && j < img.cols / 2)
				img(i, j) = Vec3b(255, 255, 255);

			if (i < img.rows / 2 && j >= img.cols / 2)
				img(i, j) = Vec3b(0, 0, 255);

			if (i >= img.rows / 2 && j < img.cols / 2)
				img(i, j) = Vec3b(0, 255, 0);
			
			if (i >= img.rows / 2 && j >= img.cols / 2)
				img(i, j) = Vec3b(0, 255, 255);
		}
	}
	imshow("color", img);
}

void lab01_invert_matrix() {

	Mat_<float> matrix(3, 3, CV_32FC1);
	matrix(0, 0) = 1.0f;
	matrix(0, 1) = 2.0f;
	matrix(0, 2) = 3.0f;
	matrix(1, 0) = 4.0f;
	matrix(1, 1) = 5.0f;
	matrix(1, 2) = 6.0f;
	matrix(2, 0) = 7.0f;
	matrix(2, 1) = 8.0f;
	matrix(2, 2) = 10.0f;

	float determinant = matrix(0, 0) * (matrix(1, 1) * matrix(2, 2) - matrix(2, 1) * matrix(1, 2)) -
		matrix(0, 1) * (matrix(1, 0) * matrix(2, 2) - matrix(1, 2) * matrix(2, 0)) +
		matrix(0, 2) * (matrix(1, 0) * matrix(2, 1) - matrix(1, 1) * matrix(2, 0));

	if (abs(determinant) < 1e-6)
		printf("Determinant = 0\n");
	else
	{
		Mat_<float> inverse = matrix.inv();
		printf("Inverse:\n");
		for (int i = 0; i < matrix.rows; i++) {
			for (int j = 0; j < matrix.cols; j++)
				printf("%.3lf ", inverse(i, j));
			printf("\n");
		}
		Mat_<float> I3 = Mat::eye(3, 3, CV_32FC1);

		bool ok = true;

		Mat_<float> check = matrix * inverse;
		for (int i = 0; i < matrix.rows; i++) {
			for (int j = 0; j < matrix.cols; j++)
				if (abs(check(i, j) - I3(i, j)) < 1e-6)
					ok = false;
		}
		if (ok == true)
			printf("Inverse is correct\n");
		else
			printf("Inverse is incorrect\n");
	}
}
void lab02_RGB(char* path)
{
	Mat_<Vec3b> img = imread(path, IMREAD_COLOR);
	std::vector<Mat> rgb;
	split(img, rgb);

	imshow("red", rgb.at(2));
	imshow("green", rgb.at(1));
	imshow("blue", rgb.at(0));

	imshow("color", img);
	waitKey(0);
}
void lab02_color_to_grayscale(char* path)
{
	Mat_<Vec3b> img = imread(path, IMREAD_COLOR);
	Mat_<uchar> dst(img.rows, img.cols);
	for(int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)		
			dst(i, j) = (img(i, j)[0] + img(i, j)[1] + img(i, j)[2])/3;
	
	imshow("color", img);
	imshow("converted1", dst);
	waitKey(0);
}
void lab02_grayscale_to_bw(char* path)
{
	Mat_<uchar> img = imread(path, IMREAD_GRAYSCALE);
	Mat_<uchar> dst(img.rows, img.cols);
	int threshold;

	std::cout << "Enter threshold: ";
	std::cin >> threshold;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			if (img(i, j) < threshold)
				dst(i, j) = 0;
			else
				dst(i, j) = 255;
		}
	imshow("original", img);
	imshow("black&white", dst);	
	waitKey(0);
}
void lab02_rgb_to_hsv(char* path)
{
	Mat_<Vec3b> img = imread(path, IMREAD_COLOR);
	Mat_<uchar> h(img.rows, img.cols);
	Mat_<uchar> s(img.rows, img.cols);
	Mat_<uchar> v(img.rows, img.cols);

	float r, g, b;
	float H, S, V;
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			r = (float)img(i, j)[2] / 255;
			g = (float)img(i, j)[1] / 255;
			b = (float)img(i, j)[0] / 255;

			float M = max(r, max(g, b));
			float m = min(r, min(g, b));

			float C = M - m;
			//Value
			V = M;

			//Saturation
			if (V != 0)
				S = C / V;
			else
				S = 0;

			//Hue
			if (C != 0)
			{
				if (abs(M - r) <= 1e-6)
					H = 60 * (g - b) / C;
				if (abs(M - g) <= 1e-6)
					H = 120 + 60 * (b - r) / C;
				if (abs(M - b) <= 1e-6)
					H = 240 + 60 * (r - g) / C;
			}
			else
				H = 0;

			if (H < 0)
				H += 360;

			h(i, j) = H * 255 / 360;
			s(i, j) = S * 255;
			v(i, j) = V * 255;
		}

	imshow("H", h);
	imshow("S", s);
	imshow("V", v);

	std::vector<Mat_<uchar>> src = { h, s, v };

	Mat_<Vec3b> HSV_img;
	Mat_<Vec3b> RGB_merged;

	merge(src, HSV_img);
	
	cvtColor(HSV_img, RGB_merged, COLOR_HSV2BGR_FULL, 0);

	imshow("Back from HSV", RGB_merged);
}

bool isInside(Mat_<uchar> img, int i, int j)
{
	if (i < 0 || j < 0)
		return false;
	else
		if (i >= img.rows || j >= img.cols)
			return false;
		else
			return true;
}
void calculateHistogramWithBins(Mat_<uchar> img, int nrBins, int*& hist, float*& pdf)
{
	hist = new int[256];
	for (int i = 0; i < 256; i++)
		hist[i] = 0;
	
	for(int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)		
			hist[int(img(i,j)/ (256.0/nrBins))]++;

	pdf = new float[256];
	int M = img.rows * img.cols;

	for (int i = 0; i < 256; i++)
		pdf[i] = hist[i] * 1.0f / M;
}
void showHist(int* hist, int nrBins, int height)
{
	Mat_<uchar> histogram(height, nrBins);
	int max_hist = 0;
	for (int i = 0; i < nrBins; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];

	double scale = 1.0;
	scale = (double)height / max_hist;
	int baseline = height - 1;
	for(int i = 0; i < nrBins; i++)
		for (int j = 0; j <= baseline - cvRound(hist[i] * scale); j++)		
			histogram(j, i) = 0;
		
	imshow("Histogram", histogram);
}
int getNearest(int x, int y, int target) {
	if (target - x >= y - target)
		return y;
	else
		return x;
}
int getNearestElement(int arr[], int n, int target) {
	if (target <= arr[0])
		return arr[0];
	if (target >= arr[n - 1])
		return arr[n - 1];
	int left = 0, right = n, mid = 0;
	while (left < right) {
		mid = (left + right) / 2;
		if (arr[mid] == target)
			return arr[mid];
		if (target < arr[mid]) {
			if (mid > 0 && target > arr[mid - 1])
				return getNearest(arr[mid - 1], arr[mid], target);
			right = mid;
		}
		else {
			if (mid < n - 1 && target < arr[mid + 1])
				return getNearest(arr[mid], arr[mid + 1], target);
			left = mid + 1;
		}
	}
	return arr[mid];
}
void calculateMaxVector(Mat_<uchar> img, int*& maxList, int& n)
{
	int* hist;
	float* FDP;

	for (int i = 0; i < 256; i++)
		maxList[i] = 0;

	n = 1;
	calculateHistogramWithBins(img, 256, hist, FDP);

	int WH = 5;
	float TH = 0.0003f;
	int k = 0 + WH;
	while (k <= 255 - WH)
	{
		float sum = 0.0f;
		for (int x = k - WH; x <= k + WH; x++)
			sum += FDP[x];
		float v = sum / (2 * WH + 1);

		int ok = 1;

		for (int x = k - WH; x <= k + WH; x++)
			if (FDP[k] < FDP[x])
				ok = 0;

		if ((FDP[k] > v + TH) && ok)
			maxList[n++] = k;
		k++;
	}
	maxList[n++] = 255;
}
void binarisation(Mat_<uchar> img)
{
	imshow("Original", img);
	int* maxList = new int[256];
	int n;	

	calculateMaxVector(img, maxList, n);
	
	Mat_<uchar> binarised;
	binarised = img.clone();

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			for (int x = 0; x < n - 1; x++)
				binarised(i, j) = getNearestElement(maxList, n, binarised(i, j));

	imshow("Binarised", binarised);
	int* hist;
	float* pdf;
	calculateHistogramWithBins(binarised, 256, hist, pdf);
	showHist(hist, 256, 400);

}
uchar capValue(uchar toBeCapped, float toAdd)
{
	uchar temp = toBeCapped;
	if (toBeCapped + toAdd < 0)
		temp = 0;
	else
		if (toBeCapped + toAdd > 255)
			temp = 255;
		else
			temp = temp +  toAdd;
	return temp;
}
void floydSteinberg(Mat_<uchar> img)
{
	imshow("Original", img);
	int* maxList = new int[256];
	int n;
	Mat_<uchar> img1 = img.clone();
	
	calculateMaxVector(img, maxList, n);

	for (int i = 0; i < img1.rows; i++)
		for (int j = 0; j < img1.cols; j++)
		{
			uchar oldpixel = img1(i, j);
			uchar newpixel = getNearestElement(maxList, n, oldpixel);
			img1(i, j) = newpixel;
			float error = oldpixel * 1.0 - newpixel;

			if (isInside(img1, i, j + 1))
			{
				img1(i, j + 1) = capValue(img1(i, j + 1), 7 * error / 16);
			}				
			if (isInside(img1, i + 1, j - 1))
			{
				img1(i + 1, j - 1) = capValue(img1(i + 1, j - 1), 3 * error / 16);
			}
			if (isInside(img1, i + 1, j))
			{
				img1(i + 1, j) = capValue(img1(i + 1, j), 5 * error / 16);
			}
			if (isInside(img1, i + 1, j + 1))
			{
				img1(i + 1, j + 1) = capValue(img1(i + 1, j + 1), error / 16);
			}
		}
	imshow("Corrected", img1);
}

void onMouse(int event, int x, int y, int flags, void* param)
{
	Mat* img = (Mat*)param;
	
	if (event == EVENT_LBUTTONDOWN)
	{		
		Vec3b color = Vec3b(img->at<Vec3b>(y, x)[0], img->at<Vec3b>(y, x)[1], img->at<Vec3b>(y, x)[2]);
		if (color != Vec3b(255, 255, 255))
		{
			Mat_<uchar> indices;
			indices = Mat::zeros(img->rows, img->cols, CV_8UC1);
			Mat_<Vec3b> showProperties = Mat::zeros(img->rows, img->cols, CV_8UC3);
			showProperties.setTo(255);

			for (int i = 0; i < img->rows; i++)
				for (int j = 0; j < img->cols; j++)
					if (img->at<Vec3b>(i, j) == color)
						indices(i, j) = 1;

			int area = 0;
			for (int i = 0; i < indices.rows; i++)
				for (int j = 0; j < indices.cols; j++)
					if (indices(i, j) == 1)
					{
						area += 1;
						showProperties(i, j) = Vec3b(0, 255, 0);
					}
			imshow("Area", showProperties);
			showProperties.setTo(255);
			int centerX = 0, centerY = 0;

			for (int i = 0; i < indices.rows; i++)
				for (int j = 0; j < indices.cols; j++)
					if (indices(i, j) == 1)
					{
						centerX += i;
						centerY += j;
					}
			centerX /= area;
			centerY /= area;
			
			double numerator = 0, denominator = 0;
			double D1 = 0, D2 = 0;
			for (int i = 0; i < indices.rows; i++)
				for (int j = 0; j < indices.cols; j++)
					if (indices(i, j) == 1)
					{
						numerator += (i - centerX) * (j - centerY);
						D1 += pow((j - centerY), 2);
						D2 += pow((i - centerX), 2);
					}
			numerator *= 2;
			denominator = D1 - D2;

			double angle = atan2(numerator, denominator);
			angle /= 2;

			Point P1(centerY + cos(angle) * 50, centerX + sin(angle) * 50);
			Point P2(centerY + cos(angle + CV_PI) * 50, centerX + sin(angle + CV_PI) * 50);

			Point P3(centerY + cos(angle + CV_PI/2.0) * 50, centerX + sin(angle + CV_PI/2.0) * 50);
			Point P4(centerY + cos(angle - CV_PI/2.0) * 50, centerX + sin(angle - CV_PI/2.0) * 50);
			circle(showProperties, Point(centerY, centerX), 5, Scalar(0, 0, 0), -1);
			line(showProperties, P1, P2, Scalar(255, 0, 0), 2, -1);
			line(showProperties, P3, P4, Scalar(255, 0, 0), 2, -1);
			imshow("Center/Symmetry axis", showProperties);

			showProperties.setTo(255);
			
			int perimeter = 0;
			for (int i = 0; i < indices.rows; i++)
				for (int j = 0; j < indices.cols; j++)
					if (indices(i, j) == 1)
					{
						if (isInside(indices, i - 1, j - 1) && indices(i - 1, j - 1) == 0)
						{
							perimeter++;
							showProperties(i, j) = Vec3b(0, 0, 255);
						}
						if (isInside(indices, i, j - 1) && indices(i, j - 1) == 0)
						{
							perimeter++;
							showProperties(i, j) = Vec3b(0, 0, 255);
						}
						if (isInside(indices, i + 1, j - 1) && indices(i + 1, j - 1) == 0)
						{
							perimeter++;
							showProperties(i, j) = Vec3b(0, 0, 255);
						}
						if (isInside(indices, i - 1, j) && indices(i - 1, j) == 0)
						{
							perimeter++;
							showProperties(i, j) = Vec3b(0, 0, 255);
						}
						if (isInside(indices, i + 1, j) && indices(i + 1, j) == 0)
						{
							perimeter++;
							showProperties(i, j) = Vec3b(0, 0, 255);
						}
						if (isInside(indices, i - 1, j - 1) && indices(i - 1, j - 1) == 0)
						{
							perimeter++;
							showProperties(i, j) = Vec3b(0, 0, 255);
						}
						if (isInside(indices, i, j + 1) && indices(i, j + 1) == 0)
						{
							perimeter++;
							showProperties(i, j) = Vec3b(0, 0, 255);
						}
						if (isInside(indices, i, j + 1) && indices(i + 1, j + 1) == 0)
						{
							perimeter++;
							showProperties(i, j) = Vec3b(0, 0, 255);
						}
					}
			perimeter *= CV_PI / 4;

			imshow("Perimeter", showProperties);
			showProperties.setTo(255);

			double T = 4 * CV_PI * (area / pow(perimeter, 2));

			double R;
			int cmax = 0, cmin = indices.cols, rmax = 0, rmin = indices.rows;
			for (int i = 0; i < indices.rows; i++)
				for (int j = 0; j < indices.cols; j++)
					if (indices(i, j) == 1)
					{
						if (j < cmin)
							cmin = j;
						if (j > cmax)
							cmax = j;
						if (i < rmin)
							rmin = i;
						if (i > rmax)
							rmax = i;
					}
			R = (cmax - cmin + 1) * 1.0 / (rmax - rmin + 1);
			
			rectangle(showProperties, Rect(Point(cmin, rmin), Point(cmax, rmax)), Scalar(0, 255, 0), 2, -1);
			imshow("Aspect Ratio", showProperties);

			int* XProj = new int[img->rows]{ 0 }, * YProj = new int[img->cols]{ 0 };

			for(int i = 0; i < img->rows; i++)
				for(int j = 0; j < img->cols; j++)
					if (indices(i, j) == 1)
					{
						XProj[i]++;
						YProj[j]++;
					}
			Mat_<uchar> projections = Mat(img->rows, img->cols, CV_8UC1);
			projections.setTo(255);
			for(int i = 0; i < img->rows; i++)
				for (int j = 0; j < XProj[i]; j++)				
					projections(i, j) = 0;

			for (int i = 0; i < img->cols; i++)
				for (int j = 0; j < YProj[i]; j++)
					projections(j, i) = 128;

			printf("Area: %d\n", area);
			printf("Center coordinates: (%d,%d)\n", centerX, centerY);
			printf("Phi: %lf\n", angle);
			printf("Perimeter: %d\n", perimeter);
			printf("Thinness ratio: %lf\n", T);
			printf("Aspect ratio: %lf\n\n", R);
				
			imshow("X-Axis & Y-Axis projections", projections);
		}
	}	
}

void V4(Mat_<uchar> graph, int i, int j, Point*& neighbors)
{
	int di[4] = { -1,  0, 1, 0 };
	int dj[4] = { 0, -1, 0, 1 };
	for (int k = 0; k < 4; k++)
		if (isInside(graph, i + di[k], j + dj[k]))
			neighbors[k] = Point(i + di[k], j + dj[k]);
}
void V8(Mat_<uchar> graph, int i, int j, Point*& neighbors)
{
	int di[8] = { -1,  0, 1, 0, -1, 1, 1, -1 };
	int dj[8] = { 0, -1, 0, 1, -1, 1, -1, -1 };
	for (int k = 0; k < 8; k++)
		if (isInside(graph, i + di[k], j + dj[k]))
			neighbors[k] = Point(i + di[k], j + dj[k]);
}
void Vp(Mat_<uchar> graph, int i, int j, Point*& neighbors, int&size)
{
	int k = 0;
	if (isInside(graph, i, j - 1))
		neighbors[k++] = Point(i, j - 1);
	if (isInside(graph, i - 1, j - 1))
		neighbors[k++] = Point(i - 1, j - 1);
	if (isInside(graph, i - 1, j))
		neighbors[k++] = Point(i - 1, j);
	if (isInside(graph, i - 1, j + 1))
		neighbors[k++] = Point(i - 1, j + 1);

	size = k;
}

void createColorImg(const String name, Mat_<int> labels, int nrLabels)
{
	Mat_<Vec3b> colorImg = Mat::zeros(labels.rows, labels.cols, CV_8UC3);
	colorImg.setTo(255);

	std::default_random_engine gen;
	std::uniform_int_distribution<int> d(0, 255);
	uchar x = d(gen);

	std::vector<Vec3b> colors(nrLabels + 1);
	colors.at(0) = Vec3b(255, 255, 255);
	for (int label = 1; label <= nrLabels; label++)
	{		
		colors[label] = Vec3b(d(gen), d(gen), d(gen));
	}

	for(int i = 0; i < labels.rows; i++)
		for (int j = 0; j < labels.cols; j++)
		{
			int label = labels(i, j);		
			Vec3b& pixel = colorImg(i, j);
			pixel = colors[label];
		}
	imshow(name, colorImg);
}
void onePass(Mat_<uchar> img, int n)
{
	void (*function_ptr)(Mat_<uchar>, int, int, Point*&);
	function_ptr = V4;
	Point* neighbors = new Point[8]{ 0 };

	int label = 0;
	Mat_<int> labels = Mat::zeros(img.rows, img.cols, CV_8UC1);

	if (n == 8)
		function_ptr = V8;

	for(int i = 0; i < img.rows; i++)
		for(int j = 0 ; j < img.cols; j++)
			if (img(i, j) == 0 && labels(i, j) == 0)
			{
				label++;
				std::queue<Point> Q;
				labels(i, j) = label;
				Q.push(Point(i,j));
				while (!Q.empty())
				{
					Point q = Q.front();
					Q.pop();
					function_ptr(img, q.x, q.y, neighbors);
					for (int k = 0; k < sizeof(neighbors); k++)
					{
						if (img(neighbors[k].x, neighbors[k].y) == 0 && labels(neighbors[k].x, neighbors[k].y) == 0)
						{
							labels(neighbors[k].x, neighbors[k].y) = label;
							Q.push(Point(neighbors[k]));
						} 
					}
				}
			}
	createColorImg("One pass", labels, label);
}
void twoPass(Mat_<uchar> input)
{
	Point* neighbors = new Point[4]{ 0 };
	int size = 0;
	int label = 0;
	Mat_<int> labels = Mat_<int>::zeros(input.rows, input.cols);
	std::vector<std::vector<int>> edges(1000);
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
		{
			if (input(i, j) == 0 && labels(i, j) == 0)
			{
				std::vector<int> L(0);
				Vp(input, i, j, neighbors, size);

				for (int k = 0; k < size; k++)
					if (labels(neighbors[k].x, neighbors[k].y) > 0)
					{
						L.push_back(labels(neighbors[k].x, neighbors[k].y));

					}

				if (L.size() == 0)
				{
					label++;
					labels(i, j) = label;
					edges.resize(label + 1);
				}
				else
				{
					int x = L[0];
					for (int i = 1; i < L.size(); i++)
						if (L[i] < x)
							x = L[i];

					labels(i, j) = x;
					for (int y : L)
					{
						if (y != x)
						{
							edges[x].push_back(y);
							edges[y].push_back(x);
						}
					}
				}
			}
		}
	createColorImg("Two pass 1", labels, label);

	int newLabel = 0;
	std::vector<int> newLabels(label + 1);
	for (int i = 1; i <= label; i++)
	{
		if (newLabels[i] == 0)
		{
			newLabel++;
			std::queue<int> Q;
			newLabels[i] = newLabel;
			Q.push(i);
			while (!Q.empty())
			{
				int x = Q.front();
				Q.pop();
				for (int y : edges[x])
					if (newLabels[y] == 0)
					{
						newLabels[y] = newLabel;
						Q.push(y);
					}
			}
		}
	}
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++)
			labels(i, j) = newLabels[labels(i, j)];

	createColorImg("Two pass 2", labels, label);
}

void findContour(Mat_<uchar> img)
{
	Mat_<uchar> contour = Mat_<uchar>::zeros(img.rows, img.cols);
	contour.setTo(255);
	int dir = 7;
	std::vector<Point> pixels;
	std::vector<int> dirs;

	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	bool found = false;
	for (int i = 0; i < img.rows && !found; i++)
		for (int j = 0; j < img.cols && !found; j++)
			if (img(i, j) == 0)
			{
				pixels.push_back(Point(i, j));
				dirs.push_back(dir);				
				found = true;
			}

	bool ok = false;
	while (!ok)
	{
		int n = pixels.size();
		if (n > 2 && pixels.at(n - 2) == pixels.at(0) && pixels.at(n - 1) == pixels.at(1))
			ok = true;

		Point pixel = pixels.at(pixels.size() - 1);
		dir = dirs.at(pixels.size() - 1);

		int newDir;
		if (dir % 2 == 0)
			newDir = (dir + 7) % 8;
		else
			newDir = (dir + 6) % 8;

		for (int k = 0; k < 8; k++)
		{
			Point newPixel(pixel.x + di[(newDir + k) % 8], pixel.y + dj[(newDir + k) % 8]);

			if (isInside(img, newPixel.x, newPixel.y) && img(newPixel.x, newPixel.y) == 0)
			{
				pixels.push_back(newPixel);
				dirs.push_back((newDir + k) % 8);
				contour(newPixel.x, newPixel.y) = 0;
				break;
			}
		}		
	}
	imshow("Contour", contour);
	printf("Contour: \n");
	for (int dir : dirs)
	{
		printf("%d ", dir);
	}
	std::vector<int> derivate;
	
	printf("Derivative: \n");
	for (int i = 0; i < dirs.size() - 2; i++) {
		printf("%d ", (dirs.at(i + 1) - dirs.at(i) + 8) % 8);
		derivate.push_back((dirs.at(i + 1) - dirs.at(i) + 8) % 8);

	}
} 
void buildContour(Mat_<Vec3b> img)
{
	std::ifstream f("reconstruct.txt");
	int x, y, dirSize;
	f >> x >> y >> dirSize;
	Point startPoint(x, y);
	std::vector<int> dirs(dirSize);
	
	int di[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	int dir;
	f >> dir;
	for (int i = 0; i < dirSize; i++)
	{
		Point nextPoint(startPoint.x + di[dir], startPoint.y + dj[dir]);
		if(isInside(img, nextPoint.x, nextPoint.y))
			img(nextPoint.x, nextPoint.y) = Vec3b(0, 255, 0);
		startPoint = nextPoint;
		f >> dir;
	}
	imshow("Reconstructed", img);
}
Mat_<uchar> structingElement(int size, int type)
{
	Mat_<uchar> element = Mat_<uchar>::zeros(size, size);
	element.setTo(255);
	switch (type) {
		//type: square
	case 0: 
		element.setTo(0); 
		break;
		//type: cross
	case 1: 
		if (size % 2 == 1)
		{
			for (int i = 0; i < size; i++)
				element(i, size / 2) = 0;
			for (int i = 0; i < size; i++)
				element(size / 2, i) = 0;
		}
		else
		{
			for (int i = 0; i < size; i++)
				element(i, size / 2) = 0;
			for (int i = 0; i < size; i++)
				element(size / 2, i) = 0;
			for (int i = 0; i < size; i++)
				element(i, size / 2 - 1) = 0;
			for (int i = 0; i < size; i++)
				element(size / 2 - 1, i) = 0;
		}
	/*	element(1, 1) = 0;
		element(0, 1) = 0;
		element(1, 2) = 0;
		element(2, 1) = 0;
		element(1, 0) = 0;*/
		break;
		//type: disc
	case 2:
		for (int i = 0; i < size; i ++ )
			for (int j = 0; j < size; j++)
				if ((i - size/2) * (i - size / 2) + (j - size / 2) * (j - size / 2) <= size)
					element(i, j) = 0;
		/*element(0, 0) = 0;
		element(1, 1) = 0;
		element(2, 2) = 0;
		element(0, 2) = 0;
		element(2, 0) = 0;*/
		break;
		//type: diamond
	case 3:
		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
				if (abs(i + j) <= size)
					element(i, j) = 0;
		break;
	default: return element;
	}
	return element;
}

Mat_<uchar> dilation(Mat_ <uchar> img, Mat_<uchar> element)
{
	Mat_<uchar> output = Mat_<uchar>::zeros(img.rows, img.cols);
	output.setTo(255);

	for (int i = 0; i < img.rows; i++) 	
		for (int j = 0; j < img.cols; j++) 		
			if (img(i, j) == 0) 
			{
				for (int elementRows = 0; elementRows < element.rows; elementRows++)
					for (int elementCols = 0; elementCols < element.cols; elementCols++) {
						int i2 = i + elementRows - element.rows / 2;
						int j2 = j + elementCols - element.cols / 2;
						if(isInside(img, i2, j2))
						if (element(elementRows, elementCols) == 0) 
							output(i2, j2) = 0;						
					}				
			}
	imshow("Dilation", output);
	return output;
}
Mat_<uchar> erosion(Mat_<uchar> img, Mat_<uchar> element)
{
	Mat_<uchar> output = img.clone();

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			if (img(i, j) == 0)
			{
				bool check = true;
				for (int elementRows = 0; elementRows < element.rows; elementRows++)
					for (int elementCols = 0; elementCols < element.cols; elementCols++) {
						int i2 = i + elementRows - element.rows / 2;
						int j2 = j + elementCols - element.cols / 2;
						if(isInside(img, i2, j2))
						if (element(elementRows, elementCols) == 0 && img(i2, j2) == 255) 
							check = false;						
					}
				if (check == false)
					output(i, j) = 255;
			}	
	imshow("Erosion", output);
	return output;
}
Mat_<uchar> opening(Mat_<uchar> img, Mat_<uchar> element)
{
	Mat_<uchar> eroded = erosion(img, element);
	Mat_<uchar> openingImg = dilation(eroded, element);
	imshow("Opening", openingImg);
	return openingImg;
}
Mat_<uchar> closing(Mat_<uchar> img, Mat_<uchar> element)
{
	Mat_<uchar> dilated = dilation(img, element);
	Mat_<uchar> closingImg = erosion(dilated, element);
	imshow("Closing", closingImg);
	return closingImg;
}
Mat_<uchar> getContour(Mat_<uchar> img)
{
	Mat_<uchar> box = structingElement(3, 0);
	Mat_<uchar> output = Mat_<uchar>::zeros(img.rows, img.cols);
	Mat_<uchar> eroded = erosion(img, box);

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			if (img(i, j) == 0 && eroded(i, j) == 0)
				output(i, j) = 255;

	return output;
}
Mat_<uchar> intersection(Mat_<uchar> A, Mat_<uchar> B)
{
	Mat_<uchar> output = Mat_<uchar>::zeros(A.rows, B.cols);
	output.setTo(255);
	for (int i = 0; i < A.rows; i++)
		for (int j = 0; j < B.rows; j++)
			if (A(i, j) == 0 && B(i, j) == 0)
				output(i, j) = 0;
	return output;
}
Mat_<uchar> reunion(Mat_<uchar> A, Mat_<uchar> B)
{
	Mat_<uchar> output = Mat_<uchar>::zeros(A.rows, B.cols);
	output.setTo(255);
	for (int i = 0; i < A.rows; i++)
		for (int j = 0; j < B.rows; j++)
			if (A(i, j) == 0 || B(i, j) == 0)
				output(i, j) = 0;
	return output;
}
bool isIdentity(Mat_<uchar> A, Mat_<uchar> B)
{
	for (int i = 0; i < A.rows; i++)
		for (int j = 0; j < A.cols; j++)
			if (A(i, j) != B(i, j))
				return false;
	return true;
}
void fillRegion(Mat_<uchar> input)
{

	Mat_<uchar> xK, xK_pred, invertedInput, element, output;

	xK_pred = Mat_<uchar>::zeros(input.rows, input.cols);
	xK_pred.setTo(255);

	xK_pred(input.rows / 2, input.cols / 2) = 0;

	element = structingElement(3, 1);
	invertedInput = invert(input);

	bool finished;
	imshow("Filled contour", element);
	do {
		xK = intersection(dilation(xK_pred, element), invertedInput);
		finished = isIdentity(xK, xK_pred);
		xK_pred = xK.clone();
		
	} while (!finished);

	
	output = reunion(input, xK);
	imshow("Filled contour", output);
	waitKey(0);
}
int main()
{
	char* path = "Morphological_Op_Images/3_Open/mon1thr1_bw.bmp";
	char* path1 = "contours/gray_background.bmp";

	//Laborator_05
	Mat_<uchar> screen = imread(path, IMREAD_GRAYSCALE);
	//dilation(screen, structingElement(10, 1));
	//erosion(screen, structingElement(3, 1));
  //  opening(screen, structingElement(10, 1));
  //  closing(screen, structingElement(10, 1));
	//erosion(screen);
	//fillRegion(screen);
	waitKey(0);
	return 0;
}