// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <iostream>
using namespace std;
#include <random>;
#include <math.h>
#define PI 3.14159265358979323846


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}


//LAB1
//ex 2
void negative_image() {
	Mat img = imread("Images/cameraman.bmp",
		IMREAD_GRAYSCALE);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<uchar>(i, j) = 255 - img.at<uchar>(i, j);
		}
	}
	imshow("negative image", img);
	waitKey(0);
}

//ex 3
void factor_aditiv() {
	//m-am ajutat de functia de mai sus numita testNegativeImage
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				int var = val + 50;
				if (var < 0) {
					var = 0;
				}
				else if (var > 255) {
					var = 255;
				}
				dst.at<uchar>(i, j) = var;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("new image", dst);
		waitKey();
	}
}

//ex 4
void factor_multiplicativ() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				int var = val * 3;
				if (var < 0) {
					var = 0;
				}
				else if (var > 255) {
					var = 255;
				}
				dst.at<uchar>(i, j) = var;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imwrite("C:/Users/ARY/Desktop/imagine_rezultat.bmp", dst);
		imshow("input image", src);
		imshow("new image", dst);
		waitKey();
	}
}

//ex 5
void imagine_color(){
	Vec3b pixel;
	Mat dst(256, 256, CV_8UC3);
	for (int i = 0; i < 256; i++)
		for (int j = 0; j < 256; j++) 
		{
			pixel = dst.at< Vec3b>(i, j);
			if (i < 128 && j < 128)
				pixel[0] = pixel[1] = pixel[2] = 255;
			else if (i < 128 && j >= 128) 
			{
				pixel[0] = pixel[1] = 0;
				pixel[2] = 255;
			}
			else if (i >= 128 && j < 128) 
			{
				pixel[1] = 255;
				pixel[0] = pixel[2] = 0;
			}
			else if (i >= 128 && j >= 128) 
			{
				pixel[0] = 0;
				pixel[1] = pixel[2] = 255;
			}
			dst.at<Vec3b>(i, j) = pixel;
		}
	imshow("Image", dst);
	waitKey(0);
}

//ex 6
void inversa_matrice3x3() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) 
	{
		Mat_<float> Mat(3, 3);
		Mat_<float> inv(3, 3);
		Mat_<float> adj(3, 3);
		Mat(0, 0) = -1.0f;
		Mat(0, 1) = 2.0f;
		Mat(0, 2) = 1.0f;
		Mat(1, 0) = 3.0f;
		Mat(1, 1) = 0.0f;
		Mat(1, 2) = 2.0f;
		Mat(2, 0) = 4.0f;
		Mat(2, 1) = -1.0f;
		Mat(2, 2) = 2.0f;
		float det = (Mat(0, 0) * Mat(1, 1) * Mat(2, 2)) + (Mat(1, 0) * Mat(2, 1) * Mat(0, 2)) + (Mat(0, 1) * Mat(1, 2) * Mat(2, 0))
			- (Mat(0, 2) * Mat(1, 1) * Mat(2, 0)) - (Mat(1, 0) * Mat(0, 1) * Mat(2, 2)) - (Mat(2, 1) * Mat(1, 2) * Mat(0, 0));
		if (det == 0) 
		{
			printf("Matricea nu e inversabila\n");
			return;
		}
		float a11, a12, a13, a21, a22, a23, a31, a32, a33;
		a11 = Mat(1, 1) * Mat(2, 2) - Mat(1, 2) * Mat(2, 1);
		a12 = -1.0f * (Mat(1, 0) * Mat(2, 2) - Mat(1, 2) * Mat(2, 0));
		a13 = Mat(1, 0) * Mat(2, 1) - Mat(1, 1) * Mat(2, 0);
		a21 = -1.0f * (Mat(0, 1) * Mat(2, 2) - Mat(0, 2) * Mat(2, 1));
		a22 = Mat(0, 0) * Mat(2, 2) - Mat(0, 2) * Mat(2, 0);
		a23 = -1.0f * (Mat(0, 0) * Mat(2, 1) - Mat(0, 1) * Mat(2, 0));
		a31 = Mat(0, 1) * Mat(1, 2) - Mat(0, 2) * Mat(1, 1);
		a32 = -1.0f * (Mat(0, 0) * Mat(1, 2) - Mat(0, 2) * Mat(1, 0));
		a33 = Mat(0, 0) * Mat(1, 1) - Mat(0, 1) * Mat(1, 0);
		adj(0, 0) = a11;
		adj(1, 0) = a12;
		adj(2, 0) = a13;
		adj(0, 1) = a21;
		adj(1, 1) = a22;
		adj(2, 1) = a23;
		adj(0, 2) = a31;
		adj(1, 2) = a32;
		adj(2, 2) = a33;
		inv = 1.0f / det * adj;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++)
				printf("%f ", inv(i, j));
			printf("\n");
		}
		printf("\n");
	}
}

//LAB2
//ex 1
void imagineIn3Matrici() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat dst1(height, width, CV_8UC3);
		Mat dst2(height, width, CV_8UC3);
		Mat dst3(height, width, CV_8UC3);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				dst1.at<Vec3b>(i, j) = Vec3b(src.at<Vec3b>(i, j)[0], 0, 0);
				dst2.at<Vec3b>(i, j) = Vec3b(0, src.at<Vec3b>(i, j)[1], 0);
				dst3.at<Vec3b>(i, j) = Vec3b(0, 0, src.at<Vec3b>(i, j)[2]);
			}
		}
		imshow("input image", src);
		imshow("Red", dst3);
		imshow("Green", dst2);
		imshow("Blue", dst1);
		waitKey();
	}
}

//ex 2
void conversieRGBgray() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat dst(height, width, CV_8UC1);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				dst.at<uchar>(i, j) = (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2]) / 3;
			}
		}
		imshow("input image", src);
		imshow("grayscaled image", dst);
		waitKey();
	}
}

//ex 3
void grayscale_albnegru(int prag) {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst(height, width, CV_8UC1);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) > prag) {
					dst.at<uchar>(i, j) = 255;
				}
				else {
					dst.at<uchar>(i, j) = 0;
				}
			}
		}
		imshow("output image", dst);
		imshow("input image", src);
		waitKey();
	}
}

//ex 4
void RGB_HSV() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat Himg(height, width, CV_8UC1);
		Mat Simg(height, width, CV_8UC1);
		Mat Vimg(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) 
			{
				Vec3b pixel = src.at<Vec3b>(i, j);
				float R = (float)pixel[2] / 255;
				float G = (float)pixel[1] / 255;
				float B = (float)pixel[0] / 255;
				float M = max(max(R, G), B);
				float m = min(min(R, G), B);
				float C = M - m;
				Vimg.at<uchar>(i, j) = M * 255; //Value
				Simg.at<uchar>(i, j) = (M != 0) ? (C / M) * 255 : 0; //Saturation
				float H = 0.0f;
				if (C != 0.0f) 
				{   //Hue
					if (M == R) 
						H = 60.0f * (G - B) / C;
					if (M == G) 
						H = 120.0f + 60.0f * (B - R) / C;
					if (M == B) 
						H = 240.0f + 60.0f * (R - G) / C;
				}
				if (H < 0.0f) 
					H = H + 360.0f;
				Himg.at<uchar>(i, j) = H * 255.0f / 360.0f;
			}
		imshow("input image", src);
		imshow("H", Himg);
		imshow("S", Simg);
		imshow("V", Vimg);
		waitKey();
	}
}

//ex 5
bool isInside(const Mat& src, int x, int y) {
	return x >= 0 && y >= 0 && x < src.cols && y < src.rows;
}

bool isInside(int height, int width, int i, int j) {
	return i >= 0 && i < height && j >= 0 && j < width;
}

//LAB3 
//ex 1
void calcul_histograma(Mat src, int h[])
{
	memset(h, 0, 256*sizeof(int));
	int height = src.rows;
	int width = src.cols;
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			uchar k = src.at<uchar>(i, j);
			h[k]++;
		}
}
void afisare_histograma()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int h[256];
		calcul_histograma(src, h);
		for (int i = 0; i < 256; i++)
			printf("%d ", h[i]);
		printf("\n");
		imshow("input image", src);
		waitKey();
	}
}

//ex 2
void FDP(Mat src, float fdp[])
{
	memset(fdp, 0, 256 * sizeof(int));
	int height = src.rows;
	int width = src.cols;
	int M = height * width;
	int h[256] = {0};
	calcul_histograma(src, h);
	for (int i = 0; i < 256; i++)
		fdp[i] = h[i] / float(M);
}
void afisare_FDP()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		float fdp[256];
		FDP(src, fdp);
		for (int i = 0; i < 256; i++)
			printf("%d ", fdp[i]);
		printf("\n");
		imshow("input image", src);
		waitKey();
	}
}

//ex 3
void testShowHistogram()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int h[256];
		calcul_histograma(src, h);
		imshow("input image", src);
		showHistogram("histogram", h, 256, 200);
		waitKey();
	}
}

//ex 4
int* hist_acumulatoare(Mat img, int m) {
	int* h = (int*)calloc(m, sizeof(int));
	int height = img.rows;
	int width = img.cols;
	int size = ceil(256/(float)m);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int k = ceil(img.at<uchar>(i, j)/(float) size);
			h[k]++;
		}
	}
	return h;
}

//ex 5
int* h(Mat src) 
{
	int* h = (int*)calloc(256, sizeof(int));
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			h[src.at<uchar>(i, j)]++;
	return h;
}

float* f(Mat img) 
{
	int M = img.rows*img.cols;
	int* histogram = h(img);
	float* fdp = (float*)calloc(256, sizeof(float));
	for (int i = 0; i < 256; i++)
		fdp[i] = (float)histogram[i]/M;
	return fdp;
}

void det_praguri_multiple() 
{
	char fname[MAX_PATH];
	float TH = 0.0003;
	int WH = 5;
	int m = 1;
	while (openFileDlg(fname)) 
	{
		int h[256] = { 0 };
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = Mat(src.rows, src.cols, CV_LOAD_IMAGE_GRAYSCALE);
		float* FDP = f(src);
		for (int i = WH; i < 255-WH; i++) 
		{
			float media = 0.0f;
			for (int k = i - WH; k <= i + WH; k++)
				media += FDP[k];
			if (FDP[i] > media/(2*WH+1) + TH) 
			{
				bool max = true;
				for (int j = i - WH; j <= i + WH; j++) 
				{
					if (FDP[i] < FDP[j]) 
					{
						max = false;
						break;
					}
				}
				if (max) 
				{
					h[m] = i;
					m++;
				}
			}
		}
		h[0] = 0;
		h[255] = 255;
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				int min = 255, minn = 255;
				uchar g = src.at<uchar>(i, j);
				for (int k = 0; k < 256; k++)
					if (abs(g - h[k]) < minn) 
					{
						minn = abs(g - h[k]);
						min = h[k];
					}
				dst.at<uchar>(i, j) = min;
			}
		}
		imwrite("C:/Users/ARY/Desktop/OpenCVApplication-VS2017_OCV340_basic/Images/histograma_praguri_multiple.bmp", dst);
		imshow("input image", src);
		imshow("output image", dst);
		waitKey();
	}
}

//ex 6
void FloydSteinberg() 
{
	char fname[MAX_PATH];
	float TH = 0.0003;
	int WH = 5;
	int m = 1;
	while (openFileDlg(fname))
	{
		int h[256] = { 0 };
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		Mat dst = Mat(src.rows, src.cols, CV_LOAD_IMAGE_GRAYSCALE);
		float* FDP = f(src);
		for (int i = WH; i < 255 - WH; i++)
		{
			float media = 0.0f;
			for (int k = i - WH; k <= i + WH; k++)
				media += FDP[k];
			if (FDP[i] > media / (2 * WH + 1) + TH)
			{
				bool max = true;
				for (int j = i - WH; j <= i + WH; j++)
				{
					if (FDP[i] < FDP[j])
					{
						max = false;
						break;
					}
				}
				if (max)
				{
					h[m] = i;
					m++;
				}
			}
		}
		h[0] = 0;
		h[255] = 255;
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				int min = 255, minn = 255;
				uchar oldpixel = src.at<uchar>(i, j);
				for (int k = 0; k < 256; k++) 
					if (abs(oldpixel - h[k]) < minn) 
					{
						minn = abs(oldpixel - h[k]);
						min = h[k];
					}
				uchar newpixel = min;
				dst.at<uchar>(i, j) = min;
				uchar error = oldpixel - newpixel;
				if (isInside(dst, i, j + 1))
					dst.at<uchar>(i, j + 1) += 7*error/16;
				if (isInside(dst, i + 1, j - 1))
					dst.at<uchar>(i + 1, j - 1) += 3*error/16;
				if (isInside(dst, i + 1, j))
					dst.at<uchar>(i + 1, j) += 5*error/16;
				if (isInside(dst, i + 1, j + 1))
					dst.at<uchar>(i + 1, j + 1) += error/16;
			}
		}
		imshow("input image", src);
		imshow("output image", dst);
		waitKey();
	}
}

//LAB4
//ex 1
int aria(Mat src) 
{
	int ai = 0;
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			if (src.at<uchar>(i, j) == 0)
				ai++;
	return ai;
}

void centru_de_masa(Mat src, int* r, int* c) 
{
	*r = 0, *c = 0;
	Mat dst = src.clone();
	int ai = aria(src);
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			if (src.at<uchar>(i, j) == 0) 
			{
				*c += j;
				*r += i;
			}
	*r /= ai;
	*c /= ai;
	dst.at<uchar>(*r, *c) = 127;
	imshow("centrul de masa", dst); //b.2
}

void axa_de_alungire(Mat src) 
{
	float sumR = 0, sumC = 0, tan = 0;
	int r, c, L=50;
	centru_de_masa(src, &r, &c);
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			if (src.at<uchar>(i, j) == 0) 
			{
				sumR += (i - r)*(j - c);
				sumC += (j - c)*(j - c) - (i - r)*(i - r);
			}
	sumR *= 2;
	float phi = atan2(sumR, sumC) / 2;
	float ra = r + L * sin(phi);
	float ca = c + L * cos(phi);
	Mat dst = src.clone();
	line(dst, Point(ca, ra), Point(c, r), 127, 2);
	imshow("axa de alungire", dst); //b.3
}

int perimetrul(Mat src) 
{
	int p = 0;
	Mat dst = Mat(src.rows, src.cols, CV_LOAD_IMAGE_GRAYSCALE);
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			if (src.at<uchar>(i, j) == 0) 
			{
				if (isInside(src, i - 1, j) && src.at<uchar>(i - 1, j) != 0) {
					p++;
					dst.at<uchar>(i, j) = 0;
				}
				else if (isInside(src, i + 1, j) && src.at<uchar>(i + 1, j) != 0) {
					p++;
					dst.at<uchar>(i, j) = 0;
				}
				else if (isInside(src, i, j + 1) && src.at<uchar>(i, j + 1) != 0) {
					p++;
					dst.at<uchar>(i, j) = 0;
				}
				else if (isInside(src, i, j - 1) && src.at<uchar>(i, j - 1) != 0) {
					p++;
					dst.at<uchar>(i, j) = 0;
				}
			}
	imshow("contur", dst); //b.1
	return p;
}

void proiectiile(Mat src) //c.
{
	Mat dst1 = Mat(src.rows, src.cols, CV_8UC1);
	Mat dst2 = Mat(src.rows, src.cols, CV_8UC1);
	for (int i = 0; i < src.rows; i++) 
	{
		int r = 0;
		for (int j = 0; j < src.cols; j++)
			if (src.at<uchar>(i, j) == 0)
				r++;
		for (int j = 0; j < src.cols; j++)
			if (j > r)
				dst1.at<uchar>(i, j) = 255;
			else
				dst1.at<uchar>(i, j) = 0;
	}
	for (int j = 0; j < src.cols; j++) {
		int c = 0;
		for (int i = 0; i < src.rows; i++)
			if (src.at<uchar>(i, j) == 0)
				c++;
		for (int i = 0; i < src.rows; i++)
			if (i > c)
				dst2.at<uchar>(i, j) = 255;
			else
				dst2.at<uchar>(i, j) = 0;
	}
	imshow("proiectie hi(r)", dst1);
	imshow("proiectie vi(c)", dst2);
}

void onMouse(int event, int x, int y, int flags, void* param) //a.
{
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN) 
	{
		Mat dst = Mat((*src).rows, (*src).cols, CV_8UC1);
		for (int i = 0; i < (*src).rows; i++)
			for (int j = 0; j < (*src).cols; j++)
			{
				Vec3b s1 = (*src).at<Vec3b>(y, x);
				Vec3b s2 = (*src).at<Vec3b>(i, j);
				if (s2[0] == s1[0] && s2[1] == s1[1] && s2[2] == s1[2])
					dst.at<uchar>(i, j) = 0;
				else
					dst.at<uchar>(i, j) = 255;
			}

		int A = aria(dst);
		printf("Aria = %d\n", A);

		int ci, ri;
		centru_de_masa(dst, &ri, &ci);
		printf("Centrul de masa = %d, %d\n", ri, ci);

		axa_de_alungire(dst);

		int P = perimetrul(dst);
		printf("Perimetrul = %d\n", P);

		float T = 4 * PI * (A / (P*P));
		printf("Factorul de subtiere = %f\n", T);

		Mat e = dst.clone();
		int cmax = 0, cmin = INT_MAX, rmax = 0, rmin = INT_MAX;
		for (int i = 0; i < (*src).rows; i++)
			for (int j = 0; j < (*src).cols; j++)
				if ((dst).at<uchar>(i, j) == 0) 
				{
					if (i < rmin) rmin = i;
					if (j < cmin) cmin = j;
					if (i > rmax) rmax = i;
					if (j > cmax) cmax = j;
				}
		line(e, Point(cmin, rmin), Point(cmax, rmin), 127, 2);
		line(e, Point(cmin, rmax), Point(cmax, rmax), 127, 2);
		line(e, Point(cmin, rmin), Point(cmin, rmax), 127, 2);
		line(e, Point(cmax, rmin), Point(cmax, rmax), 127, 2);
		float R = (float)(cmax-cmin+1) / (rmax-rmin+1);
		printf("Elongatia = %f\n", R);
		imshow("Elongatia", e);

		proiectiile(dst);

		waitKey();
	}
}

//LAB5
//ex 2
Mat_<Vec3b> culori_pentru_etichete(int height, int width, Mat_<int> labels) 
{
	int maxLabel = 0;
	Mat_<Vec3b> dst(height, width);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			if (labels(i, j) > maxLabel)
				maxLabel = labels(i, j);
	std::default_random_engine gen;
	std::uniform_int_distribution<int> d(0, 255);
	std::vector<Vec3b> colors(maxLabel+1);
	for (int i = 0; i <= maxLabel; i++) 
		colors.at(i) = Vec3b(d(gen), d(gen), d(gen));
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) 
		{
			int label = labels(i, j);
			if (label > 0)
				dst(i, j) = colors.at(labels(i, j));
			else
				dst(i, j) = Vec3b(255, 255, 255);
		}
	return dst;
}

//ex 1
void etichetare_BFS() 
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) 
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat_<int> labels = Mat_<int>(src.rows, src.cols, 0);
		int label = 0;
		int dx[] = {-1,0,1,0,-1,1,-1,1};
		int dy[] = {0,-1,0,1,1,-1,-1,1};
		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++)
				if (src.at<uchar>(i, j) == 0 && labels(i, j) == 0) 
				{
					label++;
					labels(i, j) = label;
					std::queue<Point> Q;
					Q.push(Point(j, i));
					while (!Q.empty()) 
					{
						Point q = Q.front();
						Q.pop();
						for (int k = 0; k < 8; k++) //vecinatate 8
						{
							int newX = q.x + dx[k];
							int newY = q.y + dy[k];
							if (isInside(src.rows, src.cols, newY, newX)) {
								uchar neighbor = src.at<uchar>(newY, newX);
								if (neighbor == 0 && labels(newY, newX) == 0) {
									labels(newY, newX) = label;
									Q.push(Point(newX, newY));
								}
							}
						}
					}
				}
		Mat_<Vec3b> bfs = culori_pentru_etichete(src.rows, src.cols, labels);
		imshow("input image", src);
		imshow("output image", bfs);
		waitKey(0);
	}
}

//ex 3
void algoritmul_doua_treceri() 
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) 
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat_<int> labels = Mat_<int>(src.rows, src.cols, 0);
		int* newlabels = (int*)calloc(src.rows*src.cols+1, sizeof(int));
		int label = 0, newlabel = 0;
		int dx[] = {0,-1,-1,-1}; //-1,0,1,0
		int dy[] = {-1,-1,0,1}; //0,-1,0,1
		std::vector<std::vector<int>> edges;
		edges.resize(src.rows*src.cols+1);
		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++)
				if (src.at<uchar>(i, j) == 0 && labels(i, j) == 0) 
				{
					std::vector<int> L;
					for (int k = 0; k < 4; k++) //vecinatate 4
					{
						int newX = i + dx[k];
						int newY = j + dy[k];
						if (isInside(src.rows, src.cols, newX, newY))
							if (labels(newX, newY) > 0)
								L.push_back(labels(newX, newY));
					}
					if (L.size() == 0) 
					{
						label++;
						labels(i, j) = label;
					}
					else 
					{
						int min = *min_element(L.begin(), L.end());
						labels(i, j) = min;
						for (int k : L)
							if (k != min) {
								edges[min].push_back(k);
								edges[k].push_back(min);
							}
					}
				}
		for (int j = 1; j <= label; j++)
			if (newlabels[j] == 0) 
			{
				newlabel++;
				newlabels[j] = newlabel;
				std::queue<int> Q;
				Q.push(j);
				while (!Q.empty()) {
					int x = Q.front();
					Q.pop();
					for (int y : edges[x])
						if (newlabels[y] == 0) 
						{
							newlabels[y] = newlabel;
							Q.push(y);
						}
				}
			}
		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++)
				labels(i, j) = newlabels[labels(i, j)];
		Mat_<Vec3b> culori = culori_pentru_etichete(src.rows, src.cols, labels);
		imshow("input image", src);
		imshow("output image", culori);
		waitKey(0);
	}
}

//LAB6
//ex 1 + ex 2
void urmarire_contur()
{
	char fname[MAX_PATH];
	int dx[] = { 1,1,0,-1,-1,-1,0,1 };
	int dy[] = { 0,-1,-1,-1,0,1,1,1 };
	int cod[10000], derivata[10000];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat contur = Mat(src.rows, src.cols, CV_8UC1, Scalar(255));
		Point P0 = Point(0, 0);
		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++)
				if (src.at<uchar>(i, j) == 0)
				{
					P0.x = j;
					P0.y = i;
					j = src.cols + 1;
					i = src.rows + 1;
				}
		Point P1 = Point(P0.x, P0.y), Pn = Point(P0.x, P0.y), Pn1 = Point(P0.x, P0.y);
		int dir = 7, contor = 0, k = 0, ok = 0;
		contur.at<uchar>(P0.y, P0.x) = 0;

		while (!((Pn.x == P1.x && Pn.y == P1.y) && (contor >= 2)))
		{
			contor++;
			if (dir % 2 == 0)
				dir = (dir + 7) % 8;
			else
				dir = (dir + 6) % 8;

			while (src.at<uchar>(Pn.y + dy[dir], Pn.x + dx[dir]) > 0)
				dir = (dir + 1) % 8;

			if (!ok)
			{
				P1.x = Pn.x + dx[dir];
				P1.y = Pn.y + dy[dir];
				ok = 1;
			}
			Pn.x = Pn.x + dx[dir];
			Pn.y = Pn.y + dy[dir];
			contur.at<uchar>(Pn.y, Pn.x) = 0;
			cod[k] = dir;
			if (k >= 1)
				derivata[k-1] = (cod[k] - cod[k-1] + 8) % 8;
			k++;
		}
		printf("Codul inlantuit: ");
		for (int i = 0; i < k; i++)
			printf("%d ", cod[i]);
		printf("\nDerivata: ");
		for (int i = 0; i < k - 1; i++)
			printf("%d ", derivata[i]);
		imshow("input image", src);
		imshow("Contur", contur);
		waitKey();
	}
}

//ex 3
void excellent() {

	Mat src = imread("./Images/gray_background.bmp", IMREAD_GRAYSCALE);
	FILE* fp = fopen("./Images/reconstruct.txt", "r");
	int dx[] = { 1,1,0,-1,-1,-1,0,1 };
	int dy[] = { 0,-1,-1,-1,0,1,1,1 };
	int x, y, n, dir;
	fscanf(fp, "%d %d", &y, &x);
	Point P = Point(x, y);
	src.at<uchar>(y, x) = 0;
	fscanf(fp, "%d", &n);
	for (int i = 0; i < n; i++) 
	{
		fscanf(fp, "%d", &dir);
		x = P.x + dx[dir];
		y = P.y + dy[dir];
		P = Point(x, y);
		src.at<uchar>(y, x) = 0;
	}
	imshow("reconstructed image", src);
	waitKey(0);
}

//LAB7
//ex 1
Mat element_structural() 
{
	Mat B(3, 3, CV_8UC1, Scalar(255));
	B.at<uchar>(0, 1) = 0;
	B.at<uchar>(1, 1) = 0;
	B.at<uchar>(1, 0) = 0;
	B.at<uchar>(1, 2) = 0;
	B.at<uchar>(2, 1) = 0;
	return B;
}

Mat dilatare(Mat src, Mat B) 
{
	Mat dst = src.clone();
	int height = src.rows;
	int width = src.cols;
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			if (src.at<uchar>(i, j) == 0) 
				for (int si = 0; si < B.rows; si++)
					for (int sj = 0; sj < B.cols; sj++)
						if (B.at<uchar>(si, sj) == 0) 
						{
							int newI = i + si - B.rows/2;
							int newJ = j + sj - B.cols/2;
							if (isInside(height, width, newI, newJ))
								dst.at<uchar>(newI, newJ) = 0;
						}
	return dst;
}

Mat eroziune(Mat src, Mat B) 
{
	Mat dst = src.clone();
	int height = src.rows;
	int width = src.cols;
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			if (src.at<uchar>(i, j) == 0)
				for (int si = 0; si < B.rows; si++)
					for (int sj = 0; sj < B.cols; sj++)
						if (B.at<uchar>(si, sj) == 0) 
						{
							int newI = i + si - B.rows / 2;
							int newJ = j + sj - B.cols / 2;
							if (isInside(height, width, newI, newJ))
								if (src.at<uchar>(newI, newJ) == 255)
									dst.at<uchar>(i, j) = 255;
						}
	return dst;
}

void deschidere() 
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) 
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat B = element_structural();
		Mat er = eroziune(src, B);
		Mat dil = dilatare(er, B);
		imshow("input image", src);
		imshow("deschidere", dil);
		waitKey(0);
	}
}

void inchidere() 
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) 
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat B = element_structural();
		Mat dil = dilatare(src, B);
		Mat er = eroziune(dil, B);
		imshow("input image", src);
		imshow("inchidere", er);
		waitKey(0);
	}
}

//ex 2
void dilatare_eroziune(int n) 
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) 
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat dst1 = src.clone(), dst2 = src.clone();
		Mat B = element_structural();
		while (n > 0) 
		{
			Mat copie1 = dilatare(dst1, B);
			Mat copie2 = eroziune(dst2, B);
			dst1 = copie1.clone();
			dst2 = copie2.clone();
			n--;
		}
		imshow("input image", src);
		imshow("dilatare", dst1);
		imshow("eroziune", dst2);
		waitKey(0);
	}
}

//ex 3
void extragere_contur() 
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) 
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat dst = src.clone();
		Mat B = element_structural();
		int height = src.rows;
		int width = src.cols;
		Mat er = eroziune(src, B);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				if (src.at<uchar>(i, j) == er.at<uchar>(i, j))
					dst.at<uchar>(i, j) = 255;
				else
					dst.at<uchar>(i, j) = 0;
		imshow("input image", src);
		imshow("Contur", dst);
		waitKey(0);
	}
}

//ex 4
bool imagini_egale(Mat mat1, Mat mat2) 
{
	for (int i = 0; i < mat1.rows; i++)
		for (int j = 0; j < mat1.cols; j++)
			if (mat1.at<uchar>(i, j) != mat2.at<uchar>(i, j))
				return false;
	return true;
}

void umplere_regiuni() 
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) 
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;

		Mat B = element_structural();
		Mat dst = Mat(height, width, CV_8UC1, Scalar(255));
		Mat copie = Mat(height, width, CV_8UC1, Scalar(255));
		Mat neg = Mat(height, width, CV_8UC1);
		dst.at<uchar>(height/2, width/2) = 0;

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				neg.at<uchar>(i, j) = (src.at<uchar>(i, j) == 0) ? 255 : 0;

		while (1) 
		{
			Mat d = dilatare(dst, B);
			for (int i = 0; i < height; i++)
				for (int j = 0; j < width; j++)
					if (d.at<uchar>(i, j) == 0 && neg.at<uchar>(i, j) == 0)
						copie.at<uchar>(i, j) = 0;
					else copie.at<uchar>(i, j) = 255;
			if (imagini_egale(dst, copie))
				break;
			dst = copie.clone();
		}
		imshow("input image", src);
		imshow("output image", dst);
		waitKey(0);
	}
}

//LAB8
//ex 1
int* Himg(Mat src) 
{
	int* h = (int*)calloc(256, sizeof(int));
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) 
			h[src.at<uchar>(i, j)]++;
	return h;
}

float* FDPimg(Mat src) 
{
	int M = src.rows * src.cols;
	int* h = Himg(src);
	float* fdp = (float*)calloc(256, sizeof(float));
	for (int k = 0; k < 256; k++)
		fdp[k] = (float)h[k] / M;
	return fdp;
}

int L = 255;
float media(Mat src) 
{
	float media = 0.0f;
	float* p = FDPimg(src);
	for (int g = 0; g <= L; g++)
		media += g * p[g];
	return media;
}

float deviatia_standard(Mat src) 
{
	float* p = FDPimg(src);
	float m = media(src);
	float dev = 0.0f;
	for (int g = 0; g <= L; g++)
		dev += ((g-m) * (g-m)) * p[g];
	return sqrt(dev);
}

int* histograma_cumulativa(Mat src) 
{
	int* h = Himg(src);
	int C[256] = {0};
	for (int g = 0; g <= L; g++)
		for (int j = 0; j < g; j++)
			C[g] += h[j];
	return C;
}

void afisare() 
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) 
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		float m = media(src);
		float d = deviatia_standard(src);
		int* h = Himg(src);
		int* c = histograma_cumulativa(src);
		printf("Media = %f\n", m);
		printf("Deviatia standard = %f\n", d);
		imshow("input image", src);
		showHistogram("histograma", h, 255, 255);
		showHistogram("histograma cumulativa", c, 255, 255);
		waitKey();
	}
}

//ex 2
float media_interval(int* h, int a, int b)
{
	int N = 0;
	float m = 0.0f;
	for (int g = a; g <= b; g++) 
	{
		m += g * h[g];
		N += h[g];
	}
	m /= (float)N;
	return m;
}

void binarizare_automata_globala() 
{
	float eroare = 0.1f;
	char fname[MAX_PATH];
	int Imin = 0, Imax = L, T, Tk = 0;
	while (openFileDlg(fname)) 
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int* h = Himg(src);
		Mat dst(src.rows, src.cols, CV_8UC1);
		for (int i = 0; i <= L; i++)
			if (h[i] >= 0)
			{
				Imin = i;
				break;
			}
		for (int i = Imin+1; i <= L; i++)
			if (h[i] >= 0)
				Imax = i;
		T = (Imin + Imax) / 2;
		do {
			float mG1 = media_interval(h, Imin, T);
			float mG2 = media_interval(h, T+1, Imax);
			Tk = T;
			T = (mG1 + mG2) / 2;
		} while (abs(T - Tk) > eroare);
		
		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++)
				if (src.at<uchar>(i, j) < T)
					dst.at<uchar>(i, j) = 0;
				else
					dst.at<uchar>(i, j) = 255;
		printf("Pragul = %d\n", T);
		imshow("input image", src);
		imshow("output image", dst);
		waitKey();
	}
}

//ex 3
Mat_<uchar> negativul_imaginii(Mat_<uchar> src) 
{
	Mat_<uchar> dst = Mat(src.rows, src.cols, CV_8UC1);
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			dst(i, j) = 255 - src(i, j);
	return dst;
}

void latirea_ingustarea_histogramei(int goutMIN, int goutMAX) 
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) 
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat dst = Mat(src.rows, src.cols, CV_8UC1);
		int* h1 = (int*)calloc(256, sizeof(int));
		int ginMIN = 256, ginMAX = -1;
		calcul_histograma(src, h1);
		imshow("Imaginea initiala", src);
		showHistogram("Histograma initiala", h1, 255, 255);
		for (int i = 0; i < 256; i++) 
		{
			if (h1[i] != 0 && ginMIN > i)
				ginMIN = i;
			if (h1[i] != 0 && ginMAX < i)
				ginMAX = i;
		}
		float r = (float)(goutMAX - goutMIN) / (ginMAX - ginMIN);
		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++) 
			{
				uchar gout = goutMIN + (src.at<uchar>(i, j) - ginMIN) * r;
				if (gout < 0) gout = 0;
				if (gout > 255) gout = 255;
				dst.at<uchar>(i, j) = gout;
			}
		int* h2 = (int*)calloc(256, sizeof(int));
		calcul_histograma(dst, h2);
		imshow("Imaginea noua", dst);
		showHistogram("Histograma noua", h2, 255, 255);
		waitKey(0);
	}
}

Mat_<uchar> corectia_gamma(Mat_<uchar> src, float gamma) 
{
	Mat_<uchar> dst = Mat(src.rows, src.cols, CV_8UC1);
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) 
		{
			float gout = L * pow(((float)src(i, j) / L), gamma);
			if (gout >= 255)
				dst(i, j) = 255;
			else {
				if (gout < 0) dst(i, j) = 0;
				else 
					dst(i, j) = gout;
			}
		}
	return dst;
}

void modificarea_luminozitatii(int offset)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int* H = h(src);
		imshow("Imaginea initiala", src);
		showHistogram("Histograma initiala", H, 255, 255);
		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++)
			{
				int gout = src.at<uchar>(i, j) + offset;
				if (gout < 0) gout = 0;
				if (gout > 255) gout = 255;
				src.at<uchar>(i, j) = (uchar)gout;
			}
		H = h(src);
		showHistogram("Histograma noua", H, 255, 255);
		imshow("Imaginea noua", src);
		waitKey();
	}
}

//ex 4
float* FDPC(Mat src)
{
	float* FDP = f(src);
	float* FDPC = (float*)calloc(256, sizeof(float));
	FDPC[0] = FDP[0];
	for (int g = 1; g < 256; g++)
		FDPC[g] = FDPC[g-1] + FDP[g];
	return FDPC;

}

void egalizarea_histogramei()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int gInMin = 256, gInMax = -1;
		int* H = h(src);
		float* CPDF = FDPC(src);
		showHistogram("Histograma initiala", H, 255, 255);
		imshow("Imaginea initiala", src);
		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++)
			{
				uchar gout = 255 * CPDF[src.at<uchar>(i, j)];
				if (gout < 0) gout = 0;
				if (gout > 255) gout = 255;
				src.at<uchar>(i, j) = gout;
			}
		H = h(src);
		showHistogram("Histograma egalizata", H, 255, 255);
		imshow("Dupa egalizarea histogramei", src);
		waitKey();
	}
}

//LAB9
//ex 1



int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n\n");

		printf(" 10 - L1 Negative Image \n");
		printf(" 11 - L1 Schimba nivelele de gri cu un factor aditiv \n");
		printf(" 12 - L1 Schimba nivelele de gri cu un factor multiplicativ\n");
		printf(" 13 - L1 Imagine color de dimensiune 256x256 impartita in 4 cadrane de culori diferite\n");
		printf(" 14 - L1 Matrice 3x3 de tip float si inversa ei\n\n");

		printf(" 15 - L2 Copiaza canalele RGB ale unei imagini in 3 matrice\n");
		printf(" 16 - L2 Conversie RGB - grayscale\n");
		printf(" 17 - L2 Conversie grayscale - alb-negru\n");
		printf(" 18 - L2 Conversie RGB la HSV\n");
		printf(" 19 - L2 Verifica daca pozitia indicata e inauntrul imaginii\n\n");

		printf(" 20 - L3 Calculati histograma pt o imagine grayscale\n");
		printf(" 21 - L3 Calculati FDP\n");
		printf(" 22 - L3 Afisati histograma calculata folosind functia din laborator\n");
		printf(" 23 - L3 Calculati histograma folosind un nr redus de acumulatoare m<=256\n");
		printf(" 24 - L3 Algoritmul de reducere a nivelurilor de gri\n");
		printf(" 25 - L3 Floyd-Steinberg\n\n");

		printf(" 26 - L4 Pentru un obiect binar selectat prin click, se calculeaza mai multe trasaturi geometrice\n\n");

		printf(" 27 - L5 Algoritmul de traversare in latime si etichetarea obiectelor\n");
		printf(" 28 - L5 Algoritmul de etichetare cu doua treceri\n\n");

		printf(" 29 - L6 Algoritmul de urmarire a conturului si codul inlantuit\n");
		printf(" 30 - L6 Reconstruire contur obiect peste o imagine, cunoscand punctul de start si codul inlantuit\n\n");
		
		printf(" 31 - L7 Operatii morfologice pe imagini binare\n");
		printf(" 32 - L7 Deschidere\n");
		printf(" 33 - L7 Inchidere\n");
		printf(" 34 - L7 Algoritmul de extragere a conturului\n");
		printf(" 35 - L7 Algoritmul de umplere a regiunilor\n\n");

		printf(" 36 - L8 Proprietati statistice ale imaginilor de intensitate\n");
		printf(" 37 - L8 Determinare automata a pragului de binarizare\n");
		printf(" 38 - L8 Negativul imaginii\n");
		printf(" 39 - L8 Modificarea contrastului - latirea/ingustarea histogramei\n");
		printf(" 40 - L8 Corectia gamma\n");
		printf(" 41 - L8 Modificarea luminozitatii\n");
		printf(" 42 - L8 Algoritmul de egalizare a histogramei\n\n");

		//printf(" 19 - L4 Aria, centrul de masa, axa de alungire, perimetrul, factorul de subtiere si elongatia\n");
		//printf(" 20 - L5 Algoritmul de traversare in latime pt etichetarea componentelor conexe\n");
		printf(" 19 - L6 Algoritmul de urmarire a conturului\n");
		printf(" 20 - L6 Reconstruieste conturul unui obiect\n");
		printf(" 21 - L4 EX1\n");
		printf(" 22 - L8 Histograma Cumulativa\n");
		printf(" 23 - L5 BFS\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				negative_image();
				break;
			case 11:
				factor_aditiv();
				break;
			case 12:
				factor_multiplicativ();
				break;
			case 13:
				imagine_color();
				break;
			case 14:
				inversa_matrice3x3();
				break;
			case 15:
				imagineIn3Matrici();
				break;
			case 16:
				conversieRGBgray();
				break;
			case 17:
				int prag;
				printf("Dati valoarea prag: ");
				scanf("%d", &prag);
				grayscale_albnegru(prag);
				break;
			case 18:
				RGB_HSV();
				break;
			case 19:
				char fname[MAX_PATH];
				while (openFileDlg(fname)) 
				{
					Mat img = imread(fname);
					bool r = isInside(img, 2000, 5);
					printf("%d\n", r);
					waitKey();
				}
				break;
			case 20:
				afisare_histograma();
				break;
			case 21:
				afisare_FDP();
				break;
			case 22:
				testShowHistogram();
				break;
			case 23:
				char fname1[MAX_PATH];
				while (openFileDlg(fname1))
				{
					Mat img = imread(fname1);
					int* v = hist_acumulatoare(img, 128);
					showHistogram("h", v, 128, 256);
					waitKey();
				}
				break;
			case 24:
				det_praguri_multiple();
				break;
			case 25:
				FloydSteinberg();
				break;
			case 26:
				char fname2[MAX_PATH];
				while (openFileDlg(fname2)) 
				{
					Mat img = imread(fname2);
					namedWindow("My Window", 1);
					setMouseCallback("My Window", onMouse, &img);
					imshow("My Window", img);
					waitKey();
				}
				break;
			case 27:
				etichetare_BFS();
				break;
			case 28:
				algoritmul_doua_treceri();
				break;
			case 29:
				urmarire_contur();
				break;
			case 30:
				excellent();
				break;
			case 31:
				int n;
				scanf("%d", &n);
				dilatare_eroziune(n);
				break;
			case 32:
				deschidere();
				break;
			case 33:
				inchidere();
				break;
			case 34:
				extragere_contur();
				break;
			case 35:
				umplere_regiuni();
				break;
			case 36:
				afisare();
				break;
			case 37:
				binarizare_automata_globala();
				break;
			case 38:
				char fname3[MAX_PATH];
				while (openFileDlg(fname3)) 
				{
					Mat_<uchar> src = imread(fname3, CV_LOAD_IMAGE_GRAYSCALE);
					Mat_<uchar> dst = negativul_imaginii(src);
					imshow("negative image", dst);
					waitKey();
				}
				break;
			case 39:
				int goutMIN, goutMAX;
				scanf("%d %d", &goutMIN, &goutMAX);
				latirea_ingustarea_histogramei(goutMIN, goutMAX);
				//10, 250 pentru latire
				//50, 150 pentru ingustare
				break;
			case 40:
				float gamma;
				scanf("%f", &gamma);
				char fname4[MAX_PATH];
				while (openFileDlg(fname4))
				{
					Mat_<uchar> src = imread(fname4, CV_LOAD_IMAGE_GRAYSCALE);
					Mat_<uchar> dst = corectia_gamma(src, gamma);
					imshow("de/comprimare gamma", dst);
					waitKey();
				}
				break;
			case 41:
				int offset;
				scanf("%d", &offset);
				modificarea_luminozitatii(offset);
				break;
			case 42:
				egalizarea_histogramei();
				break;
			/*
			case 17:
				afisati_histograma();
				break;
			case 18:
				TraversareInLatime();
				break;
			//case 19:
				//MouseCallBack();
				//break;
			case 19:
				urmarire_contur();
				break;
			case 20:
				testBorderTracingAlgorithm();
				break;
			case 21:
				char fname6[MAX_PATH];
				while (openFileDlg(fname6)) {
					Mat img = imread(fname6);
					namedWindow("My Window", 1);
					setMouseCallback("My Window", onMouse, &img);
					imshow("My Window", img);
					waitKey();
				}
				break;
			case 22:
				calculateHistogramCumulative();
				break;
			case 23:
				getBfsLabeling();
				break;
			case 24:
				char fname7[MAX_PATH];
				while (openFileDlg(fname7)) {
					Mat img = imread(fname7, IMREAD_GRAYSCALE);
					calculateHistogramFromGivenImage(img);
				}
				break;
			/*case 24:
				//int n;
				//scanf("%d", &n);
				getRegionFilling();
				break;
			case 25:
				Mat element = getStructElement();
				imshow("Element Structural", element);
				waitKey(0);
				break;*/
		}
	}
	while (op!=0);
	return 0;
}