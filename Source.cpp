#include "opencv2/opencv.hpp"

#include "opencv/highgui.h"
#include <iostream>
#include <stdio.h>
#include<time.h>
#include<chrono>
#include<Windows.h>
typedef std::chrono::high_resolution_clock eelock;

using namespace std;
using namespace cv;

float x[5] = { 0 }, y[5] = { 0 };
int cur_x = 960, cur_y = 540;

/** Function Headers */
void detectAndDisplay(Mat frame);
Vec3f detectCircle(Mat frame);
void calibration(Mat frame);
void preprocess_calc(vector<Vec3f>, int, int);

int getEyeball(cv::Mat &eye, std::vector<cv::Vec3f> &circles)
{
	std::vector<int> sums(circles.size(), 0);
	for (int y = 0; y < eye.rows; y++)
	{
		uchar *ptr = eye.ptr<uchar>(y);
		for (int x = 0; x < eye.cols; x++)
		{
			int value = static_cast<int>(*ptr);
			for (int i = 0; i < circles.size(); i++)
			{
				cv::Point center((int)std::round(circles[i][0]), (int)std::round(circles[i][1]));
				int radius = (int)std::round(circles[i][2]);
				if (std::pow(x - center.x, 2) + std::pow(y - center.y, 2) < std::pow(radius, 2))
				{
					sums[i] += value;
				}
			}
			++ptr;
		}
	}
	int smallestSum = 9999999;
	int smallestSumIndex = -1;
	for (int i = 0; i < circles.size(); i++)
	{
		if (sums[i] < smallestSum)
		{
			smallestSum = sums[i];
			smallestSumIndex = i;
		}
	}
	return smallestSumIndex;
}
int detectEyes(cv::Mat &eye, std::vector<cv::Vec3f> &circles)
{
		if (circles.size() > 0)
		{
			int eyeball = getEyeball(eye, circles);
			return eyeball;
		}
}

void LeftClick()
{
	INPUT    Input = { 0 };
	// left down 
	Input.type = INPUT_MOUSE;
	Input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
	::SendInput(1, &Input, sizeof(INPUT));

	// left up
	::ZeroMemory(&Input, sizeof(INPUT));
	Input.type = INPUT_MOUSE;
	Input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
	::SendInput(1, &Input, sizeof(INPUT));
}

/** Global variables */
String face_cascade_name = "haarcascade_frontalface_default.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);
int filenumber = 0, eyenumber = 0;

/* function main */
int main(int argc, const char** argv)
{
	/*
	//Calibration Phase
	char calibrate_filenames[5][30] = {"calibrate/upper_left.jpg","calibrate/bottom_left.jpg","calibrate/bottom_right.jpg","calibrate/upper_right.jpg","calibrate/center.jpg"};
	for (int i = 0; i < 5; i++)
	{
		char current[30];
		strcpy(current,calibrate_filenames[i]);
		Mat tmp = imread(current,1);
		calibration(tmp);
	}
	exit;
	*/
	//detectCircle();
	CvCapture* capture;
	VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened())  // check if succeeded to connect to the camera
		CV_Assert("Cam open failed");
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);
	Mat frame;
	time_t runtime1, runtime2;
	clock_t start;
	
	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error loading\n"); return -1; };

	//-- 2. Read the video stream

	/*Mat c = imread("saves/50.jpg", 1);
	calibration(c);
	//waitKey(0);
	return 0;
	*/

	//capture = cvCaptureFromCAM(CV_CAP_ANY);
	//cap >> frame;
	if (/*capture*/true)
	{
		while (true)
		{
			time(&runtime1);
			start = clock();
			cap.read(frame);
			flip(frame, frame, +1);
			namedWindow("live", CV_WINDOW_AUTOSIZE);
			imshow("live", frame);
			waitKey(1);
			//waitKey(3000);
			//-- 3. Apply the classifier to the frame
			if (!frame.empty())
			{

				//imwrite("Face.jpg", frame);
				//frame = imread("saves/50.jpg", 1);
				//auto temp = eelock::now();
				detectAndDisplay(frame);
				//cout << "\nFPS - " << double(1000)/(std::chrono::duration_cast<std::chrono::milliseconds>(eelock::now() - temp).count());

			}
			else
			{
				printf(" --(!) No captured frame -- Break!"); break;
			}
			//cout << "\nOld time - " << runtime1 << "\t New time - " << time(0) << endl;
			//cout << "\nThe runtime is - " << difftime(time(0),runtime1);
			//cout << "\nFPS - " << (1.0 / difftime(time(0), runtime1));
			//cout << "\nCPS" << (clock() - start) / (double)CLOCKS_PER_SEC;
			//cout << "\nFPS - " << double(1000)/(std::chrono::duration_cast<std::chrono::milliseconds>(eelock::now() - temp).count());
			//int c = waitKey(10000);
			//if ((char)c == 'x') { break; }
		}
	}
	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame)
{
	auto temp = eelock::now();

	std::vector<Rect> faces;
	Mat frame_gray;
	//Mat im2(frame.rows,frame.cols, CV_8UC1, Scalar(0, 0, 0));
	Mat res = frame;
	Mat wres = frame.clone(); 
	int a;
	fstream myfile("tmp.txt", ios::app | ios::in);
	while (myfile >> a)
	{
		filenumber = a;
		//cout << filenumber << endl;
	}
//	myfile.clear();
//	myfile.flush();
//	myfile.close();
//	return;
	//cout << "\n\nSize of Input Image- " << frame.size();

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	//imshow("window3", frame_gray);
	//frame_gray = frame;

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	if (faces.size() != 1)
	{
		myfile.close();
		return;
	}
	for (size_t i = 0; i < faces.size(); i++)
	{
		//cout << "\nNumber of faces - " << faces.size();
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		//imshow(window_name, res);
		//cout << "\n\nSize of Gray Image- " << frame_gray.size();
		Mat faceROI = frame_gray(faces[i]);

		//Offset Calc for face ROI
		Size t1_size;
		Point t1_point;
		faceROI.locateROI(t1_size, t1_point);
	
		//cout << "\n\nSize of face ROI - " << faces[i].size();
		//cout << "\n\nFace ROI x - " << faces[i].x;
		//imshow("Face", faceROI);			//Show face only
		//waitKey();
		//cout << faceROI << endl;
		std::vector<Rect> eyes;
		//-- In each face, detect eyes
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		if (eyes.size() == 0)
		{
			myfile.close();
			return;
		}
		for (size_t j = 0; j < eyes.size(); j++)
		{

			//cout << "\nNumber of eyes - " << eyes.size();
			//cout << "width - " << eyes[0].width;
			//cout << "height - " << eyes[0].height;
			Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);

			vector<Vec3f> result_final;
			Vec3f final = detectCircle(faceROI(eyes[j]));
			Mat eyesROI = faceROI(eyes[j]);

			Size t2_size;
			Point t2_point;
			eyesROI.locateROI(t2_size, t2_point);
			if (final[0] == 0)
				return;
			Point r(t2_point.x + final[0], t2_point.y + final[1]);
			circle(frame, r, 3, Scalar(0, 255, 0), -1, 8, 0);



			/*if (result_final.size() == 0)
				return;
			int a_x = faces[0].x + eyes[0].x, a_y = faces[0].y + eyes[0].y;
			preprocess_calc(result_final, a_x, a_y);
			return;
			char filename[20];
			
			sprintf_s(filename, sizeof(filename), "Eye/a - %d.jpg", filenumber);
			//sprintf_s(filename, sizeof(filename), "Eye/a.jpg");
			//imshow("Eye1", faceROI(eyes[0]));
			imwrite(filename, faceROI(eyes[0]));
			
			sprintf_s(filename, sizeof(filename), "Eye/b - %d.jpg", filenumber);
			//sprintf_s(filename, sizeof(filename), "Eye/b.jpg");
			//imshow("Eye2", faceROI(eyes[1]));
			imwrite(filename, faceROI(eyes[1]));
			
			//Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
			//int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			//circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);
			//imshow(window_name, res);							Show for each eye
			//waitKey();
			//circle(im2, center, radius, Scalar(255, 0, 0), 4, 8, 0);
			//bitwise_and(frame, im2, res);
			
			//vector<Vec3f> result_final = detectCircle(faceROI(eyes[1])); */
		}
		namedWindow("Result", CV_WINDOW_NORMAL);
		imshow("Result", frame);
		//cout << "\nFPS - " << double(1000) / (std::chrono::duration_cast<std::chrono::milliseconds>(eelock::now() - temp).count());

		waitKey(1);
		waitKey(0);

		destroyAllWindows();
		return;

		//End of center of eye


		imwrite("Face.jpg", frame);
		imshow("Output", frame);
		//exit;
	}

	char buffer[5];
	_itoa_s(filenumber, buffer, 10);
	//-- Show what you got
	imshow(buffer, res);
	int c = waitKey(0);
	if ((char)c == 's')
	{
		char filename[20];
		sprintf_s(filename, sizeof(filename), "saves/%d.jpg", filenumber++);
		_itoa_s(filenumber, buffer, 10);
		myfile.clear();
		myfile << filenumber << endl;
		myfile.flush();
		myfile.close();
		imwrite(filename, wres);
	}
	cvDestroyWindow("Output");
	cvDestroyWindow(buffer);
	return;
	//waitKey(1000);
	//cout << filename;
	//imwrite(filename,frame);
	//imread("/Eyes/1.jpg",);

}

Vec3f detectCircle(Mat frame)
{
	//Mat src = imread("Eye/a - 39.jpg", 1);
	//Mat src_gray = src;
	Mat src = frame;
	Mat src_gray = src;
	imshow("temp0", src_gray);
	waitKey(1);

	//cvtColor(src, src_gray, CV_BGR2GRAY);
	//GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);	//Causes the center to shift
	//medianBlur(src_gray, src_gray, 5);
	
	//namedWindow("Hough Circle Transform", CV_WINDOW_NORMAL);
	//imshow("Hough Circle Transform", src_gray);
	
	vector<Vec3f> circles;
	HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1, 1000, 100, 17, 5, 15);
	//cout << "\n\nCircles - " << circles.size();
	if (circles.size() == 0)
		return 0;
	int ne;		//Index of the darkest center
	if (circles.size() >= 1)
		ne = detectEyes(frame, circles);
	else
		ne = 0;

	Point center(cvRound(circles[ne][0]), cvRound(circles[ne][1]));
	int radius = cvRound(circles[ne][2]);
	cout << "\nCircle parameters:-\nX: " << cvRound(circles[ne][0]) << "\tY: " << cvRound(circles[ne][1]) << "\trad : " << cvRound(circles[ne][2]) << "\n";
	// circle center
	circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);	//Center of the eye
	//cout << cvRound(circles[i][0]) << '\t' << cvRound(circles[i][1]) << '\t' << radius << endl;
	//cout << "Radius  " << radius << endl;
	//cout << "HERE  " << src_gray.rows / 8 << endl;
	// circle outline
	circle(src, center, radius, Scalar(0, 0, 255), 3, 8, 0);	//The outline of circle
	namedWindow("Hough Circle Transform Demo", CV_WINDOW_NORMAL);
	imshow("Hough Circle Transform Demo", src);

	//waitKey(0);
	return circles[ne];
}

void calibration(Mat frame)
{
	int calibrate_filenumber = 0;
	std::vector<Rect> faces;
	Mat frame_gray;
	//Mat im2(frame.rows,frame.cols, CV_8UC1, Scalar(0, 0, 0));
	Mat res = frame;
	Mat wres = frame.clone();
	int a;
	

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);


	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	if (faces.size() != 1)
		return;
	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;

		//-- In each face, detect eyes
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		
		/*if (eyes.size() != 2)
			return;*/

		for (size_t j = 0; j < eyes.size(); j++)
		{
			char filename[20];
			//sprintf_s(filename, sizeof(filename), "Eye/a - 39.jpg");
			//imwrite(filename, faceROI(eyes[0]));


			Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);
			Mat temp = faceROI(eyes[1]);
			imshow("temp", temp);
			waitKey(1);


			//vector<Vec3f> circles = detectCircle(temp);
			//cout << circles.size();
			//preprocess_calc(circles);
			waitKey(0);
			
			fstream myfile("calibrate.txt", ios::app | ios::in);
			//myfile << endl << circles[j][0] << '\t' << circles[j][1] << '\t' << circles[j][2] << endl;
			myfile.clear();
			myfile.flush();
			myfile.close();

		}
	}
}

void preprocess_calc(vector<Vec3f> comp, int a_x, int a_y)
{
	//SetCursorPos(cur_x, cur_y);
	
	//int x[5], y[5], c[5];
	int dist_x, dist_y;
	int jump_x, jump_y;
	int pos_x, pos_y;
	int pixeltoeyedensity_x, pixeltoeyedensity_y;
	//fstream myfile("calibrate.txt", ios::in);
	char calibrate_filenames[5][30] = { "calibrate/upper_left.jpg","calibrate/bottom_left.jpg","calibrate/bottom_right.jpg","calibrate/upper_right.jpg","calibrate/center.jpg" };
	//myfile >> x[0] >> y[0] >> c[0];
	/*for (int i = 0; i < 5; i++)
	{
		myfile >> x[i] >> y[i] >> c[i];
	}
	
	myfile.close();

	dist_x = sqrt((x[0] - x[3])*(x[0] - x[3]) + (y[0] - y[3])*(y[0] - y[3]));
	
	dist_y = sqrt((x[0] - x[1])*(x[0] - x[1]) + (y[0] - y[1])*(y[0] - y[1]));
	
	pixeltoeyedensity_x = 1920 / dist_x;
	pixeltoeyedensity_y = 1080 / dist_y;

	cout << "X dir - " << pixeltoeyedensity_x << "\nY dir - " << pixeltoeyedensity_y << endl;
	waitKey(0);
	
	//Mapping
	jump_x = comp[0][0] - x[0];
	jump_y = comp[0][1] - y[0];
	
	pos_x = jump_x * pixeltoeyedensity_x;
	pos_y = jump_y * pixeltoeyedensity_y;

	while (true)
	{
		//Use that windows.h function to click on screen indefinitely
		if (waitKey(0) == 'q')
			break;
	}
	*/
	/*if (comp[0][0] < x[0])
		cur_x += 50;
	if (comp[0][0] > x[1])
		cur_x -= 50;
	if (comp[0][1] < y[0])
		cur_y += 50;
	if (comp[0][1] > y[0])
		cur_y -= 50;
	*/
	if (comp.size() != 1)
		return;

	comp[0][0] += a_x;
	comp[0][1] += a_y;

	cout << comp[0][1] << '\t' << y[0] << endl;
	cout << comp[0][0] << '\t' << x[0] << endl;
	if (comp[0][0] < x[0])
	{
		x[0] = comp[0][0];
		cur_x -= 50;
		if (cur_x < 200)
		{
			cur_x += 50;
		}
	}
	if (comp[0][0] > x[0])
	{
		x[0] = comp[0][0];
		cur_x += 50;
		if (cur_x>1500)
		{
			cur_x -= 50;
		}
	}
	if (comp[0][1] < y[0])
	{
		y[0] = comp[0][1];
		cur_y -= 50;
		if (cur_y < 200)
		{
			cur_y += 50;
		}
	}
	if (comp[0][1] > y[0])
	{
		y[0] = comp[0][1];
		cur_y += 50;
		if (cur_y>700)
		{
			cur_y -= 50;
		}
	}
	SetCursorPos(cur_x,cur_y);
	LeftClick();
		//waitKey(0);
}