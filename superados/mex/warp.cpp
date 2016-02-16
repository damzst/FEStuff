#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>

// Contants:
#define N_WIN 300
#define N_SRC 1920
#define N_SPH 1500
#define N_PTZ 1500
#define FE_L 1
#define FE_M 952
#define FE_U0 960
#define FE_V0 960
#define PTZ_FOV 70
#define PTZ_M N_PTZ/2/tan(PTZ_FOV/2*M_PI/180)
#define PTZ_U0 N_PTZ/2
#define PTZ_V0 N_PTZ/2

// Global variables
cv::Mat src, sph, sphrot, ptz;
cv::Mat sph_x, sph_y, sphrot_x, sphrot_y, ptz_x, ptz_y;
cv::Mat mrot;

void map_sph(void) {
	float theta, phi, r; //coordinates in the sphere

	// create destination and the maps
	sph.create(N_SPH, N_SPH, src.type());
	sph_x.create(N_SPH, N_SPH, CV_32FC1);
	sph_y.create(N_SPH, N_SPH, CV_32FC1);

	// calculate sph_x and sph_y
	for (int i = 0; i < N_SPH; ++i)
	{
		theta = i*M_PI/2/N_SPH;
		r = (FE_L + FE_M)*sin(theta)/(FE_L + cos(theta));
		for (int j = 0; j < N_SPH; ++j)
		{
			phi = (j*2 - N_SPH)*M_PI/N_SPH;
			sph_x.at<float>(i,j) = r*cos(phi) + FE_U0;
			sph_y.at<float>(i,j) = r*sin(phi) + FE_V0;
		}		
	}
}

void find_mrot(int up, int vp) {
	float thetap, phip, rp; //coordinates of the point of interest in the sphere
	float ct, st, cp, sp;
	float m11, m12, m13, m22, m23, m33;
	cv::Mat rotz;

	phip = atan2(vp-FE_V0,up-FE_U0);
	rp = sqrt(pow(up-FE_U0,2)+pow(vp-FE_V0,2));
	thetap = acos((pow(FE_L+FE_M,2)-FE_L*pow(rp,2))/(pow(rp,2)+pow(FE_L+FE_M,2))); //simplified because FE_L = 1

	// P = cartesian coordinates of the point of interest in the sphere
	// Pz = [0,0,-1]
	// Pk = cross(P,Pz)
	// The magnitude of Pk is always abs(sin(thetap)).
	// thetap is between 0 and pi/2 => abs(sin(thetap)) = sin(thetap).
	// Pk = normalize() = Pk/sin(thetap)
	// kx = Pk(1); ky = Pk(2); kz = Pk(3) = 0
	// kx = -sin(phip);
	// ky = cos(phip);
	// Using the matrix seen in findMR and simplifying ETH:
	ct = cos(thetap);
	st = sin(thetap);
	cp = cos(phip);
	sp = sin(phip);

	m11 = ct-pow(sp,2)*(ct-1);
	m12 = cp*sp*(ct-1);
	m13 = cp*st;
	m22 = ct-pow(cp,2)*(ct-1);
	m23 = sp*st;
	m33 = ct;

	mrot.create(3, 3, CV_32FC1);

	mrot.at<float>(0,0) = m11;
	mrot.at<float>(0,1) = m12;
	mrot.at<float>(0,2) = m13;
	mrot.at<float>(1,0) = m12;
	mrot.at<float>(1,1) = m22;
	mrot.at<float>(1,2) = m23;
	mrot.at<float>(2,0) = -m13;
	mrot.at<float>(2,1) = -m23;
	mrot.at<float>(2,2) = m33;

	// Only six number are needed

	// In case that the camera is in the ceiling:
	rotz.create(3, 3, CV_32FC1);
	rotz.zeros(3, 3, CV_32FC1);
	rotz.at<float>(0,0) = cos(phip+M_PI/2);
	rotz.at<float>(0,1) = -sin(phip+M_PI/2);
	rotz.at<float>(0,2) = 0;
	rotz.at<float>(1,0) = sin(phip+M_PI/2);
	rotz.at<float>(1,1) = cos(phip+M_PI/2);
	rotz.at<float>(1,2) = 0;
	rotz.at<float>(2,0) = 0;
	rotz.at<float>(2,1) = 0;
	rotz.at<float>(2,2) = 1;

	mrot = mrot*rotz;
}

void map_sphrot(void) {
	float theta, phi, r; //coordinates in the sphere
	float thetax;
	cv::Mat poi;

	// create destination and the maps
	sphrot.create(N_SPH, N_SPH, src.type());
	sphrot_x.create(N_SPH, N_SPH, CV_32FC1);
	sphrot_y.create(N_SPH, N_SPH, CV_32FC1);
	poi.create(3,1,CV_32FC1);

	// calculate sph_x and sph_y
	for (int i = 0; i < N_SPH; ++i)
	{
		theta = i*M_PI/2/N_SPH;
		for (int j = 0; j < N_SPH; ++j)
		{
			phi = (j*2-N_SPH)*M_PI/N_SPH;
			poi.at<float>(0,0) = sin(theta)*cos(phi);
			poi.at<float>(1,0) = sin(theta)*sin(phi);
			poi.at<float>(2,0) = cos(theta);

			poi = mrot*poi;

			r = sqrt(pow(poi.at<float>(0,0),2) + pow(poi.at<float>(1,0),2));
			if (r>1) r = 1;

			if (poi.at<float>(2,0)<0) thetax = 0;
			else thetax = asin(r);

			phi = atan2(poi.at<float>(1,0),poi.at<float>(0,0));

			sphrot_x.at<float>(i,j) = (phi+M_PI)/(2*M_PI)*N_SPH;
			sphrot_y.at<float>(i,j) = thetax/(M_PI/2)*N_SPH;
		}		
	}
}

void map_ptz(void) {
	float r, phi, theta;

	// create destination and the maps
	ptz.create(N_PTZ, N_PTZ, src.type());
	ptz_x.create(N_PTZ, N_PTZ, CV_32FC1);
	ptz_y.create(N_PTZ, N_PTZ, CV_32FC1);

	// calculate sph_x and sph_y
	for (int i = 0; i < N_PTZ; ++i)
	{
		for (int j = 0; j < N_PTZ; ++j)
		{
			r = sqrt(pow(i-PTZ_U0,2) + pow(j-PTZ_V0,2));
			phi = atan2(j-PTZ_V0,i-PTZ_U0);
			theta = acos(PTZ_M/sqrt(pow(r,2)+pow(PTZ_M,2))); //simplified because PTZ_L = 0

			ptz_x.at<float>(j,i) = (phi+M_PI)/(2*M_PI)*N_PTZ;
			ptz_y.at<float>(j,i) = theta/(M_PI/2)*N_PTZ;
		}		
	}
}

int main( int argc, const char** argv )
{
	src = cv::imread("IM1.jpg", CV_LOAD_IMAGE_UNCHANGED); //load the image
	
	if (src.empty()) //check whether the image is loaded or not
	{
		std::cout << "Error : Image cannot be loaded..!!" << std::endl;
		system("pause"); //wait for a key press
		return -1;
	}

    // Update sph_x & sph_y. Then apply remap
    map_sph();
    remap(src, sph, sph_x, sph_y, CV_INTER_NN, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

    // Update sphrot_x & sphrot_y. Then apply remap
	find_mrot(852,1457);
    map_sphrot();
    remap(sph, sphrot, sphrot_x, sphrot_y, CV_INTER_NN, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

	// Update zpt_x & zpt_y. Then apply remap
    map_ptz();
    remap(sphrot, ptz, ptz_x, ptz_y, CV_INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

	cv::namedWindow("src", CV_WINDOW_NORMAL);
	cv::resizeWindow("src", N_WIN, N_WIN);
	cv::imshow("src", src);

	cv::namedWindow("sph", CV_WINDOW_NORMAL);
	cv::resizeWindow("sph", N_WIN, N_WIN);
	cv::imshow("sph", sph);
	
	cv::namedWindow("sphrot", CV_WINDOW_NORMAL);
	cv::resizeWindow("sphrot", N_WIN, N_WIN);
	cv::imshow("sphrot", sphrot);

	cv::namedWindow("ptz", CV_WINDOW_NORMAL);
	cv::resizeWindow("ptz", N_WIN, N_WIN);
	cv::imshow("ptz", ptz);
	
	cv::waitKey(0); //wait infinite time for a keypress

	cv::destroyWindow("src");
	cv::destroyWindow("sph");
	cv::destroyWindow("sphrot");
	cv::destroyWindow("ptz");

	return 0;
}