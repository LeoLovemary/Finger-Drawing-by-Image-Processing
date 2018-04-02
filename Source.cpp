#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/video/background_segm.hpp>
#include <sstream>
#include <iostream>


using namespace std;
using namespace cv;

int Cr_lbound = 130;
int Cr_hbound = 180;
int Cb_lbound = 80;
int Cb_hbound = 150;
Mat camera_frame;
Mat img_ycbcr;
Mat img_skin_region;
Mat img_roi;
Mat fgMaskMOG;
Ptr<BackgroundSubtractor> pMOG;
vector< vector<Point> > stroke(1);
int stroke_idx=0;
bool drawing_flag=false;

RNG rng(12345);
void skin_filter(Mat, Mat);
int gesture(Mat);
int  distance(Point*, Point*);
void put_hand(Mat, Mat);
void draw(Mat);

int main(int argc, char** argv)
{

	VideoCapture captureDevice;
	captureDevice.open(0);
	captureDevice >> camera_frame;
    
	VideoWriter outputVideo("outputVideo.avi", CV_FOURCC('M','J','P','G'), 10,
             Size(captureDevice.get(CV_CAP_PROP_FRAME_WIDTH),captureDevice.get(CV_CAP_PROP_FRAME_HEIGHT)) );
	
	Mat img_clench = imread("clench.jpg", 1);
	Mat img_palm = imread("palm.png", 1);
	Mat img_point = imread("point.png", 1);
	resize(img_clench, img_clench, Size(80, 120));
	resize(img_palm, img_palm, Size(80, 120));
	resize(img_point, img_point, Size(80, 120));
	
	namedWindow("camera_frame", WINDOW_AUTOSIZE);
	
	int gesture_flag;
	pMOG = new BackgroundSubtractorMOG();

	while(true){
	
	    captureDevice >> camera_frame;
	    if( camera_frame.empty() ){
			cout << "Error: camera frame empty."  << endl;
			return -1;
		}
		
		flip(camera_frame, camera_frame, 1);
		gesture_flag = gesture(camera_frame);
		
        if(gesture_flag==1){
		    put_hand(camera_frame, img_palm);
		}
		else if(gesture_flag==2){
		    put_hand(camera_frame, img_point);
		}
		else if(gesture_flag==3){
		    put_hand(camera_frame, img_clench);
		}
		
		draw(camera_frame);
		
	    imshow("camera_frame", camera_frame);
		outputVideo.write(camera_frame);
		
	    if(waitKey(10)>0)
		    break;
	}
	return(0);
}

void skin_filter(Mat img_ycbcr, Mat img_skin_region){
    
	int Cr, Cb;
    for(int i=0; i<img_ycbcr.rows; i++){
		for (int j=0; j<img_ycbcr.cols; j++){
		    Cr = img_ycbcr.at<Vec3b>(i,j)[1];
			Cb = img_ycbcr.at<Vec3b>(i,j)[2];
			if( Cr>Cr_lbound && Cr<Cr_hbound && Cb>Cb_lbound && Cb<Cb_hbound){
				img_skin_region.at<uchar>(i,j) = 255;
		    }
			else{
			    img_skin_region.at<uchar>(i,j) = 0;
			}
        }
    }
}

int gesture(Mat camera_frame){
    
	int x_shift = 100;
    img_skin_region = Mat::zeros(camera_frame.size(), CV_8U);
	Rect roi(camera_frame.cols/2-x_shift, 0, camera_frame.cols/2+x_shift, camera_frame.rows);
	img_roi = camera_frame(roi);
    cvtColor(img_roi, img_ycbcr, CV_BGR2YCrCb);
	skin_filter(img_ycbcr, img_skin_region);
	namedWindow("skin", WINDOW_AUTOSIZE);
	imshow("skin", img_skin_region);
    rectangle(camera_frame, roi, Scalar(0, 0, 255), 3);


	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(img_skin_region, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	Mat drawing = Mat::zeros(img_skin_region.size(), CV_8UC3);
	
	int max_size=0, max_n, current_size;
	for (int i = 0; i < contours.size(); i++){
	    
		current_size = contourArea(contours[i]);
		if(current_size > max_size){
		    max_size = current_size;
		    max_n    = i;
		}
	}
	
	if(max_size<=10000)
	    return 0;
	
	
	vector< vector<Point> > hull(1);
	convexHull( Mat(contours[max_n]), hull[0], false);
	
	vector< vector<int> > hull2(1);
	convexHull( Mat(contours[max_n]), hull2[0], false);
	vector< Vec4i > defects;
	convexityDefects( Mat(contours[max_n]), hull2[0],  defects);
	int n_defects     = 0;
	int defects_avg_h = 0;
	int defects_max_h = camera_frame.rows;
	int n_higher      = 0;
    for (int i = 0; i < defects.size() ; ++i){
	     
		if(defects[i][3]>4000){
	        circle(drawing, Point(contours[max_n][defects[i][2]].x+camera_frame.cols/2-x_shift, contours[max_n][defects[i][2]].y),
                   3, Scalar(0, 0, 255), -1);
			defects_avg_h += contours[max_n][defects[i][2]].y;
			++n_defects;
			if(contours[max_n][defects[i][2]].y < defects_max_h)
			    defects_max_h = contours[max_n][defects[i][2]].y;
		}
	}
	if(n_defects!=0)
	    defects_avg_h = defects_avg_h/n_defects;
		
	vector<vector<Point> > approx_hand(1);
	approxPolyDP(hull[0], approx_hand[0], 10, false);

	for (int i = 0; i < approx_hand[0].size()-1; ){
  	    if( distance(&approx_hand[0][i], &approx_hand[0][i+1]) < 800 ){
		    approx_hand[0][i].x = (approx_hand[0][i].x+approx_hand[0][i+1].x)/2;
		    approx_hand[0][i].y = (approx_hand[0][i].y+approx_hand[0][i+1].y)/2;
			approx_hand[0].erase( approx_hand[0].begin()+i+1 );
		}
		else
		    ++i;
  	}
	if( distance(&approx_hand[0][0], &approx_hand[0].back()) < 800 ){
        approx_hand[0].back().x = (approx_hand[0][0].x+approx_hand[0].back().x)/2;
        approx_hand[0].back().y = (approx_hand[0][0].y+approx_hand[0].back().y)/2;
    	approx_hand[0].erase( approx_hand[0].begin() );
    }

	int top_y = camera_frame.rows;
	int top_idx = 0;
	for (int i = 0; i < approx_hand[0].size(); ++i){
	    approx_hand[0][i].x += camera_frame.cols/2 - x_shift;
  	    circle(drawing, approx_hand[0][i], 3, Scalar(204, 255, 204), -1);
		if(approx_hand[0][i].y <= defects_max_h)
		    ++n_higher;
		if( approx_hand[0][i].y <= top_y){
		    top_idx = i;
			top_y = approx_hand[0][i].y;
		}
  	}
	
	for (int i = 0; i < contours[max_n].size(); i++)
	    contours[max_n][i].x += camera_frame.cols/2 - x_shift;
	
	
	drawContours(drawing, contours, max_n, Scalar(153, 153, 255), 1, 8, vector<Vec4i>(), 0, Point());
    drawContours(drawing, approx_hand, 0, Scalar(255, 128, 0), 1, 8, vector<Vec4i>(), 0, Point());
	/// Show in a window
	namedWindow("Hull demo", CV_WINDOW_AUTOSIZE);
	imshow("Hull demo", drawing);
	
	if(n_defects>=5){
		
		for (int i = 0; i < approx_hand[0].size(); ++i)
		    if(approx_hand[0][i].y<defects_avg_h)
	            circle(camera_frame, approx_hand[0][i], 5, Scalar(153, 255, 51), -1);
  	    
		if(drawing_flag){
		    vector<Point> temp_v;
			stroke.push_back(temp_v);
		    ++stroke_idx;
		}
			
		drawing_flag = false;
			
	    return 1;
	}
	else if( n_higher>=3 && n_defects<=2 && approx_hand[0].size()>=5 ){
	    
		for (int i = 0 ; i < stroke.size() ; ++i)
		    stroke[i].clear();
			
	    stroke.clear();
			
	    vector<Point> temp_v;
		stroke.push_back(temp_v);
		stroke_idx = 0;
		drawing_flag=false;
		
	    return 3;
	}
    else{
	    circle(camera_frame, approx_hand[0][top_idx], 8, Scalar(255, 102, 118), -1);
		drawing_flag = true;
		if(drawing_flag==true){
            stroke[stroke_idx].push_back( Point(approx_hand[0][top_idx].x, approx_hand[0][top_idx].y) );
		}
		
	    return 2;
	}

}

int  distance(Point* p1, Point* p2){

    return ( ((*p1).x-(*p2).x)*((*p1).x-(*p2).x)+ ((*p1).y-(*p2).y)*((*p1).y-(*p2).y));
 
}

void put_hand(Mat camera_frame, Mat img_hand){

    int m=0;
    for (int i = 0 ; i<80 ; ++i){
    	for (int j = 0; j< 120 ; ++j){
    		camera_frame.at<Vec3b>(j, i)[0] = img_hand.at<Vec3b>(j, m)[0];	
    		camera_frame.at<Vec3b>(j, i)[1] = img_hand.at<Vec3b>(j, m)[1];	
    		camera_frame.at<Vec3b>(j, i)[2] = img_hand.at<Vec3b>(j, m)[2];	
    	}
		++m;
    }
}

void draw(Mat camera_frame){

	for(int i=0 ; i<stroke.size() ; ++i){
	    
		if(stroke[i].size()==0)
		    continue;
	    for (int j = 0 ; j<stroke[i].size()-1 ; ++j){
	        line(camera_frame, stroke[i][j], stroke[i][j+1], Scalar(255, 102, 118), 4);
	    
	    }
    }
}







