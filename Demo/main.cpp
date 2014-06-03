#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>


using namespace cv;
using namespace std;

int mode = 1; //1-square, 0-finding intersections

//mode0--
Point2f computeIntersect(Vec2f line1, Vec2f line2);
vector<Point2f> lineToPointPair(Vec2f line);
bool acceptLinePair(Vec2f line1, Vec2f line2, float minTheta);
//--mode0

//mode1--
double angle(cv::Point pt1,cv::Point pt2,cv::Point pt0);
vector<vector<Point> > squaresPrev;
//--mode1

Mat blueFilter(const Mat& src)
{
    assert(src.type() == CV_8UC3);
    
    Mat blueOnly;
    inRange(src, Scalar(190, 0, 0), Scalar(255, 250, 100), blueOnly); //b,g,r
    
    return blueOnly;
}

int lastX = -1, lastY = -1;
int prevX = -1, prevY = -1;
double area = 0;
Point topL,topR,bottomL,bottomR;
Point greenPnt;
Point bluePnt;
Point redPnt;

string intToString(int number){
    
    
	std::stringstream ss;
	ss << number;
	return ss.str();
}

IplImage* trackObject(IplImage* imgThresh,IplImage* imgTracking, IplImage* paintImg){
    
    // Calculate the moments of 'imgThresh'
    
    if (imgThresh == NULL)
        
        return imgTracking;
    
    CvMoments *moments = (CvMoments*)malloc(sizeof(CvMoments));
    
    cvMoments(imgThresh, moments, 1);
    
    double moment10 = cvGetSpatialMoment(moments, 1, 0);
    
    double moment01 = cvGetSpatialMoment(moments, 0, 1);
    
    area = cvGetCentralMoment(moments, 0, 0);
    
    
    
    if(area>20 ){
        
        
        // calculate the position
        
        int posX = moment10/area ;
        
        int posY = moment01/area;
        
        
        
        if(lastX ==-1 && lastY ==-1){
            
            cvCircle(imgTracking, cvPoint(posX, posY), std::sqrt(area), cvScalar(0,0,255), 1, 8, 0 );
            
            lastX = posX;
            lastY = posY;
            
            
        }
        
        else if(lastX>=0 && lastY>=0 && posX>=0 && posY>=0)
            
        {
            if(lastX >=50 && lastX <=imgTracking->width-50 && lastY >= 20 && lastY <=imgTracking->height-20){
                
                // Draw a yellow line from the previous point to the current point
                
                cvCircle(imgTracking, cvPoint(posX, posY), std::sqrt(area), cvScalar(0,0,255), -1, 8, 0 );
                
                
                cvLine(imgTracking, cvPoint(posX, posY), cvPoint(lastX, lastY), cvScalar(0,0,255), 4);
                cvCircle(imgTracking, cvPoint(posX, posY), std::sqrt(area), cvScalar(0,0,255), 2, 8, 0 );
                
                
            }
        }
        
        prevX = lastX;
        
        prevY = lastY;
        
        lastX = posX;
        
        lastY = posY;
        
        
        
    }
    
    free(moments);
    return imgTracking;
    
}


IplImage* imfill(IplImage* src)

{
    
    if(src == NULL)
        
        return src;
    
    CvScalar white = CV_RGB( 255, 255, 255 );
    
    IplImage* dst = cvCreateImage( cvSize(src->width,src->height), 8, 3);
    
    CvMemStorage* storage = cvCreateMemStorage(0);
    
    CvSeq* contour = 0;
    
    CvSeq* largest = 0;
    float area = 0;
    
    
    cvFindContours(src, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
    
    cvZero( dst );
    
    for( ; contour != 0; contour = contour->h_next )
        
    {
        
        float areaT = fabs(cvContourArea(contour, CV_WHOLE_SEQ));
        
        if(areaT > area){
            
            largest = contour;
            
            area = areaT;
            
        }
        
        
    }
    
    if(largest!=0){
        
        cvDrawContours( dst, largest, white, white, 0, CV_FILLED);
        
    }
    
    //IplImage* bin_imgFilled = cvCreateImage(cvGetSize(src), 8, 1);
    cvInRangeS(dst, white, white, src);
    
    if(storage){
        
        cvClearMemStorage(storage);
        cvReleaseMemStorage(&storage);
        
    }
    
    cvReleaseImage(&dst);
    return src;
    
}


IplImage* GetThresholdedImage(IplImage* imgHSV, IplImage* imgThresh){
    
    cvInRangeS(imgHSV, cvScalar(90, 100, 100), cvScalar(110, 255, 255), imgThresh); //b g r
    //100,100,100 and 110,255,255
    imgThresh = imfill(imgThresh);
    return imgThresh;
    
}


int main(int argc, char* argv[])
{
    VideoCapture cap;
    cap.open(0);
    if( !cap.isOpened() )
    {
        cout << "***Could not initialize capturing...***\n";
        return -1;
    }
    
    while(1)
    {
        Mat image, imgBlue;
        
        cap >> image;
        //to speed up
        resize(image, image, Size(0, 0), 0.25, 0.25);
        
        // imgBlue = blueFilter(image);
        // imshow("imgBlue",imgBlue);
        
        IplImage ipl;
        IplImage *imageB;
        IplImage *paintImgT;
        IplImage *imgTrackingT;
        
        ipl = image;
        IplImage *destination = cvCreateImage ( cvSize(ipl.width, ipl.height),8,3);
        cvResize(&ipl, destination);
        imageB = destination;
        
        imgTrackingT =cvCreateImage(cvGetSize(imageB),8,3);
        cvZero(imgTrackingT); //covert the image, 'imgTracking' to black
        IplImage* imgHSV = cvCreateImage(cvGetSize(imageB), IPL_DEPTH_8U, 3);
        cvCvtColor(imageB, imgHSV, CV_BGR2HSV); //Change the color format from BGR to HSV
        //Get filtered image
        IplImage* imgThresh =cvCreateImage(cvGetSize(imgHSV), IPL_DEPTH_8U, 1);
        imgThresh = GetThresholdedImage(imgHSV,imgThresh);
        //cvDilate(imgThresh,imgThresh,convKernel,3);
        
        IplImage* imgThreshRef =cvCreateImage(cvGetSize(imgHSV), IPL_DEPTH_8U, 1);
        
        imgTrackingT = trackObject(imgThresh,imgTrackingT,paintImgT);
        
        cvAdd(imageB, imgTrackingT, imageB);
        
 
        
        
        
        if(mode == 1){
            vector<vector<Point> > squares;
            
            // blur will enhance edge detection
            Mat blurred(image);
            medianBlur(image, blurred, 9);
            
            Mat gray0(blurred.size(), CV_8U), gray;
            vector<vector<Point> > contours;
            
            //find squares in every color plane of the image
            for (int c = 0; c < 3; c++)
            {
                int ch[] = {c, 0};
                mixChannels(&blurred, 1, &gray0, 1, ch, 1);
                
                // try several threshold levels
                const int threshold_level = 2;
                for (int l = 0; l < threshold_level; l++)
                {
                    // Use Canny instead of zero threshold level!
                    // Canny helps to catch squares with gradient shading
                    if (l == 0)
                    {
                        Canny(gray0, gray, 66, 133, 3); // 10,20,3
                        imshow("edges", gray);
                        waitKey(30);
                        // Dilate helps to remove potential holes between edge segments
                        dilate(gray, gray, Mat(), Point(-1,-1));
                        //GaussianBlur(gray, gray, Size(7, 7), 2.0, 2.0);
                    }
                    else
                    {
                        gray = gray0 >= (l+1) * 255 / threshold_level;
                    }
                    
                    // Find contours and store them in a list
                    findContours(gray, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
                    
                    // Test contours
                    vector<Point> approx;
                    for (size_t i = 0; i < contours.size(); i++)
                    {
                        // approximate contour with accuracy proportional
                        // to the contour perimeter
                        approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);
                        
                        // Note: absolute value of an area is used because
                        // area may be positive or negative - in accordance with the
                        // contour orientation
                        if (approx.size() == 4 &&
                            fabs(contourArea(Mat(approx))) > 1000 &&
                            isContourConvex(Mat(approx)))
                        {
                            double maxCosine = 0;
                            
                            for (int j = 2; j < 5; j++)
                            {
                                double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                                maxCosine = MAX(maxCosine, cosine);
                            }
                            
                            if (maxCosine < 0.3){
                                if(squares.size()==0){
                                    squaresPrev.clear();
                                }
                                squares.push_back(approx);
                                squaresPrev.push_back(approx);
                            }
                        }
                    }
                }
            }
            if(squares.size()>0){
                for ( int i = 0; i< 1; i++ ) { //i< squares.size()
                    // draw contour (red)
                    cv::drawContours(image, squares, i, cv::Scalar(255,0,0), 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
                    
                    // draw bounding rect (green)
                    cv::Rect rect = boundingRect(cv::Mat(squares[i]));
                    cv::rectangle(image, rect.tl(), rect.br(), cv::Scalar(0,255,0), 2, 8, 0);
                    
                    cv::Rect rectTran = boundingRect(cv::Mat(squares[i]));
                    cv::rectangle(image, cv::Point(rect.tl().x-3*rect.width,rect.tl().y), cv::Point(rect.br().x-3*rect.width,rect.br().y), cv::Scalar(0,255,0), 2, 8, 0);
                    greenPnt.x = lastX-(rect.tl().x-3*rect.width);
                    greenPnt.y = lastY-rect.tl().y;
                    putText(image,intToString(greenPnt.x)+","+intToString(greenPnt.y),cv::Point(lastX,lastY-30),1,1,cvScalar(0,255,0),2);
                    
                    cvRectangle(imageB, cv::Point(rect.tl().x-3*rect.width,rect.tl().y), cv::Point(rect.br().x-3*rect.width,rect.br().y), cv::Scalar(0,255,0), 2, 8, 0);
                    
                    char buffer[25];
                    sprintf(buffer, "(%d,%d)", greenPnt.x,greenPnt.y);
                    CvFont font;
                    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1, 1);
                    cvPutText(imageB, buffer, cvPoint(rect.br().x-3*rect.width,rect.br().y), &font, cvScalar(255));
                    
                    // draw rotated rect (blue)
                    cv::RotatedRect minRect = minAreaRect(cv::Mat(squares[i]));
                    cv::Point2f rect_points[4];
                    minRect.points( rect_points );
                    for ( int j = 0; j < 4; j++ ) {
                        cv::line( image, rect_points[j], rect_points[(j+1)%4], cv::Scalar(0,0,255), 1, 8 ); // blue
                    }
                    topL = rect_points[1];
                    topR = rect_points[2];
                    bottomL = rect_points[0];
                    bottomR = rect_points[3];
                    if(bottomR.y < topL.y){
                        topL = rect_points[2];
                        topR = rect_points[3];
                        bottomL = rect_points[1];
                        bottomR = rect_points[0];
                    }
                }
                //cout << "Detected " << squares.size() << " squares." << endl;
            }else{
                cout << "none\n";
                for ( int i = 0; i< MIN(1,squaresPrev.size()); i++ ) { ///squaresPrev.size()
                    // draw contour (red)
                    cv::drawContours(image, squaresPrev, i, cv::Scalar(0,0,255), 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
                    
                    // draw bounding rect (green)
                    cv::Rect rect = boundingRect(cv::Mat(squaresPrev[i]));
                    cv::rectangle(image, rect.tl(), rect.br(), cv::Scalar(0,255,0), 2, 8, 0);
                  //  greenPnt.x = lastX-rect.tl().x;
                  //  greenPnt.y = lastY-rect.tl().y;
                   // putText(image,intToString(greenPnt.x)+","+intToString(greenPnt.y),cv::Point(lastX,lastY+30),1,1,cvScalar(0,255,0),2);
                    // draw rotated rect (red)
                    cv::RotatedRect minRect = minAreaRect(cv::Mat(squaresPrev[i]));
                    cv::Point2f rect_points[4];
                    minRect.points( rect_points );
                    for ( int j = 0; j < 4; j++ ) {
                        cv::line( image, rect_points[j], rect_points[(j+1)%4], cv::Scalar(255,0,0), 1, 8 ); // blue
                    }
                    double dist = sqrt(pow(topL.x-lastX,2)+pow(topL.y-lastY,2));
                    
                    double m1 = ((minRect.angle+90)*3.14) /180.0;
                    //double m2 = (minRect.angle*3.14) /180.0;
                    //cout << minRect.angle*3.14 <<endl;
                    //redPnt.x = (int)(lastX - (topL.x + dist*cos(m1)));
                    //redPnt.y = (int)(lastY - (topL.y + dist*sin(m2)));
                    
                    //putText(image,intToString(redPnt.x)+","+intToString(redPnt.y),cv::Point(lastX,lastY+60),1,1,cvScalar(0,0,255),1);
                }
                
            }
            
            
            
            
            
            
        }
        if(mode == 0){
            //convert the image into grayscale
            Mat image8u;
            cvtColor(image, image8u, CV_BGR2GRAY);
            
            Mat thresh;
            threshold(image8u, thresh, 100.0, 255.0, THRESH_BINARY); //200,255
            
            //Blur
            GaussianBlur(thresh, thresh, Size(7, 7), 2.0, 2.0);
            
            Mat edges;
            Canny(thresh, edges, 99.0, 166.0, 3); //66,133
            imshow("edges", edges);
            waitKey(30);
            
            vector<Vec2f> lines;
            HoughLines( edges, lines, 1, CV_PI/180, 50, 0, 0 );
            
            cout << "Detected " << lines.size() << " lines." << endl;
            
            // compute the intersection from the lines detected...
            vector<Point2f> intersections;
            for( size_t i = 0; i < lines.size(); i++ )
            {
                for(size_t j = 0; j < lines.size(); j++)
                {
                    Vec2f line1 = lines[i];
                    Vec2f line2 = lines[j];
                    if(acceptLinePair(line1, line2, CV_PI / 32))
                    {
                        Point2f intersection = computeIntersect(line1, line2);
                        intersections.push_back(intersection);
                    }
                }
                
            }
            
            if(intersections.size() > 0)
            {
                vector<Point2f>::iterator i;
                for(i = intersections.begin(); i != intersections.end(); ++i)
                {
                    //cout << "Intersection is " << i->x << ", " << i->y << endl;
                    circle(image, *i, 1, Scalar(0, 255, 0), 3);
                }
            }
        }
        //Actual coordinates
        //putText(image,intToString(lastX)+","+intToString(lastY),cv::Point(lastX,lastY+30),1,1,cvScalar(0,0,255),2);
        //cout << "Coordinates in Frame: (" << lastX << ", " << lastY << ")" <<endl;
        
        cvShowImage("Testing", imageB);
        cvReleaseImage(&destination);
        cvReleaseImage(&imgHSV);
        cvReleaseImage(&imgThresh);
        cvReleaseImage(&imgThreshRef);
        
        
        putText(image,"TopL",topL,1,1,cvScalar(255,0,0),2);
        putText(image,"TopR",topR,1,1,cvScalar(255,0,0),2);
        putText(image,"bottomL",bottomL,1,1,cvScalar(255,0,0),2);
        putText(image,"bottomR",bottomR,1,1,cvScalar(255,0,0),2);
        
        imshow("corners", image);
        waitKey(30);
        
        image.release();
        //cvReleaseImage(&imageB);
        //cvReleaseImage(&paintImgT);
        cvReleaseImage(&imgTrackingT);
        
    }
    
    
    
    return 0;
}

//mode0
bool acceptLinePair(Vec2f line1, Vec2f line2, float minTheta)
{
    float theta1 = line1[1], theta2 = line2[1];
    
    if(theta1 < minTheta)
    {
        theta1 += CV_PI; // dealing with 0 and 180 ambiguities...
    }
    
    if(theta2 < minTheta)
    {
        theta2 += CV_PI; // dealing with 0 and 180 ambiguities...
    }
    
    return abs(theta1 - theta2) > minTheta;
}

//mode0
Point2f computeIntersect(Vec2f line1, Vec2f line2)
{
    vector<Point2f> p1 = lineToPointPair(line1);
    vector<Point2f> p2 = lineToPointPair(line2);
    
    float denom = (p1[0].x - p1[1].x)*(p2[0].y - p2[1].y) - (p1[0].y - p1[1].y)*(p2[0].x - p2[1].x);
    Point2f intersect(((p1[0].x*p1[1].y - p1[0].y*p1[1].x)*(p2[0].x - p2[1].x) -
                       (p1[0].x - p1[1].x)*(p2[0].x*p2[1].y - p2[0].y*p2[1].x)) / denom,
                      ((p1[0].x*p1[1].y - p1[0].y*p1[1].x)*(p2[0].y - p2[1].y) -
                       (p1[0].y - p1[1].y)*(p2[0].x*p2[1].y - p2[0].y*p2[1].x)) / denom);
    
    return intersect;
}

//mode0
vector<Point2f> lineToPointPair(Vec2f line)
{
    vector<Point2f> points;
    
    float r = line[0], t = line[1];
    double cos_t = cos(t), sin_t = sin(t);
    double x0 = r*cos_t, y0 = r*sin_t;
    double alpha = 1000;
    
    points.push_back(Point2f(x0 + alpha*(-sin_t), y0 + alpha*cos_t));
    points.push_back(Point2f(x0 - alpha*(-sin_t), y0 - alpha*cos_t));
    
    return points;
}

//mode1
double angle( cv::Point pt1, cv::Point pt2, cv::Point pt0 ) {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}
