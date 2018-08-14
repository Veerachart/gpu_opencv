#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/Point32.h>
#include <gpu_opencv/FaceServo.h>
#include <iostream>
#include <iomanip>
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

namespace enc = sensor_msgs::image_encodings;

template<class T>
void convertAndResize(const T& src, T& gray, T& resized, double scale)
{
    if (src.channels() == 3)
    {
        cvtColor( src, gray, CV_BGR2GRAY );
    }
    else
    {
        gray = src;
    }

    Size sz(cvRound(gray.cols * scale), cvRound(gray.rows * scale));

    if (scale != 1)
    {
        resize(gray, resized, sz);
    }
    else
    {
        resized = gray;
    }
}

static void matPrint(Mat &img, int lineOffsY, Scalar fontColor, const string &ss)
{
    int fontFace = FONT_HERSHEY_DUPLEX;
    double fontScale = 0.8;
    int fontThickness = 2;
    Size fontSize = cv::getTextSize("T[]", fontFace, fontScale, fontThickness, 0);

    Point org;
    org.x = 1;
    org.y = 3 * fontSize.height * (lineOffsY + 1) / 2;
    putText(img, ss, org, fontFace, fontScale, CV_RGB(0,0,0), 5*fontThickness/2, 16);
    putText(img, ss, org, fontFace, fontScale, fontColor, fontThickness, 16);
}


static void displayState(Mat &canvas, bool bHelp, bool bLargestFace, bool bFilter, double fps)
{
    Scalar fontColorRed = CV_RGB(255,0,0);
    Scalar fontColorNV  = CV_RGB(118,185,0);

    ostringstream ss;
    ss << "FPS = " << setprecision(1) << fixed << fps;
    matPrint(canvas, 0, fontColorRed, ss.str());
    ss.str("");
    ss << "[" << canvas.cols << "x" << canvas.rows << "], " <<
        "GPU, " <<
        (bLargestFace ? "OneFace, " : "MultiFace, ") <<
        (bFilter ? "Filter:ON" : "Filter:OFF");
    matPrint(canvas, 1, fontColorRed, ss.str());

    // by Anatoly. MacOS fix. ostringstream(const string&) is a private
    // matPrint(canvas, 2, fontColorNV, ostringstream("Space - switch GPU / CPU"));
    if (bHelp)
    {
        matPrint(canvas, 2, fontColorNV, "M - switch OneFace / MultiFace");
        matPrint(canvas, 3, fontColorNV, "F - toggle rectangles Filter");
        matPrint(canvas, 4, fontColorNV, "H - toggle hotkeys help");
        matPrint(canvas, 5, fontColorNV, "1/Q - increase/decrease scale");
        matPrint(canvas, 6, fontColorNV, "+/- - increase/decrease threshold");
    }
    else
    {
        matPrint(canvas, 2, fontColorNV, "H - toggle hotkeys help");
    }
}

int maximum( int a, int b, int c )
{
   int max = ( a < b ) ? b : a;
   return ( ( max < c ) ? c : max );
}


class FacesDetector_GPU
{
public:
    FacesDetector_GPU(ros::NodeHandle nh, ros::NodeHandle nh_private);
    void imageCallback(const sensor_msgs::Image::ConstPtr& msg);
    void update();
private:
    ros::NodeHandle nh_, nh_private_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
    ros::Publisher face_pub_;
    CascadeClassifier_GPU front_cascade;
    CascadeClassifier_GPU side_cascade;
    /* parameters */
    bool running;
    double scaleFactor;
    bool findLargestObject;
    bool filterRects;
    bool helpScreen;
    int neighborThreshold;
    
    int detections_num_front;
    int detections_num_left;
    int detections_num_right;

    KalmanFilter kf;        // Tracking the face
    int face_state;         // 0 = Left, 1 = front, 2 = right
    bool has_face;          // To set that now there is a face being tracked
    int count_face;         // Counting the times face is correctly tracked, if > 3 --> has_face = true;
    ros::Time last_time;
    ros::Time loss_time;    // The time when the tracked face is considered as loss
                            // So that 5 seconds after this, the new face must be around the previously tracked face, not the one with the most hits
                            // Otherwise, just use the most hits
    Point3f prev_face;  // Keep previous face for checking

    // For grouping faces detection from front, left, and right detectors, then estimate the direction. Output only one face.
    void groupFaces(vector<Rect> &inputFront, vector<Rect> &inputLeft, vector<Rect> &inputRight, Rect &output, float* best_prob,
            vector<Rect> &outputList, vector<vector<float> > &probs, int groupThreshold=2, double eps=0.2);
};

int main(int argc, char **argv)
{
    if (getCudaEnabledDeviceCount() == 0)
    {
        return cerr << "No GPU found or the library is compiled without GPU support" << endl, -1;
    }
    printShortCudaDeviceInfo(getDevice());
    
    ros::init(argc, argv, "face_detect_blimp");
    ros::NodeHandle nh, nh_private("~");
    FacesDetector_GPU face_detector(nh, nh_private);
    ros::Rate r(30);
    while (ros::ok()) {
        face_detector.update();
        ros::spinOnce();
        r.sleep();
    }
    
    return 0;
}

FacesDetector_GPU::FacesDetector_GPU(ros::NodeHandle nh, ros::NodeHandle nh_private) : nh_(nh), nh_private_(nh_private), it_(nh_){
    string front_cascade_name, side_cascade_name;
    string default_cascade = "/home/veerachart/opencv/data/haarcascades_GPU/haarcascade_frontalface_alt.xml";
    string default_side_cascade = "/home/veerachart/opencv/data/haarcascades_GPU/haarcascade_profileface.xml";
    //nh_.param<string>("cascade_name", cascade_name, default_cascade);
    nh_private_.param("cascade_front", front_cascade_name, default_cascade);
    nh_private_.param("cascade_side", side_cascade_name, default_cascade);
    if (!front_cascade.load(front_cascade_name))
    {
        ROS_ERROR("Could not load cascade classifier \"%s\"", front_cascade_name.c_str());
        return;
    }
    if (!side_cascade.load(side_cascade_name))
    {
        ROS_ERROR("Could not load cascade classifier \"%s\"", side_cascade_name.c_str());
        return;
    }
    
    namedWindow("result", 1);
    
    image_sub_ = it_.subscribe("image", 1, &FacesDetector_GPU::imageCallback, this);
    image_pub_ = it_.advertise("face_detect", 1);
    
    //face_pub_ = nh_.advertise<geometry_msgs::PolygonStamped>("/face", 1);
    face_pub_ = nh.advertise<gpu_opencv::FaceServo>("face", 1);
    
    running = true;
    scaleFactor = 1.0;
    findLargestObject = false;
    filterRects = false;
    helpScreen = false;
    neighborThreshold = 5;

    has_face = false;
    count_face = 0;
}

void FacesDetector_GPU::imageCallback(const sensor_msgs::Image::ConstPtr& msg) {
    if (!running)
        ros::shutdown();
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, enc::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    Mat faces_downloaded, left_downloaded, right_downloaded, frameDisp, resized_cpu;

    GpuMat frame_gpu, gray_gpu, resized_gpu, facesBuf_gpu, flipped_gpu, leftBuf_gpu, rightBuf_gpu;
    
    frame_gpu.upload(cv_ptr->image);
    
    convertAndResize(frame_gpu, gray_gpu, resized_gpu, scaleFactor);
    flip(resized_gpu, flipped_gpu, 1);        // flip across y axis for left-right detections
    
    TickMeter tm;
    tm.start();

    //cascade_gpu.visualizeInPlace = true;
    front_cascade.findLargestObject = findLargestObject;
    side_cascade.findLargestObject = findLargestObject;

    detections_num_front = front_cascade.detectMultiScale(resized_gpu, facesBuf_gpu, 1.2,
                                                          (filterRects || findLargestObject) ? neighborThreshold : 0);
    facesBuf_gpu.colRange(0, detections_num_front).download(faces_downloaded);

    detections_num_left = side_cascade.detectMultiScale(resized_gpu, leftBuf_gpu, 1.2,
                                                        (filterRects || findLargestObject) ? neighborThreshold : 0);
    leftBuf_gpu.colRange(0, detections_num_left).download(left_downloaded);

    detections_num_right = side_cascade.detectMultiScale(flipped_gpu, rightBuf_gpu, 1.2,
                                                        (filterRects || findLargestObject) ? neighborThreshold : 0);
    rightBuf_gpu.colRange(0, detections_num_right).download(right_downloaded);

    resized_gpu.download(resized_cpu);

    tm.stop();
    double detectionTime = tm.getTimeMilli();
    double fps = 1000 / detectionTime;
    
    /*cout << setfill(' ') << setprecision(2);
    cout << setw(6) << fixed << fps << " FPS, " << detections_num << " det";
    if ((filterRects || findLargestObject) && detections_num > 0)
    {
        Rect *faceRects = useGPU ? faces_downloaded.ptr<Rect>() : &facesBuf_cpu[0];
        for (int i = 0; i < min(detections_num, 2); ++i)
        {
            cout << ", [" << setw(4) << faceRects[i].x
                 << ", " << setw(4) << faceRects[i].y
                 << ", " << setw(4) << faceRects[i].width
                 << ", " << setw(4) << faceRects[i].height << "]";
        }
    }
    cout << endl;*/
    
    cvtColor(resized_cpu, frameDisp, CV_GRAY2BGR);
    /*for (int i = 0; i < detections_num_front; ++i)
    {
        Rect r = faces_downloaded.ptr<cv::Rect>()[i];
        rectangle(frameDisp, r, Scalar(0,255,0));
    }
    for (int i = 0; i < detections_num_left; ++i)
    {
        Rect r = left_downloaded.ptr<cv::Rect>()[i];
        rectangle(frameDisp, r, Scalar(0,0,255));
    }*/
    for (int i = 0; i < detections_num_right; ++i)
    {
        Rect r = right_downloaded.ptr<cv::Rect>()[i];
        r.x = resized_gpu.cols - r.br().x;             // flip back
        right_downloaded.at<Rect>(0,i) = r;
        //rectangle(frameDisp, r, Scalar(255,0,0));
    }

    Rect the_face;          // For the most probable face
    vector<vector<float> > probs;           // For the probability of being Front, Left, or Right
    float best_prob[3];
    vector<Rect> facesList;
    vector<Rect> frontVector, leftVector, rightVector;
    frontVector.assign(faces_downloaded.begin<Rect>(), faces_downloaded.end<Rect>());
    leftVector.assign(left_downloaded.begin<Rect>(), left_downloaded.end<Rect>());
    rightVector.assign(right_downloaded.begin<Rect>(), right_downloaded.end<Rect>());
    groupFaces(frontVector, leftVector, rightVector, the_face, best_prob, facesList, probs, neighborThreshold, 0.4);

    for (int i = 0; i < facesList.size(); i++) {
        rectangle(frameDisp, facesList[i], Scalar(255,255,255), 2);
    }
    gpu_opencv::FaceServo face_msg;
    face_msg.header.stamp = ros::Time::now();

    char buff[20];
    if (facesList.size()) {
        //rectangle(frameDisp, the_face, Scalar(0,255,255), 2);
        Scalar face_color(0,255,0);
        int new_face_state = 0;

        //sprintf(buff, "%.2f,%.2f,%.2f", probs[0], probs[1], probs[2]);
        //putText(frameDisp, buff, the_face.tl(), FONT_HERSHEY_DUPLEX, 0.8, face_color, 2, 8, false);

        float new_face_x;       // Position relative to center
        float new_face_y;       // Position relative to center

        if (has_face) {             // already tracking some face
            // Check all in the facesList (not just only the one with the highest face hit)
            int best_face = -1;
            float best_dist = 1e6;
            Point2f current_center(kf.statePost.at<float>(0,0) + resized_gpu.cols/2., kf.statePost.at<float>(1,0) + resized_gpu.rows/2.);
            float current_size = kf.statePost.at<float>(2,0);
            float current_sd = sqrt(kf.errorCovPost.at<float>(2,2));
            for (int i = 0; i < facesList.size(); i++) {
                Rect face = facesList[i];
                float dist = norm(Point2f(face.x + face.width/2., face.y + face.height/2.) - current_center);
                float size_diff = fabsf(current_size - float(face.width));
                if (size_diff > 3*current_sd)
                    continue;           // Too much size different, can't belong to the same face that is being tracked
                if (dist < best_dist) {
                    best_dist = dist;
                    best_face = i;
                }
            }

            new_face_x = facesList[best_face].x +facesList[best_face].width/2. - resized_gpu.cols/2.;
            new_face_y = facesList[best_face].y +facesList[best_face].height/2. - resized_gpu.rows/2.;
            Mat measurement = (Mat_<float>(3,1) << new_face_x, new_face_y, facesList[best_face].width);
            Mat est = kf.statePost.rowRange(0,3);
            Mat err = est - measurement;
            if (best_dist < 3*sqrt(max(kf.errorCovPost.at<float>(0,0), kf.errorCovPost.at<float>(1,1)))) {
                if (probs[best_face][1] > 0.6) {            // Left must be more than 0.6
                    face_color = Scalar(0,0,255);
                    new_face_state = 1;
                }
                if (probs[best_face][2] > 0.6) {            // Right must be more than 0.6
                    face_color = Scalar(255,0,0);
                    new_face_state = -1;
                }
                setIdentity(kf.measurementNoiseCov, Scalar::all(facesList[best_face].width * facesList[best_face].width / 16.));
                kf.errorCovPost.at<float>(2,2) = the_face.width * the_face.width / 64.;
                kf.errorCovPost.at<float>(5,5) = the_face.width * the_face.width / 64.;
                kf.correct(measurement);

                // check with previous direction
                if (abs(new_face_state - face_state) > 1) {         // left<-->right should not occur without passing front
                    if (new_face_state > face_state) {
                        face_state++;
                    }
                    else {
                        face_state--;
                    }
                }
                else {
                    face_state = new_face_state;
                }
            }
            else {
                ROS_INFO("Outside 3SD");
            }
        }
        else {
            kf = KalmanFilter(6,3,0);
            setIdentity(kf.measurementMatrix);
            setIdentity(kf.transitionMatrix);
            setIdentity(kf.processNoiseCov, Scalar::all(100));          // SD 10 px
            kf.processNoiseCov.at<float>(2,2) = 16;
            kf.processNoiseCov.at<float>(5,5) = 16;
            if (loss_time.isZero() || ros::Time::now() - loss_time > ros::Duration(5.0)) {      // First time, no loss yet; or loss for more than 5 seconds
                // Use the face with the most hit
                new_face_x = (the_face.x + the_face.width/2.) - resized_gpu.cols/2.;
                new_face_y = (the_face.y + the_face.height/2.) - resized_gpu.rows/2.;
                if (best_prob[1] > 0.6) {           // Left must be more than 0.6
                    face_color = Scalar(0,0,255);
                    new_face_state = 1;
                }
                if (best_prob[2] > 0.6) {           // Right must be more than 0.6
                    face_color = Scalar(255,0,0);
                    new_face_state = -1;
                }
                setIdentity(kf.errorCovPost, Scalar::all(the_face.width * the_face.width / 16.));
                kf.errorCovPost.at<float>(2,2) = the_face.width * the_face.width / 64.;
                kf.errorCovPost.at<float>(5,5) = the_face.width * the_face.width / 64.;
                kf.statePost = (Mat_<float>(6,1) << new_face_x, new_face_y , the_face.width, 0, 0, 0);
            }
            else {          // After face loss within 5 seconds
                // Use the face closest to the previously tracked face
                float best_dist = 1e6;
                int best_face = -1;
                for (int i = 0; i < facesList.size(); i++) {
                    Rect face = facesList[i];
                    Point3f face_i(face.x + face.width/2., face.y+face.height/2., face.width);
                    float dist = norm(face_i - prev_face);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_face = i;
                    }
                }

                Rect new_face = facesList[best_face];

                new_face_x = new_face.x + new_face.width/2. - resized_gpu.cols/2.;
                new_face_y = new_face.y + new_face.height/2. - resized_gpu.rows/2.;
                Mat measurement = (Mat_<float>(3,1) << new_face_x, new_face_y, new_face.width);
                if (probs[best_face][1] > 0.6) {            // Left must be more than 0.6
                    face_color = Scalar(0,0,255);
                    new_face_state = 1;
                }
                if (probs[best_face][2] > 0.6) {            // Right must be more than 0.6
                    face_color = Scalar(255,0,0);
                    new_face_state = -1;
                }
                setIdentity(kf.errorCovPost, Scalar::all(new_face.width * new_face.width / 16.));
                kf.errorCovPost.at<float>(2,2) = new_face.width * new_face.width / 64.;
                kf.errorCovPost.at<float>(5,5) = new_face.width * new_face.width / 64.;
                kf.statePost = (Mat_<float>(6,1) << new_face_x, new_face_y , new_face.width, 0, 0, 0);
            }
            kf.errorCovPre = kf.errorCovPost;
            kf.statePre = kf.statePost;
            face_state = new_face_state;
            last_time = ros::Time::now();
            has_face = true;
            ROS_INFO("Start tracking face");
        }

        if (count_face >= 10) {     // When face has been tracked for 10 frames
            face_msg.x = (int) kf.statePost.at<float>(0,0);
            face_msg.y = (int) kf.statePost.at<float>(1,0);
            face_msg.size = (int) kf.statePost.at<float>(2,0);
            face_msg.direction = face_state;
            face_pub_.publish(face_msg);

            // draw on the image for debug
            Scalar tracked_color = (face_state == 0 ? Scalar(64,255,64) : (face_state == 1) ? Scalar(64,64,255) : Scalar(255,64,64));
            rectangle(frameDisp, Point(face_msg.x - face_msg.size/2. + resized_gpu.cols/2., face_msg.y - face_msg.size/2. + resized_gpu.rows/2.),
                    Point(face_msg.x + face_msg.size/2. + resized_gpu.cols/2., face_msg.y + face_msg.size/2. + resized_gpu.rows/2.),
                    tracked_color, 2);
            sprintf(buff, "%d, %d, %d", face_msg.x, face_msg.y, face_msg.size);
            putText(frameDisp, buff, Point(face_msg.x - face_msg.size/2. + resized_gpu.cols/2., face_msg.y - face_msg.size/2. + resized_gpu.rows/2.), FONT_HERSHEY_DUPLEX, 0.8, tracked_color, 2, 8, false);
            circle(frameDisp, Point(face_msg.x + resized_gpu.cols/2., face_msg.y + resized_gpu.rows/2.), 2, tracked_color, -1);
            circle(frameDisp, Point(face_msg.x + resized_gpu.cols/2., face_msg.y + resized_gpu.rows/2.), sqrt(max(kf.errorCovPost.at<float>(0,0),kf.errorCovPost.at<float>(1,1))), tracked_color, 1);
        }
        else {
            count_face++;
        }
    }
    else {
        if (count_face >= 10) {     // When face has been tracked for 10 frames
            face_msg.x = (int) kf.statePost.at<float>(0,0);
            face_msg.y = (int) kf.statePost.at<float>(1,0);
            face_msg.size = (int) kf.statePost.at<float>(2,0);
            face_msg.direction = face_state;
            face_pub_.publish(face_msg);

            // draw on the image for debug

            Scalar tracked_color = (face_state == 0 ? Scalar(64,255,64) : (face_state == 1) ? Scalar(64,64,255) : Scalar(255,64,64));
            rectangle(frameDisp, Point(face_msg.x - face_msg.size/2. + resized_gpu.cols/2., face_msg.y - face_msg.size/2. + resized_gpu.rows/2.),
                    Point(face_msg.x + face_msg.size/2. + resized_gpu.cols/2., face_msg.y + face_msg.size/2. + resized_gpu.rows/2.),
                    tracked_color, 2);
            sprintf(buff, "%d, %d, %d", face_msg.x, face_msg.y, face_msg.size);
            putText(frameDisp, buff, Point(face_msg.x - face_msg.size/2. + resized_gpu.cols/2., face_msg.y - face_msg.size/2. + resized_gpu.rows/2.), FONT_HERSHEY_DUPLEX, 0.8, tracked_color, 2, 8, false);
            circle(frameDisp, Point(face_msg.x + resized_gpu.cols/2., face_msg.y + resized_gpu.rows/2.), 2, tracked_color, -1);
            circle(frameDisp, Point(face_msg.x + resized_gpu.cols/2., face_msg.y + resized_gpu.rows/2.), sqrt(max(kf.errorCovPost.at<float>(0,0),kf.errorCovPost.at<float>(1,1))), tracked_color, 1);
        }
        else {
            count_face = 0;     // reset the counter when the face has not been registered & then not found
        }
    }

    displayState(frameDisp, helpScreen, findLargestObject, filterRects, fps);
    imshow("result", frameDisp);
    sensor_msgs::ImagePtr out_msg = cv_bridge::CvImage(msg->header, "bgr8", frameDisp).toImageMsg();
    image_pub_.publish(out_msg);
    
    char key = (char)waitKey(1);

    switch (key)
    {
    case 27:
        running = false;
        break;
    case 'm':
    case 'M':
        findLargestObject = !findLargestObject;
        break;
    case 'f':
    case 'F':
        filterRects = !filterRects;
        break;
    case '1':
        scaleFactor *= 1.05;
        break;
    case 'q':
    case 'Q':
        scaleFactor /= 1.05;
        break;
    case 'h':
    case 'H':
        helpScreen = !helpScreen;
        break;
    case '+':
        neighborThreshold += 1;
        neighborThreshold = min(neighborThreshold, 16);
        cout << "Increased threshold to " << neighborThreshold << endl;
        break;
    case '-':
        neighborThreshold -= 1;
        neighborThreshold = max(neighborThreshold, 0);
        cout << "Decreased threshold to " << neighborThreshold << endl;
        break;
    }
}

void FacesDetector_GPU::groupFaces(vector<Rect> &inputFront, vector<Rect> &inputLeft, vector<Rect> &inputRight, Rect &output, float* best_prob,
        vector<Rect> &outputList, vector<vector<float> > &probs, int groupThreshold, double eps)
{
    // Adapted from groupRectangles(inputList, threshold, eps)
    outputList.clear();

    size_t lengthF = inputFront.size();
    size_t lengthL = inputLeft.size();
    size_t lengthR = inputRight.size();

    vector<Rect> concatinatedList;
    concatinatedList.insert(concatinatedList.end(), inputFront.begin(), inputFront.end());
    concatinatedList.insert(concatinatedList.end(), inputLeft.begin(), inputLeft.end());
    concatinatedList.insert(concatinatedList.end(), inputRight.begin(), inputRight.end());

    vector<int> labels;
    int nclasses = partition(concatinatedList, labels, SimilarRects(eps));

    vector<Rect> rrects(nclasses);
    vector<int> rweights(nclasses, 0);
    vector<int> frontCounts(nclasses, 0);
    vector<int> leftCounts(nclasses, 0);
    vector<int> rightCounts(nclasses, 0);
    vector<int> rejectLevels(nclasses, 0);
    vector<double> rejectWeights(nclasses, DBL_MIN);
    int i, j, nlabels = (int)labels.size();

    // Front
    for( i = 0; i < lengthF; i++ )
    {
        if (concatinatedList[i].width > 150)
            continue;               // remove too large face
        int cls = labels[i];
        rrects[cls].x += concatinatedList[i].x;
        rrects[cls].y += concatinatedList[i].y;
        rrects[cls].width += concatinatedList[i].width;
        rrects[cls].height += concatinatedList[i].height;
        rweights[cls]++;
        frontCounts[cls]++;
    }

    // Left
    for( ; i < lengthF + lengthL; i++ )
    {
        if (concatinatedList[i].width > 150)
            continue;               // remove too large face
        int cls = labels[i];
        rrects[cls].x += concatinatedList[i].x;
        rrects[cls].y += concatinatedList[i].y;
        rrects[cls].width += concatinatedList[i].width;
        rrects[cls].height += concatinatedList[i].height;
        rweights[cls]++;
        leftCounts[cls]++;
    }

    // Right
    for( ; i < nlabels; i++ )
    {
        if (concatinatedList[i].width > 150)
            continue;               // remove too large face
        int cls = labels[i];
        rrects[cls].x += concatinatedList[i].x;
        rrects[cls].y += concatinatedList[i].y;
        rrects[cls].width += concatinatedList[i].width;
        rrects[cls].height += concatinatedList[i].height;
        rweights[cls]++;
        rightCounts[cls]++;
    }

    // Averaged rectangles
    for( i = 0; i < nclasses; i++ )
    {
        Rect r = rrects[i];
        float s = 1.f/rweights[i];
        rrects[i] = Rect(saturate_cast<int>(r.x*s),
             saturate_cast<int>(r.y*s),
             saturate_cast<int>(r.width*s),
             saturate_cast<int>(r.height*s));
    }

    int maxCounts = 0;
    int maxClass = -1;
    for( i = 0; i < nclasses; i++ )
    {
        Rect r1 = rrects[i];
        int n1 = rweights[i];
        if( n1 <= groupThreshold )
            continue;
        if( n1 > maxCounts) {
            maxCounts = n1;
            maxClass = i;
        }
        // filter out small face rectangles inside large rectangles
        for( j = 0; j < nclasses; j++ )
        {
            int n2 = rweights[j];

            if( j == i || n2 <= groupThreshold )
                continue;
            Rect r2 = rrects[j];

            int dx = saturate_cast<int>( r2.width * eps );
            int dy = saturate_cast<int>( r2.height * eps );

            if( i != j &&
                r1.x >= r2.x - dx &&
                r1.y >= r2.y - dy &&
                r1.x + r1.width <= r2.x + r2.width + dx &&
                r1.y + r1.height <= r2.y + r2.height + dy &&
                (n2 > std::max(3, n1) || n1 < 3) )
                break;
        }

        if( j == nclasses )
        {
            outputList.push_back(r1);
            float s = 1.f/rweights[i];
            vector<float> prob;
            prob.push_back(s*frontCounts[i]);
            prob.push_back(s*leftCounts[i]);
            prob.push_back(s*rightCounts[i]);
            probs.push_back(prob);
        }
    }

    if (maxClass >= 0) {
        output = rrects[maxClass];
        float s = 1.f/rweights[maxClass];
        best_prob[0] = s*frontCounts[maxClass];
        best_prob[1] = s*leftCounts[maxClass];
        best_prob[2] = s*rightCounts[maxClass];
    }
}

void FacesDetector_GPU::update() {
    if (has_face) {         // Predict face
        ros::Time new_time = ros::Time::now();
        float delta_t = (new_time - last_time).toSec();
        last_time = new_time;
        kf.transitionMatrix.at<float>(0,3) = delta_t;
        kf.transitionMatrix.at<float>(1,4) = delta_t;
        kf.transitionMatrix.at<float>(2,5) = delta_t;
        kf.predict();
        if (sqrt(kf.errorCovPost.at<float>(0,0)) > kf.statePost.at<float>(2,0) || sqrt(kf.errorCovPost.at<float>(1,1)) > kf.statePost.at<float>(2,0) || kf.errorCovPost.at<float>(2,2) > 2500. ||
            kf.statePost.at<float>(0,0) + kf.statePost.at<float>(2,0)/2. < -360. || kf.statePost.at<float>(0,0) + kf.statePost.at<float>(2,0)/2. > 360. ||
            kf.statePost.at<float>(1,0) + kf.statePost.at<float>(2,0)/2. < -240. || kf.statePost.at<float>(1,0) + kf.statePost.at<float>(2,0)/2. > 240. ||
            kf.statePost.at<float>(2,0) < 20.) {        // SD > size, out of frame, too small face --> remove & reset face
            has_face = false;
            count_face = 0;
            loss_time = new_time;
            prev_face = Point3f(kf.statePost.at<float>(0,0), kf.statePost.at<float>(1,0), kf.statePost.at<float>(2,0));
            ROS_INFO("Face removed");
        }
    }
}
