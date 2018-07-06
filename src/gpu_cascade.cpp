#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/Point32.h>
#include <iostream>
#include <iomanip>
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"

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


static void displayState(Mat &canvas, bool bHelp, bool bGpu, bool bLargestFace, bool bFilter, double fps)
{
    Scalar fontColorRed = CV_RGB(255,0,0);
    Scalar fontColorNV  = CV_RGB(118,185,0);

    ostringstream ss;
    ss << "FPS = " << setprecision(1) << fixed << fps;
    matPrint(canvas, 0, fontColorRed, ss.str());
    ss.str("");
    ss << "[" << canvas.cols << "x" << canvas.rows << "], " <<
        (bGpu ? "GPU, " : "CPU, ") <<
        (bLargestFace ? "OneFace, " : "MultiFace, ") <<
        (bFilter ? "Filter:ON" : "Filter:OFF");
    matPrint(canvas, 1, fontColorRed, ss.str());

    // by Anatoly. MacOS fix. ostringstream(const string&) is a private
    // matPrint(canvas, 2, fontColorNV, ostringstream("Space - switch GPU / CPU"));
    if (bHelp)
    {
        matPrint(canvas, 2, fontColorNV, "Space - switch GPU / CPU");
        matPrint(canvas, 3, fontColorNV, "M - switch OneFace / MultiFace");
        matPrint(canvas, 4, fontColorNV, "F - toggle rectangles Filter");
        matPrint(canvas, 5, fontColorNV, "H - toggle hotkeys help");
        matPrint(canvas, 6, fontColorNV, "1/Q - increase/decrease scale");
    }
    else
    {
        matPrint(canvas, 2, fontColorNV, "H - toggle hotkeys help");
    }
}


class CascadeDetector_GPU
{
public:
    CascadeDetector_GPU(ros::NodeHandle nh, ros::NodeHandle nh_private);
    void imageCallback(const sensor_msgs::Image::ConstPtr& msg);
private:
    ros::NodeHandle nh_, nh_private_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
    ros::Publisher face_pub_;
    CascadeClassifier_GPU cascade_gpu;
    CascadeClassifier cascade_cpu;
    /* parameters */
    bool running;
    bool useGPU;
    double scaleFactor;
    bool findLargestObject;
    bool filterRects;
    bool helpScreen;
    
    int detections_num;
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
    CascadeDetector_GPU face_detector(nh, nh_private);
    ros::spin();
    
    return 0;
}

CascadeDetector_GPU::CascadeDetector_GPU(ros::NodeHandle nh, ros::NodeHandle nh_private) : nh_(nh), nh_private_(nh_private), it_(nh_){
    string cascade_name;
    string default_cascade = "/home/veerachart/opencv/data/haarcascades_GPU/haarcascade_frontalface_alt.xml";
    //nh_.param<string>("cascade_name", cascade_name, default_cascade);
    nh_private_.param("cascade_name", cascade_name, default_cascade);
    cout << cascade_name << endl;
    if (!cascade_gpu.load(cascade_name))
    {
        ROS_ERROR("Could not load cascade classifier \"%s\"", cascade_name.c_str());
        return;
    }

    if (!cascade_cpu.load(cascade_name))
    {
        ROS_ERROR("Could not load cascade classifier \"%s\"", cascade_name.c_str());
        return;
    }
    
    namedWindow("result", 1);
    
    image_sub_ = it_.subscribe("/image", 1, &CascadeDetector_GPU::imageCallback, this);
    image_pub_ = it_.advertise("/face_detect", 1);
    
    face_pub_ = nh_.advertise<geometry_msgs::PolygonStamped>("/face", 1);
    
    running = true;
    useGPU = true;
    scaleFactor = 1.0;
    findLargestObject = false;
    filterRects = true;
    helpScreen = false;
}

void CascadeDetector_GPU::imageCallback(const sensor_msgs::Image::ConstPtr& msg) {
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

    Mat frame_cpu, gray_cpu, resized_cpu, faces_downloaded, frameDisp;
    vector<Rect> facesBuf_cpu;

    GpuMat frame_gpu, gray_gpu, resized_gpu, facesBuf_gpu;
    
    (cv_ptr->image).copyTo(frame_cpu);
    frame_gpu.upload(cv_ptr->image);
    
    convertAndResize(frame_gpu, gray_gpu, resized_gpu, scaleFactor);
    convertAndResize(frame_cpu, gray_cpu, resized_cpu, scaleFactor);
    
    TickMeter tm;
    tm.start();

    if (useGPU)
    {
        //cascade_gpu.visualizeInPlace = true;
        cascade_gpu.findLargestObject = findLargestObject;

        detections_num = cascade_gpu.detectMultiScale(resized_gpu, facesBuf_gpu, 1.2,
                                                      (filterRects || findLargestObject) ? 4 : 0);
        facesBuf_gpu.colRange(0, detections_num).download(faces_downloaded);
    }
    else
    {
        Size minSize = cascade_gpu.getClassifierSize();
        cascade_cpu.detectMultiScale(resized_cpu, facesBuf_cpu, 1.2,
                                     (filterRects || findLargestObject) ? 4 : 0,
                                     (findLargestObject ? CV_HAAR_FIND_BIGGEST_OBJECT : 0)
                                        | CV_HAAR_SCALE_IMAGE,
                                     minSize);
        detections_num = (int)facesBuf_cpu.size();
    }

    geometry_msgs::PolygonStamped faces;
    geometry_msgs::Polygon* polygon = &faces.polygon;
    geometry_msgs::Point32 point;
    if (!useGPU && detections_num)
    {
        for (int i = 0; i < detections_num; ++i)
        {
            Rect r = facesBuf_cpu[i];
            rectangle(resized_cpu, r, Scalar(255));
            point.x = float(r.x);
            point.y = float(r.y);
            point.z = float(r.width);
            polygon->points.push_back(point);
        }
    }

    if (useGPU)
    {
        resized_gpu.download(resized_cpu);

         for (int i = 0; i < detections_num; ++i)
         {
            Rect r = faces_downloaded.ptr<cv::Rect>()[i];
            rectangle(resized_cpu, r, Scalar(255));
            point.x = float(r.x);
            point.y = float(r.y);
            point.z = float(r.width);
            polygon->points.push_back(point);
         }
    }

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
    
    if (useGPU) {
        if (detections_num) {
            Mat cropped_face(resized_cpu, faces_downloaded.ptr<cv::Rect>()[0]);
            imshow("Face0", cropped_face);
        }
    }
    
    cvtColor(resized_cpu, frameDisp, CV_GRAY2BGR);
    displayState(frameDisp, helpScreen, useGPU, findLargestObject, filterRects, fps);
    imshow("result", frameDisp);
    sensor_msgs::ImagePtr out_msg = cv_bridge::CvImage(msg->header, "bgr8", frameDisp).toImageMsg();
    image_pub_.publish(out_msg);
    faces.header = msg->header;
    face_pub_.publish(faces);
    
    char key = (char)waitKey(5);

    switch (key)
    {
    case 27:
        running = false;
        break;
    case ' ':
        useGPU = !useGPU;
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
    }
}
