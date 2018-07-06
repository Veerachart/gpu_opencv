#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/Point32.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

bool help_showed = false;

class App
{
public:
    App(ros::NodeHandle nh, ros::NodeHandle nh_private);
    void imageCallback (const sensor_msgs::Image::ConstPtr& msg);

    void handleKey(char key);

    void hogWorkBegin();
    void hogWorkEnd();
    string hogWorkFps() const;

    void workBegin();
    void workEnd();
    string workFps() const;

    string message() const;

private:
    App operator=(App&);

    ros::NodeHandle nh_, nh_private_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
    ros::Publisher human_pub_;
    
    bool running;

    bool use_gpu;
    bool make_gray;
    double scale;
    int gr_threshold;
    int nlevels;
    double hit_threshold;
    bool gamma_corr;
    
    cv::gpu::HOGDescriptor gpu_hog;
    cv::HOGDescriptor cpu_hog;
    Size win_size;
    Size win_stride;

    int64 hog_work_begin;
    double hog_work_fps;

    int64 work_begin;
    double work_fps;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "human_detect_blimp");
    ros::NodeHandle nh, nh_private("~");
    App app(nh, nh_private);
    ROS_INFO("INIT APP");
    /*try
    {
        App app;
        ROS_INFO("INIT APP");
    }
    catch (const Exception& e) { return cout << "error: "  << e.what() << endl, 1; }
    catch (const exception& e) { return cout << "error: "  << e.what() << endl, 1; }
    catch(...) { return cout << "unknown exception" << endl, 1; }*/
    ROS_INFO("Spin");
    ros::spin();
    
    return 0;
}


App::App(ros::NodeHandle nh, ros::NodeHandle nh_private)  : nh_(nh), nh_private_(nh_private), it_(nh_)
{
    cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());

    cout << "\nControls:\n"
         << "\tESC - exit\n"
         << "\tm - change mode GPU <-> CPU\n"
         << "\tg - convert image to gray or not\n"
         << "\t1/q - increase/decrease HOG scale\n"
         << "\t2/w - increase/decrease levels count\n"
         << "\t3/e - increase/decrease HOG group threshold\n"
         << "\t4/r - increase/decrease hit threshold\n"
         << endl;

    use_gpu = true;
    make_gray = false;
    scale = 1.05;
    gr_threshold = 2;
    nlevels = 64;

    hit_threshold = 0.;

    gamma_corr = true;

    cout << "Scale: " << scale << endl;
    cout << "Group threshold: " << gr_threshold << endl;
    cout << "Levels number: " << nlevels << endl;
    cout << "Win width: 64" << endl;
    cout << "Win stride: (8, 8)\n";
    cout << "Hit threshold: " << hit_threshold << endl;
    cout << "Gamma correction: " << gamma_corr << endl;
    cout << endl;
    
    image_sub_ = it_.subscribe("/image", 1, &App::imageCallback, this);
    image_pub_ = it_.advertise("/body_detect", 1);
    
    human_pub_ = nh_.advertise<geometry_msgs::PolygonStamped>("/human", 1);
    
    running = true;

    win_size = Size(64, 128); //(64, 128) or (48, 96)
    //win_size = Size(48, 56);
    win_stride = Size(8, 8);

    // Create HOG descriptors and detectors here
    vector<float> detector;
    if (win_size == Size(64, 128))
        detector = cv::gpu::HOGDescriptor::getPeopleDetector64x128();
    else
        //detector = vector<float>(upper_detector, upper_detector + sizeof(upper_detector)/sizeof(upper_detector[0]));
        detector = cv::gpu::HOGDescriptor::getPeopleDetector48x96();

    gpu_hog = cv::gpu::HOGDescriptor(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9,
                                     cv::gpu::HOGDescriptor::DEFAULT_WIN_SIGMA, 0.2, gamma_corr,
                                     cv::gpu::HOGDescriptor::DEFAULT_NLEVELS);
    cpu_hog = cv::HOGDescriptor(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9, 1, -1,
                                HOGDescriptor::L2Hys, 0.2, gamma_corr, cv::HOGDescriptor::DEFAULT_NLEVELS);
    gpu_hog.setSVMDetector(detector);
    cpu_hog.setSVMDetector(detector);
}


void App::imageCallback (const sensor_msgs::Image::ConstPtr& msg)
{
    if (!running)
        ros::shutdown();
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    ///////////
    Mat img_aux, img, img_to_show;
    gpu::GpuMat gpu_img;
    workBegin();

    // Change format of the image
    if (make_gray) cvtColor(cv_ptr->image, img_aux, CV_BGR2GRAY);
    else if (use_gpu) cvtColor(cv_ptr->image, img_aux, CV_BGR2BGRA);
    else cv_ptr->image.copyTo(img_aux);

    // Resize image
    img = img_aux;
    img_to_show = img;

    gpu_hog.nlevels = nlevels;
    cpu_hog.nlevels = nlevels;

    vector<Rect> found;

    // Perform HOG classification
    hogWorkBegin();
    if (use_gpu)
    {
        gpu_img.upload(img);
        gpu_hog.detectMultiScale(gpu_img, found, hit_threshold, win_stride,
                                 Size(0, 0), scale, gr_threshold);
    }
    else cpu_hog.detectMultiScale(img, found, hit_threshold, win_stride,
                                  Size(0, 0), scale, gr_threshold);
    hogWorkEnd();

    geometry_msgs::PolygonStamped bodies;
    geometry_msgs::Polygon* polygon = &bodies.polygon;
    geometry_msgs::Point32 point;
    // Draw positive classified windows
    for (size_t i = 0; i < found.size(); i++)
    {
        Rect r = found[i];
        rectangle(img_to_show, r.tl(), r.br(), CV_RGB(0, 255, 0), 3);
        point.x = float(r.x);
        point.y = float(r.y);
        point.z = float(r.width);
        polygon->points.push_back(point);
    }

    if (use_gpu)
        putText(img_to_show, "Mode: GPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
    else
        putText(img_to_show, "Mode: CPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
    putText(img_to_show, "FPS (HOG only): " + hogWorkFps(), Point(5, 65), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
    putText(img_to_show, "FPS (total): " + workFps(), Point(5, 105), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
    imshow("opencv_gpu_hog", img_to_show);
    sensor_msgs::ImagePtr out_msg = cv_bridge::CvImage(msg->header, "bgra8", img_to_show).toImageMsg();
    image_pub_.publish(out_msg);
    bodies.header = msg->header;
    human_pub_.publish(bodies);

    workEnd();

    handleKey((char)waitKey(3));
}


void App::handleKey(char key)
{
    switch (key)
    {
    case 27:
        running = false;
        break;
    case 'm':
    case 'M':
        use_gpu = !use_gpu;
        cout << "Switched to " << (use_gpu ? "CUDA" : "CPU") << " mode\n";
        break;
    case 'g':
    case 'G':
        make_gray = !make_gray;
        cout << "Convert image to gray: " << (make_gray ? "YES" : "NO") << endl;
        break;
    case '1':
        scale *= 1.05;
        cout << "Scale: " << scale << endl;
        break;
    case 'q':
    case 'Q':
        scale /= 1.05;
        cout << "Scale: " << scale << endl;
        break;
    case '2':
        nlevels++;
        cout << "Levels number: " << nlevels << endl;
        break;
    case 'w':
    case 'W':
        nlevels = max(nlevels - 1, 1);
        cout << "Levels number: " << nlevels << endl;
        break;
    case '3':
        gr_threshold++;
        cout << "Group threshold: " << gr_threshold << endl;
        break;
    case 'e':
    case 'E':
        gr_threshold = max(0, gr_threshold - 1);
        cout << "Group threshold: " << gr_threshold << endl;
        break;
    case '4':
        hit_threshold+=0.25;
        cout << "Hit threshold: " << hit_threshold << endl;
        break;
    case 'r':
    case 'R':
        hit_threshold = max(0.0, hit_threshold - 0.25);
        cout << "Hit threshold: " << hit_threshold << endl;
        break;
    case 'c':
    case 'C':
        gamma_corr = !gamma_corr;
        cout << "Gamma correction: " << gamma_corr << endl;
        break;
    }
}


inline void App::hogWorkBegin() { hog_work_begin = getTickCount(); }

inline void App::hogWorkEnd()
{
    int64 delta = getTickCount() - hog_work_begin;
    double freq = getTickFrequency();
    hog_work_fps = freq / delta;
}

inline string App::hogWorkFps() const
{
    stringstream ss;
    ss << hog_work_fps;
    return ss.str();
}


inline void App::workBegin() { work_begin = getTickCount(); }

inline void App::workEnd()
{
    int64 delta = getTickCount() - work_begin;
    double freq = getTickFrequency();
    work_fps = freq / delta;
}

inline string App::workFps() const
{
    stringstream ss;
    ss << work_fps;
    return ss.str();
}
