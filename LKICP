#include <iostream>
#include <fstream>
#include <list>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace cv;

void detect_feature_points(const Mat & color, list<Point2f> &keypoints);
Point3f pixel2cam ( const Point2d& p, const Eigen::Matrix3d& K, const double depth );
void pose_estimation_3d3d ( const vector<Point3f>& pts1, const vector<Point3f>& pts2, Eigen::Matrix3d& R, Eigen::Vector3d& t);
typedef Eigen::Matrix<double, 7,1> vec7d;

double depthscale = 5000.0;
Mat K_m = ( Mat_<double> ( 3, 3) <<517.3, 0, 318.6, 0, 516.5, 255.3, 0, 0, 1);//TUM官方给出内参


int main(int argc, char** argv)
{
    Eigen::Matrix3d K;
    K <<517.3, 0, 318.6, 0, 516.5, 255.3, 0, 0, 1;//TUM官方给出内参
    double threhold1 = 2.0;
    double threhold2 = 0.7;
    string file = "./associate.txt";
    ifstream fin(file);
    Mat color, depth, last_color, last_depth;
    Eigen::Matrix3d Rwc, Rcw;
    Eigen::Vector3d twc, tcw;
    vector<vec7d> poses;
    vector<string> time_stamps;
    list<Point2f> keypoints;

    for(int index = 0; index < 1300; index++)
    {
        string rgb_file, depth_file, time_rgb, time_depth;
        fin>>time_rgb>>rgb_file>>time_depth>>depth_file;
        color = cv::imread( "./"+rgb_file );
        depth = cv::imread( "./"+depth_file, -1 );
    //第一帧，检测特征点
    vector<Point2f> prev_keypoints, next_keypoints, next_keypoints2;
    if(index == 0)
    {
        detect_feature_points(color, keypoints);
        //初始真实位姿给定
//        Rwc = Eigen::Quaterniond( -0.3811, 0.8219, -0.3912, 0.1615);
//        twc<<-0.8683, 0.6026, 1.5627;
        //初始位姿设定为单位位姿（相机坐标为世界坐标）
        Rwc = Eigen::Matrix3d::Identity();
        twc = Eigen::Vector3d::Zero();
        Eigen::Quaterniond q(Rwc);
        vec7d pose = vec7d::Zero();
        pose<<twc(0), twc(1), twc(2),q.x(), q.y(), q.z(), q.w();
        poses.push_back(pose);
        time_stamps.push_back(time_rgb);
        last_color = color;
        last_depth = depth;
    }

    if ( color.data==nullptr || depth.data==nullptr )
        continue;

    vector<unsigned char> status;
    vector<float> error;
    prev_keypoints.clear();
    for ( auto kp:keypoints )
        prev_keypoints.push_back(kp);
//    cout<<"last color name= "<<time_stamps[index-1]<<endl;
    imshow("lastcolor ", last_color);
    waitKey(1);
//    cout<<"last color name= "<<time_stamps[index]<<endl;
    imshow("color", color);
    waitKey(1);
    calcOpticalFlowPyrLK( last_color, color, prev_keypoints, next_keypoints, status, error );
    // 把跟丢的点删掉，[并把keypoints中坐标换成第二帧跟踪到的特征点]
    int i=0;
    for ( auto iter=keypoints.begin(); iter!=keypoints.end(); i++)  //带两个变量的循环,学习一下
    {
        if ( status[i] == 0 )   
        {
            iter = keypoints.erase(iter);   //vector的成员函数erase()的返回值是指针，指向被删除元素后的元素
            continue;   
        }
//        *iter = next_keypoints[i];  //根据next_keypoints的status来判断是否跟丢,保证世界坐标中的点与当前跟踪到的点是匹配的
        next_keypoints2.push_back(next_keypoints[i]);
        iter++;
    }
    prev_keypoints.clear();
    //下面的操作是为了便于对keypoints中的点访问坐标并读取深度
    for ( auto kp:keypoints )
        prev_keypoints.push_back(kp);

    //对上一帧源像素点，若深度为0，则在next_keypoints2中删掉；其他的点计算相机坐标入栈kpvec3dcam_last、世界坐标入栈kpvec3dwor_last
    vector<Point3f> kpvec3dcam_last, kpvec3dwor_last;
    int j = 0;
    for ( auto iter=next_keypoints2.begin(); iter!=next_keypoints2.end(); j++ )
    {
        Point2f kp = prev_keypoints[j];
        unsigned int d = last_depth.ptr<unsigned short> ( int ( kp.y ) ) [ int ( kp.x ) ];  //x是向右的，y是向下的，所以y才是行，x是列
        if (d == 0)
        {
            iter = next_keypoints2.erase(iter);
            continue;
        }
        Point3f p1 = pixel2cam(kp, K, d);
        kpvec3dcam_last.push_back(p1);
        Eigen::Vector3d pt1(p1.x, p1.y, p1.z);
        Eigen::Matrix<double, 3, 1> pts = Rwc*pt1 + twc;
        kpvec3dwor_last.push_back(Point3f(pts(0,0),pts(1,0),pts(2,0)));
        iter++;
    }

    if(kpvec3dwor_last.size() < 4 || next_keypoints2.size() < 4)
    {
        cout<<"输入点对数目小于4，无法估计位姿"<<endl;
        break;
    }

    ///////PnP部分
    /// 1.求解，输入为上一帧世界坐标、当前帧像素坐标；输出为变换矩阵R, t
    Mat rvec;
    Mat tvec;
    Mat RR;
    Eigen::Matrix3d R_cwesti;
    Eigen::Vector3d t_cwesti;
//    solvePnPRansac(kpvec3dwor_last, next_keypoints2, K_m, distcoeffs, rvec, tvec, false );
    solvePnPRansac(kpvec3dwor_last, next_keypoints2, K_m, Mat(), rvec, tvec, false );//这里的rvec, tvec是旋转向量、平移向量，要转换为旋转矩阵(Eigen::Matrix类型)
    //solvePnP (worldpoints, next_keypoints2, K, Mat(), rvec, tvec, false);
    Rodrigues(rvec, RR);
    cout<<"RR= "<<RR<<endl;
    cv2eigen(RR, R_cwesti);
    t_cwesti<<tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2);
    ///2.判断外参是否正确
    /// 2.1重投影（用估计的pose将第一帧世界坐标点投影到第二帧求得像素坐标，入栈pts1）
    vector<Point2f> pts1;   //重投影点栈
    /***********************************************/
    //pts1.clear();         pts1是否会一直增加？不会，属于循环内的局部变量
    /***********************************************/
    for ( auto kp:kpvec3dwor_last )
    {
        Eigen::Vector3d pointWorld;
        Eigen::Vector3d pointPix;
        pointWorld<< kp.x, kp.y, kp.z;
        pointPix = K*(R_cwesti * pointWorld + t_cwesti);  //K_scale中主对角线元素要乘以depthscale，并不需要，没有理解depthscale的意义
        pts1.push_back( Point2f( pointPix(0,0)/pointPix(2,0), pointPix(1,0)/pointPix(2,0) ) );  //归一化的u,v是像素坐标
    }
    ///2.2比较重投影点pts1与next_keypoints2中点的距离
    int i2=0;   //满足距离要求的点数目
    for ( int i =0; i<next_keypoints2.size(); i++)
    {
        double i1 = sqrt((next_keypoints2[i].x-pts1[i].x)*(next_keypoints2[i].x-pts1[i].x)+(next_keypoints2[i].y-pts1[i].y)*(next_keypoints2[i].y-pts1[i].y));
        if(i1<=threhold1)
            i2++;
    }
    double percent = (double)i2/next_keypoints2.size();
    cout<<"percent of satisfied points =  "<<percent<<endl;
    if(percent < threhold2) //threhold2要为float，不能是int
    {
        cout<<"位姿估计不符合要求，需转为ICP求解"<<endl;
        ////////ICP部分
        /// 1.求当前帧相机坐标, 像素坐标next_keypoints2
        vector<Point3f> kpvec3dcam_next;
        for(auto kp:next_keypoints2)
        {
            unsigned int d = depth.ptr<unsigned short> ( int ( kp.y ) ) [ int ( kp.x ) ];
            Point3f p1 = pixel2cam(kp, K, d);   //p1:相机坐标Point3f格式
            kpvec3dcam_next.push_back(p1);
        }
        /// 2.求解：输入为上一帧世界坐标，当前帧相机坐标，输出为从相机坐标到世界坐标的变换矩阵R,t
        pose_estimation_3d3d(kpvec3dwor_last, kpvec3dcam_next, Rwc, twc);
        cout<<"Rwc = "<<Eigen::Quaterniond(Rwc).coeffs().transpose()<<endl<<"twc= "<<twc;
        last_color = color;
        last_depth = depth;
        keypoints.clear();
        detect_feature_points(color, keypoints);

        Eigen::Quaterniond q = Eigen::Quaterniond(Rwc);
        vec7d pose = vec7d::Zero();
        pose<<twc(0), twc(1), twc(2),q.x(), q.y(), q.z(), q.w();
        poses.push_back(pose);
         time_stamps.push_back(time_rgb);
        continue;
    }

    ///3.位姿估计符合要求，更新位姿并存储,（当前帧相对于世界坐标）
    Rcw = R_cwesti;
    tcw = t_cwesti;
    Rwc = Rcw.inverse();
    twc = -Rwc*tcw;

    Eigen::Quaterniond q = Eigen::Quaterniond(Rwc);
    vec7d pose = vec7d::Zero();
    pose<<twc(0), twc(1), twc(2),q.x(), q.y(), q.z(), q.w();
    poses.push_back(pose);
     time_stamps.push_back(time_rgb);
    //更新角点(用三维点重投影得到点作为角点，即pts1)，也是下一帧lk跟踪的角点，仅在PnP生效时如此
    keypoints.clear();
    for ( auto kp:pts1 )    //pts1是重投影的角点，将其入特征点栈
    {
        keypoints.push_back(kp);
    }
    last_color = color;
    last_depth = depth;
}   //循环结束
    ofstream outfile;
    outfile.open("data.txt", ios::binary | ios::app | ios::in | ios::out);
    for(int i = 0; i<poses.size(); i++)
    {
        string time_rgb = time_stamps[i];
        vec7d pose = poses[i];
        outfile<<time_rgb<<"  "<<pose.transpose()<<endl;
    }
        outfile.close();//关闭文件，保存文件。
}//main结束



void detect_feature_points(const Mat & color, list<Point2f> &keypoints)
{
    vector<cv::KeyPoint> kps;
    //cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
    cv::Ptr<ORB> orb = ORB::create();
    orb->detect( color, kps );
    for ( auto kp:kps )
    {
        keypoints.push_back( kp.pt );
        //prev_keypoints.push_back(kp.pt);
    }
}

Point3f pixel2cam ( const Point2d& p, const Eigen::Matrix3d& K, const double depth )
{
    double d = depth/depthscale;
    return Point3f
           (
               d*( p.x - K( 0,2 ) ) / K( 0,0 ),
               d*( p.y - K( 1,2 ) ) / K( 1,1 ),
               d );
}

void pose_estimation_3d3d (
    const vector<Point3f>& pts1,
    const vector<Point3f>& pts2,
    Eigen::Matrix3d& R, Eigen::Vector3d& t
)   //求pts2到pts1的变换
{
    Point3f p1, p2;     // 质心
    int N = pts1.size();
    for ( int i=0; i<N; i++ )
    {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = Point3f( Vec3f(p1) /  N);
    p2 = Point3f( Vec3f(p2) / N);
    vector<Point3f>     q1 ( N ), q2 ( N ); // 去质心坐标
    for ( int i=0; i<N; i++ )
    {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }


    // 计算矩阵 W=q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for ( int i=0; i<N; i++ )
    {
        W += Eigen::Vector3d ( q1[i].x, q1[i].y, q1[i].z ) * Eigen::Vector3d ( q2[i].x, q2[i].y, q2[i].z ).transpose();
    }
    cout<<"W="<<W<<endl;

    // W的奇异值分解
    Eigen::JacobiSVD<Eigen::Matrix3d> svd ( W, Eigen::ComputeFullU|Eigen::ComputeFullV );
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    //cout<<"U="<<U<<endl;
    //cout<<"V="<<V<<endl;

    R = U* ( V.transpose() );
    t = Eigen::Vector3d ( p1.x, p1.y, p1.z ) - R * Eigen::Vector3d ( p2.x, p2.y, p2.z );

//    // 将R转换为类型 cv::Mat
//    R = ( Mat_<double> ( 3,3 ) <<
//          R_ ( 0,0 ), R_ ( 0,1 ), R_ ( 0,2 ),
//          R_ ( 1,0 ), R_ ( 1,1 ), R_ ( 1,2 ),
//          R_ ( 2,0 ), R_ ( 2,1 ), R_ ( 2,2 )
//        );
//    t = ( Mat_<double> ( 3,1 ) << t_ ( 0,0 ), t_ ( 1,0 ), t_ ( 2,0 ) );
}
