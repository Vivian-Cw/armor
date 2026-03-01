#include <iostream>
#include <opencv2/opencv.hpp> 
#include <armor_detect.hpp>

using namespace std;
using namespace cv;

int main(int argc,char** argv){
    // 设置视频文件路径
    string video_path="R2.mp4";
    // 打开视频
    VideoCapture cap(video_path);
    if(!cap.isOpened()){
        cerr<<"无法打开视频文件："<<video_path<<endl;
        return -1;
    }
    // 获取视频频率
    double fps=cap.get(CAP_PROP_FPS);
    int delay=1000/fps;// 每帧延迟（ms）

    cout<<"视频频率："<<fps<<"FPS"<<endl;
    
    Mat frame;
    while(true){
        // 读取一帧
        cap >> frame;
        // 视频结束，循环播放
        if(frame.empty()){
            cap.set(CAP_PROP_POS_FRAMES,0);
            cap >> frame;
            if(frame.empty()){
                cerr<<"无法读取视频帧"<<endl;
                break;
            }
        }
        //装甲板识别
        ArmorDetect(frame);
        //显示结果
        imshow("ArmorDetect",frame);
        //按‘q’键退出
        if(waitKey(delay)=='q'){
            break;
        }
    }
    cap.release();
    destroyAllWindows();

    return 0;
}