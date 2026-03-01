#include "opencv2/opencv.hpp"
#include "iostream"
#include "vector"
#include "cmath"
#include <cstdarg>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

using namespace cv;
using namespace std;

// 灯条描述类
class LightDescription{
public:
    float width,length,angle,area;
    Point2f center;
    LightDescription(){} //默认构造函数
    LightDescription(const RotatedRect &light){
        width = light.size.width;
        length = light.size.height;
        angle = light.angle;
        area = light.size.area();
        center = light.center;
    }
};

// 对点进行排序（左上、右上、右下、左下）
void sortPoints(vector<Point2f> &points){
    if(points.size()!=4) return;
    // 计算中心点
    Point2f center(0,0);
    for(const auto &p:points){// p是对points容器中每个元素(x,y)的常量引用
      center+=p;
    }
    center/=4;
    // 定义排序后的数组
    vector<Point2f> sorted(4);
    // 对四个点进行排序
    for(const auto &p:points){
        if(p.x<center.x && p.y<center.y)
          sorted[0]=p;// 左上
        else if(p.x>center.x && p.y<center.y)
          sorted[1]=p;// 右上
        else if(p.x>center.x && p.y>center.y)
          sorted[2]=p;// 右下
        else
          sorted[3]=p;// 左下
    }
    points = sorted;
}

// 装甲板识别函数
void ArmorDetect(Mat &frame){
    // hsv：用于存储将原图转换到 HSV 颜色空间 后的图像。HSV 空间更利于基于颜色的分割
    Mat hsv,red_mask1,red_mask2,red_mask,binary,Gaussian,dilatee;
    Mat element=getStructuringElement(MORPH_RECT,Size(5,5));
    vector<vector<Point>>contours;
    vector<Vec4i> hierarchy;
    // 转换为HSV颜色空间，更好地识别红色
    cvtColor(frame,hsv,COLOR_BGR2HSV);
    // 定义红色在HSV空间中的范围（红色有两个范围，因为红色在HSV色环的两端）
    Scalar lower_red1(0,100,100);
    Scalar upper_red1(10,255,255);
    Scalar lower_red2(160,100,100);
    Scalar upper_red2(179,255,255);
    // 创建红色掩码
    inRange(hsv,lower_red1,upper_red1,red_mask1);
    inRange(hsv,lower_red2,upper_red2,red_mask2);
    // 合并两个红色掩码
    red_mask=red_mask1|red_mask2;
    // 去除噪声并连接红色区域
    GaussianBlur(red_mask,Gaussian,Size(5,5),0);
    dilate(Gaussian,dilatee,element);
    findContours(dilatee,contours,hierarchy,RETR_TREE,CHAIN_APPROX_NONE);
    vector<LightDescription> lightInfos;
    // 筛选灯条
    for(size_t i=0;i<contours.size();i++){
      double area = contourArea(contours[i]);// 计算当前轮廓的面积（像素单位）
      if(area<5 || contours[i].size() <= 1) // 条件判断：面积小于 5：过滤掉太小的噪声区域；轮廓点数 ≤ 1：无法拟合椭圆（至少需要 5 个点），跳过
        continue;
      RotatedRect Light_Rec=fitEllipse(contours[i]);// fitEllipse：对轮廓点进行椭圆拟合，返回旋转矩形（包含了灯条的中心、大小（宽度和高度）以及旋转角度）
      if(Light_Rec.size.width/Light_Rec.size.height>4)// 旋转矩形的宽度与高度之比
        continue;
      lightInfos.push_back(LightDescription(Light_Rec));// 将结果添加到 lightInfos 容器中
    }
    // 二重循环多条件匹配灯条
    for(size_t i=0;i<lightInfos.size();i++){
      for(size_t j=i+1;j<lightInfos.size();j++){
        LightDescription &leftLight = lightInfos[i];
        LightDescription &rightLight = lightInfos[j];
        float angleGap_=abs(leftLight.angle-rightLight.angle);// 角度差
        float LenGap_ratio=abs(leftLight.length-rightLight.length)/max(leftLight.length,rightLight.length);//长度差与最大长度比
        float dis=sqrt(pow((leftLight.center.x-rightLight.center.x),2)+pow((leftLight.center.y-rightLight.center.y),2));// 中心距离
        float meanLen=(leftLight.length+rightLight.length)/2;// 平均长度
        float lengap_ratio=abs(leftLight.length-rightLight.length)/meanLen;// 长度差与平均长度比
        float yGap=abs(leftLight.center.y-rightLight.center.y);// 垂直差距
        float yGap_ratio=yGap/meanLen;
        float xGap=abs(leftLight.center.x-rightLight.center.x);// 水平差距 
        float xGap_ratio=xGap/meanLen;
        float ratio=dis/meanLen;
        // 匹配条件
        if(angleGap_>15||LenGap_ratio>0.3||lengap_ratio>0.8||yGap_ratio>1.5||xGap_ratio>2.2||xGap_ratio<0.8||ratio>3||ratio<0.8){
          continue;
        }
        // 计算装甲板角点
        Point center = Point((leftLight.center.x+rightLight.center.x)/2,(leftLight.center.y+rightLight.center.y)/2);
        RotatedRect rect = RotatedRect(center,Size(dis,meanLen),(leftLight.angle+rightLight.angle)/2);
        Point2f vertices[4];
        rect.points(vertices);
        // 将点转换为向量并排序
        vector<Point2f> points(vertices,vertices+4);
        sortPoints(points);
        // 绘制装甲板边界
        for(int k=0;k<4;k++){
          line(frame,points[k],points[(k+1)%4],Scalar(0,0,255),2);
        }
        // 可视化角点并标注序号
        for(int k=0;k<4;k++){
          // 绘制角点
          circle(frame,points[k],6,Scalar(0.255,0),-1);
          // 标注角点序号
          string point_num=to_string(k+1);
          putText(frame,point_num,points[k]+Point2f(-5,5),FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 2);
        }
        // 打印角点坐标
        cout<<"装甲板坐标（左上、右上、右下、左下）："<<endl;
        for(int k=0;k<4;k++){
          cout<<points[k]<<endl;
        }
    }
}
}
