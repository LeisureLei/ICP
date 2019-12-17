#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <fstream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/registration/icp.h>  //pcl icp配准
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>
//#include <pcl/io/ply_io.h>
//#include <pcl/io/vtk_lib_io.h>
#include <cmath>
#include <vector>
#include "../include/addtimestamp/kdtree.h"

using namespace std;

pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr > twoPC; //上一帧点云和当前点云

vector<pair<Eigen::Matrix3f,Eigen::Vector3f> > posesets;

int numPointCloud = 0;

bool first = true;


/*
void estimateMotion()
{
	pair<Eigen::Matrix3d, Eigen::Vector3d> TransRT(Eigen::Matrix3d::Identity(),Eigen::Vector3d::Zero()); //初始化R,t
	double dt = 0;;
	vector<Eigen::Vector3d> nearestPointCloud;
	size_t lastpointsize = twoPC.first.size(); //上一帧点云数目
	size_t currentpointsize = twoPC.second.size(); //当前帧点云数目
	/////////////////////////////////////构建kdtree///////////////////////////////////////////////////////
	float datas[currentpointsize*3],labels[currentpointsize];
	size_t args[5];
	float dists[lastpointsize];
	for(size_t i=0;i<currentpointsize;i++){
		datas[i*3] = twoPC.second[i].x;
		datas[i*3+1] = twoPC.second[i].y;
		datas[i*3+2] =  twoPC.second[i].z;
		labels[i] = (float)i;
	}
	tree_model *model = build_kdtree(datas, labels, currentpointsize,3,2); //
	//////////////////////////////////////构建kdtree///////////////////////////////////////////////////
	

	while(dt<5)  //迭代5次
	{	dt++;
		nearestPointCloud.clear();


		for(size_t i=0;i<lastpointsize;i++)
		{	
			Eigen::Vector3d lastpoint( twoPC.first[i].x, twoPC.first[i].y, twoPC.first[i].z );   //上一帧点
			Eigen::Vector3d transformedT(TransRT.first*lastpoint + TransRT.second);    //变换后点
			//在当前点云中找最近邻
			float test[3] = { transformedT[0],transformedT[1],transformedT[2] };

			find_k_nearests(model, test, 1, args, dists); 
			nearestPointCloud.push_back(Eigen::Vector3d(twoPC.second[args[0]].x,twoPC.second[args[0]].y,twoPC.second[args[0]].z));
			//cout<<"ID:"<<args[0]<<" "<<"dis:"<<dists[0]<<endl;
		}
		
		cout<<"last pointcloud size:"<<lastpointsize<<endl;
		cout<<"current pointcloud size:"<<nearestPointCloud.size()<<endl;

		//估计运动
		Eigen::Vector3d p1 = Eigen::Vector3d::Zero();
		Eigen::Vector3d p2 = Eigen::Vector3d::Zero();

		for(size_t j=0;j<lastpointsize;j++)
		{
			p1 += Eigen::Vector3d(twoPC.first[j].x, twoPC.first[j].y, twoPC.first[j].z);
			p2 += Eigen::Vector3d(nearestPointCloud[j][0], nearestPointCloud[j][1], nearestPointCloud[j][2]);
		}
		p1 /= lastpointsize;
		p2 /= lastpointsize;
		vector<Eigen::Vector3d> X(lastpointsize), Y(lastpointsize);
		for(size_t k=0;k<lastpointsize;k++)
		{
			X[k] = Eigen::Vector3d(twoPC.first[k].x, twoPC.first[k].y, twoPC.first[k].z) - p1;
			Y[k] = Eigen::Vector3d(nearestPointCloud[k][0], nearestPointCloud[k][1], nearestPointCloud[k][2]) - p2;
		}
		Eigen::Matrix3d S = Eigen::Matrix3d::Zero();
		for(size_t l=0;l<lastpointsize;l++){
			S += X[l] * Y[l].transpose();
		}
		Eigen::JacobiSVD<Eigen::Matrix3d> svd ( S, Eigen::ComputeFullU|Eigen::ComputeFullV );
		const Eigen::Matrix3d U = svd.matrixU();
    	const Eigen::Matrix3d V = svd.matrixV();
		Eigen::Matrix3d mirror = Eigen::Matrix3d::Identity();
    	mirror(2,2) = (V*U.transpose()).determinant();
   		Eigen::Matrix3d R = V*mirror*U.transpose();
    	Eigen::Vector3d t = p2 - R*p1;
		//cout<<"R=:"<<R<<endl<<"t=:"<<t<<endl;
		//更新R,t
		TransRT.first = R;
		TransRT.second = t;

		double error = 0;
		for(size_t i=0;i<lastpointsize;i++)
		{
			Eigen::Vector3d solvedPoint(R*Eigen::Vector3d(twoPC.first[i].x, twoPC.first[i].y, twoPC.first[i].z)+t);
			Eigen::Vector3d diff(solvedPoint-Eigen::Vector3d(nearestPointCloud[i][0],nearestPointCloud[i][1],nearestPointCloud[i][2]));
			error += sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]);
		}
		cout<<"erorr:"<<error<<endl<<endl;
	}
}
*/

void icpestimate(const string& timestamp)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_trans(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setInputSource(twoPC.first);
	icp.setInputTarget(twoPC.second);

	icp.setMaxCorrespondenceDistance(0.05);	//对应点间的最大距离(单位为m)
	icp.setMaximumIterations(50);           //最大迭代次数
	icp.setTransformationEpsilon(1e-6);     //两次变化矩阵之间的差值
	icp.setEuclideanFitnessEpsilon(0.001);  //设置收敛条件是均方误差和小于阈值， 停止迭代
	icp.align(*cloud_source_trans);         //变换后的源点云
	if (icp.hasConverged())
	{
		numPointCloud++;
		cout<<"count:"<<numPointCloud<<endl;
		Eigen::Matrix4f T = icp.getFinalTransformation();
		//cout << "Converged. score =" << icp.getFitnessScore() << endl;
		cout << T << endl<<endl;

		Eigen::Matrix3d R;
		R << (double)T(0,0),(double)T(0,1),(double)T(0,2),
				(double)T(1,0),(double)T(1,1),(double)T(1,2),
					(double)T(2,0),(double)T(2,1),(double)T(2,2);
		Eigen::Vector3d t(T(0,3),T(1,3),T(2,3));
		Eigen::Quaterniond q(R);
		q.normalized();
		ofstream file("ousterPose.txt",ios::app);
		cout<<timestamp<<endl;
		file <<timestamp<<setprecision(8)<<" "<<t[0]<<" "<<t[1]<<" "<<t[2]<<" "<<q.w()<<" "<<q.x()<<" "<<q.y()<<" "<<q.z()<<endl;
		file.close();
	}
	/*
	//点云可视化
	boost::shared_ptr< pcl::visualization::PCLVisualizer > viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0); //背景色可设置
	//显示源点云
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(twoPC.first, 0, 255, 0);
	viewer->addPointCloud<pcl::PointXYZ>(twoPC.second, source_color, "source");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "source");
 
	//显示目标点云
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(twoPC.second, 255, 0, 255);
	viewer->addPointCloud<pcl::PointXYZ>(twoPC.second, target_color, "target");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target");
 
	//显示变换后的源点云
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_trans_color(cloud_source_trans, 255, 255, 255);
	viewer->addPointCloud<pcl::PointXYZ>(cloud_source_trans, source_trans_color, "source trans");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "source trans");
    
	viewer->spin();
	//保存变换结果
	pcl::io::savePLYFile("ICP_test_trans.ply", *cloud_source_trans, false);
	*/

}

class addtimePub
{
public:

	void gpsCb(const nav_msgs::Odometry& gpsMsg)
	{
		
		nav_msgs::Odometry msg;
		msg.header.stamp = gpsMsg.header.stamp.toSec();
		msg.pose.pose.position.x = gpsMsg.pose.pose.position.x ;
		msg.pose.pose.position.y = gpsMsg.pose.pose.position.y;
		msg.pose.pose.position.z= gpsMsg.pose.pose.position.z;
		msg.pose.pose.orientation.x = gpsMsg.pose.pose.orientation.x; 
		msg.pose.pose.orientation.y = gpsMsg.pose.pose.orientation.y;
		msg.pose.pose.orientation.z = gpsMsg.pose.pose.orientation.z;
		msg.pose.pose.orientation.w = gpsMsg.pose.pose.orientation.w;
		ROS_INFO("gps time:%f",gpsMsg.header.stamp.toSec());

		ofstream gpsfile("gpspose.txt",ios::app);

		gpsfile<<to_string(gpsMsg.header.stamp.toSec())<<" "<<gpsMsg.pose.pose.position.x<<" "
															<<gpsMsg.pose.pose.position.y<<" "
															<<gpsMsg.pose.pose.position.z<<" "
															<<gpsMsg.pose.pose.orientation.w<<" "
															<<gpsMsg.pose.pose.orientation.x<<" "
															<<gpsMsg.pose.pose.orientation.y<<" "
															<<gpsMsg.pose.pose.orientation.z<<endl;
		
		//gpsPub_.publish(msg);
	}

	void ousterCb(const sensor_msgs::PointCloud2& ousterMsg)
	{
		ros::Rate loop_rate(10);
		pcl::PointCloud<pcl::PointXYZ>::Ptr pclPointCloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::fromROSMsg(ousterMsg, *pclPointCloud);  //把点云转换成pcl格式
		std::vector<int> index;
		pcl::removeNaNFromPointCloud(*pclPointCloud,*pclPointCloud,index);  //去除无效点
		//cout<<"before filtered pointcloud size:"<<pclPointCloud->size()<<endl;
		//点云降采样
		pcl::VoxelGrid<pcl::PointXYZ> downSampled;   //创建滤波对象
		downSampled.setInputCloud(pclPointCloud);   //设置需要的过滤点云给滤波对象,参数要是指针类型
		downSampled.setLeafSize(0.2f,0.2f,0.2f);     //设置滤波时创建的体素体积为20cm的立方体
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downSampled(new pcl::PointCloud<pcl::PointXYZ>);
		downSampled.filter(*cloud_downSampled);    //执行滤波处理，存储输出,参数是变量
		
		//ROS_INFO ("pointcloud size:%d",cloud_downSampled.size());

		//去除outlier
		pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;  //创建滤波器对象
		sor.setInputCloud(cloud_downSampled);   //设置待滤波的点云
		sor.setMeanK(80);   //设置在进行统计时考虑的临近点个数
		sor.setStddevMulThresh(1.0);   //设置判断是否为离群点的阈值，用来倍乘标准差
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
		sor.filter(*cloud_filtered);  //滤波结果存储到cloud_filtered

		if(first){
			twoPC.first = cloud_filtered;  
			twoPC.second = cloud_filtered;
			first = false;
			ROS_INFO("ouster time: %f",ousterMsg.header.stamp.toSec());
			ofstream file("ousterPose.txt",ios::app);
			
			file <<to_string(ousterMsg.header.stamp.toSec())<<" "<<0<<" "<<0<<" "<<0<<" "<<0<<" "<<0<<" "<<0<<" "<<0<<endl;
			file.close();

			return;
		}
		else{
			twoPC.first = twoPC.second;
			twoPC.second = cloud_filtered;  //
		}

		//cout<<"pointcloud size:"<<twoPC.first.size()<<endl;

		//发布topic
		sensor_msgs::PointCloud2 outputMsg;
		pcl::toROSMsg(*twoPC.first, outputMsg);  //将滤波后点云转为ros消息
		outputMsg.header.stamp = ousterMsg.header.stamp;  //添加原始时间戳
		ROS_INFO("ouster time: %f",ousterMsg.header.stamp.toSec());
		ousterPub_.publish(outputMsg);
		string timestamp = to_string(ousterMsg.header.stamp.toSec());
		//estimateMotion();
		//icpestimate(timestamp);
		//loop_rate.sleep();
	}

	addtimePub()
	{
		gpsSub_ = nh_.subscribe("/gps/odom",1000,&addtimePub::gpsCb,this);
		ousterSub_ = nh_.subscribe("/ouster_front/os1_cloud_node/points",1000,&addtimePub::ousterCb,this);
		
		//gpsPub_ = nh_.advertise<nav_msgs::Odometry>("/gps_time_odom",1000);
		ousterPub_ = nh_.advertise<sensor_msgs::PointCloud2>("/ouster_time_odom",1000);
		
	}

private:
	ros::NodeHandle nh_;
	ros::Subscriber gpsSub_;
	ros::Subscriber ousterSub_;
	ros::Publisher gpsPub_;
	ros::Publisher ousterPub_;

};

int main(int argc, char** argv)
{
	ros::init(argc, argv, "addtimestamp");

	
	addtimePub sp;
	
	ros::spin();
	
	return 0;
	
}
