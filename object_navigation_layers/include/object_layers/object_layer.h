#ifndef OBJECT_LAYER_H_
#define OBJECT_LAYER_H_
#include <ros/ros.h>
#include <costmap_2d/layer.h>
#include <costmap_2d/layered_costmap.h>
#include <costmap_2d/GenericPluginConfig.h>
#include <dynamic_reconfigure/server.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/PointIndices.h>
#include <jsk_recognition_utils/pcl_conversion_util.h>

namespace object_navigation_layers
{

class ObjectLayer : public costmap_2d::Layer, public costmap_2d::Costmap2D
{
public:
  ObjectLayer();

  virtual void onInitialize();
  virtual void updateBounds(double robot_x, double robot_y, double robot_yaw, double* min_x, double* min_y, double* max_x,
                             double* max_y);
  virtual void updateCosts(costmap_2d::Costmap2D& master_grid, int min_i, int min_j, int max_i, int max_j);
  bool isDiscretized()
  {
    return true;
  }

  virtual void matchSize();

private:
  void reconfigureCB(costmap_2d::GenericPluginConfig &config, uint32_t level);
  dynamic_reconfigure::Server<costmap_2d::GenericPluginConfig> *dsrv_;
  void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);
  ros::Subscriber cloud_sub_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
  tf::TransformListener tf_;
};
}
#endif
