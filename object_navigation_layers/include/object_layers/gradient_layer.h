#ifndef GRADIENT_LAYER_H_
#define GRADIENT_LAYER_H_
#include <ros/ros.h>
#include <costmap_2d/layer.h>
#include <costmap_2d/layered_costmap.h>
#include <costmap_2d/GenericPluginConfig.h>
#include <dynamic_reconfigure/server.h>
#include <jsk_recognition_utils/pcl_conversion_util.h>
#include <jsk_recognition_msgs/HeightmapConfig.h>

namespace object_navigation_layers
{

class GradientLayer : public costmap_2d::Layer, public costmap_2d::Costmap2D
{
public:
  GradientLayer();

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
  void configCallback(const jsk_recognition_msgs::HeightmapConfig::ConstPtr& msg);
  void heightmapGradientCallback(const sensor_msgs::Image::ConstPtr& msg);
  ros::Subscriber image_sub_;
  ros::Subscriber config_sub_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
  tf::TransformListener tf_;
  sensor_msgs::Image::ConstPtr heightmap_gradient_msg_;
  cv::Mat heightmap_gradient_;
  jsk_recognition_msgs::HeightmapConfig::ConstPtr config_msg_;
  int height_;
  int width_;
  double min_x_;
  double max_x_;
  double min_y_;
  double max_y_;
  bool flag_new_;
};
}
#endif
