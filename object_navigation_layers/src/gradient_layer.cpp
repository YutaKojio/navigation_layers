#include <object_layers/gradient_layer.h>
#include <pluginlib/class_list_macros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <jsk_recognition_utils/pcl_conversion_util.h>
#include <geometry_msgs/PointStamped.h>

PLUGINLIB_EXPORT_CLASS(object_navigation_layers::GradientLayer, costmap_2d::Layer)

using costmap_2d::LETHAL_OBSTACLE;
using costmap_2d::NO_INFORMATION;
using costmap_2d::FREE_SPACE;

namespace object_navigation_layers
{

GradientLayer::GradientLayer() {}

void GradientLayer::onInitialize()
{
  ros::NodeHandle nh("~/" + name_);
  current_ = true;
  default_value_ = NO_INFORMATION;
  matchSize();

  dsrv_ = new dynamic_reconfigure::Server<costmap_2d::GenericPluginConfig>(nh);
  dynamic_reconfigure::Server<costmap_2d::GenericPluginConfig>::CallbackType cb = boost::bind(
      &GradientLayer::reconfigureCB, this, _1, _2);
  dsrv_->setCallback(cb);

  image_sub_ = nh.subscribe("/heightmap_gradient", 1, &GradientLayer::heightmapGradientCallback, this);
  config_sub_ = nh.subscribe("/accumulated_heightmap/output/config", 1, &GradientLayer::configCallback, this);

  global_frame_ = layered_costmap_->getGlobalFrameID();
  double transform_tolerance;
  nh.param("transform_tolerance", transform_tolerance, 0.2);

}


void GradientLayer::matchSize()
{
  Costmap2D* master = layered_costmap_->getCostmap();
  resizeMap(master->getSizeInCellsX(), master->getSizeInCellsY(), master->getResolution(),
            master->getOriginX(), master->getOriginY());
}


void GradientLayer::reconfigureCB(costmap_2d::GenericPluginConfig &config, uint32_t level)
{
  enabled_ = config.enabled;
}

void GradientLayer::configCallback(const jsk_recognition_msgs::HeightmapConfig::ConstPtr& msg)
{
  config_msg_ = msg;
  min_x_ = msg->min_x;
  max_x_ = msg->max_x;
  min_y_ = msg->min_y;
  max_y_ = msg->max_y;
}

void GradientLayer::heightmapGradientCallback(const sensor_msgs::Image::ConstPtr& msg)
{
  heightmap_gradient_msg_ = msg;
  cv_bridge::CvImagePtr cv_ptr;
  cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
  heightmap_gradient_ = cv_ptr->image;
  height_ = msg->height;
  width_ = msg->width;
  flag_new_ = true;

}

void GradientLayer::updateBounds(double robot_x, double robot_y, double robot_yaw, double* min_x,
                                 double* min_y, double* max_x, double* max_y)
{
  if (!enabled_)
    return;

  if (!config_msg_)
    return;

  if(!heightmap_gradient_msg_)
    return;

  if(!flag_new_)
    return;

  if(!layered_costmap_)
    return;

  double mark_x, mark_y;
  double heightmap_resolution_x;
  double heightmap_resolution_y;
  heightmap_resolution_x =  1.0 * (max_x_ - min_x_) / width_;
  heightmap_resolution_y =  1.0 * (max_y_ - min_y_) / height_;
  geometry_msgs::PointStamped pt, opt;

  try {
    for (int j = 0; j < height_; ++j) {
      for (int i = 0; i < width_; ++i) {
        pt.point.x = min_x_ + i * heightmap_resolution_x;
        pt.point.y = min_y_ + j * heightmap_resolution_y;
        pt.point.z = 0;
        pt.header.frame_id = heightmap_gradient_msg_->header.frame_id;
        tf_.transformPoint(global_frame_, pt, opt);
        mark_x = opt.point.x;
        mark_y = opt.point.y;

        *min_x = std::min(*min_x, mark_x);
        *min_y = std::min(*min_y, mark_y);
        *max_x = std::max(*max_x, mark_x);
        *max_y = std::max(*max_y, mark_y);
      }
    }
  }
  catch(tf::LookupException& ex) {
    ROS_ERROR("No Transform available Error: %s\n", ex.what());
  }
  catch(tf::ConnectivityException& ex) {
    ROS_ERROR("Connectivity Error: %s\n", ex.what());
  }
  catch(tf::ExtrapolationException& ex) {
    ROS_ERROR("Extrapolation Error: %s\n", ex.what());
  }
}

void GradientLayer::updateCosts(costmap_2d::Costmap2D& master_grid, int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_)
    return;

  if (!config_msg_)
    return;

  if(!heightmap_gradient_msg_)
    return;

  if(!flag_new_)
    return;

  if(!layered_costmap_)
    return;

  // setCost is for set "current" cost, not for "accumulated" cost
  // To make accumulated cost map, use grid map ex) CostmapLayer::updateWithMax
  double mark_x, mark_y;
  double heightmap_resolution_x;
  double heightmap_resolution_y;
  heightmap_resolution_x =  1.0 * (max_x_ - min_x_) / width_;
  heightmap_resolution_y =  1.0 * (max_y_ - min_y_) / height_;
  geometry_msgs::PointStamped pt, opt;
  double scale = 20;
  double offset = 1;
  double raw_cost;
  unsigned char cost;
  unsigned int mx, my;

  try {
    for (int j = 0; j < height_; ++j) {
      for (int i = 0; i < width_; ++i) {
        pt.point.x = min_x_ + i * heightmap_resolution_x;
        pt.point.y = min_y_ + j * heightmap_resolution_y;
        pt.point.z = 0;
        pt.header.frame_id = heightmap_gradient_msg_->header.frame_id;
        tf_.transformPoint(global_frame_, pt, opt);
        mark_x = opt.point.x;
        mark_y = opt.point.y;
        raw_cost = heightmap_gradient_.at<float>(j, i) * scale - offset;
        cost = std::max(std::min((unsigned char)raw_cost, LETHAL_OBSTACLE), FREE_SPACE);
        if(master_grid.worldToMap(mark_x, mark_y, mx, my)) {
          master_grid.setCost(mx, my, cost);
        }
      }
    }
  }
  catch(tf::LookupException& ex) {
    ROS_ERROR("No Transform available Error: %s\n", ex.what());
  }
  catch(tf::ConnectivityException& ex) {
    ROS_ERROR("Connectivity Error: %s\n", ex.what());
  }
  catch(tf::ExtrapolationException& ex) {
    ROS_ERROR("Extrapolation Error: %s\n", ex.what());
  }

  flag_new_ = false;

  ROS_WARN("Heightmap Cost Updated");
}

} // end namespace
