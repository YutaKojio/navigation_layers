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
  listener_.lookupTransform(heightmap_gradient_msg_->header.frame_id, global_frame_,
                            ros::Time(0), transform_);
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

  if(!layered_costmap_)
    return;

  double mark_x, mark_y;
  double heightmap_resolution_x;
  double heightmap_resolution_y;
  heightmap_resolution_x =  1.0 * (max_x_ - min_x_) / width_;
  heightmap_resolution_y =  1.0 * (max_y_ - min_y_) / height_;
  double scale = 20;
  double offset = 1;
  double raw_cost;
  unsigned char cost;
  unsigned int mx, my;
  unsigned int index;

  costmap_2d::Costmap2D* costmap = layered_costmap_->getCostmap();
  cv::Mat heightmap_gradient = heightmap_gradient_;

  clock_t start = clock();

  tf::Vector3 original_coords;
  tf::Vector3 translation_vector;
  tf::Matrix3x3 rotation_matrix;
  tf::Vector3 target_coords;

  try {
    translation_vector = transform_.getOrigin();
    rotation_matrix = transform_.getBasis();

    for (int j = 0; j < height_; j+=3) {
      for (int i = 0; i < width_; i+=3) {
        original_coords.setValue(min_x_ + i * heightmap_resolution_x,
                                 min_y_ + j * heightmap_resolution_y,
                                 0.0);
        target_coords = rotation_matrix.inverse() * (original_coords - translation_vector);
        mark_x = target_coords.x();
        mark_y = target_coords.y();
        raw_cost = heightmap_gradient.at<float>(j, i) * scale - offset;
        cost = std::max(std::min((unsigned char)raw_cost, LETHAL_OBSTACLE), FREE_SPACE);

        if(costmap->worldToMap(mark_x, mark_y, mx, my)) {
          index = my * size_x_ + mx;
          // unsigned char old_cost = costmap_->getCost(mx, my);
          unsigned char old_cost = costmap_[index];
          if(old_cost == costmap_2d::NO_INFORMATION) {
            // costmap_->SetCost(mx, my, cost);
            costmap_[index] = cost;
          }
          else {
            // costmap_->SetCost(mx, my, std::max(cost, old_cost));
            // costmap_[index] = std::max(cost, old_cost);
            costmap_[index] = cost;
          }
        }

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
  clock_t end = clock();
  const double time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
  std::cerr << "updateBounds : " << time << " [ms]" << std::endl;
}

void GradientLayer::updateCosts(costmap_2d::Costmap2D& master_grid, int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_)
    return;

  if (!config_msg_)
    return;

  if(!heightmap_gradient_msg_)
    return;

  if(!costmap_)
    return;

  if(!layered_costmap_)
    return;

  clock_t start = clock();

  // update cost of master layer (with max)
  unsigned char* master_array = master_grid.getCharMap();
  unsigned int span = master_grid.getSizeInCellsX();

  for (int j = min_j; j < max_j; j++)
  {
    unsigned int it = j * span + min_i;
    for (int i = min_i; i < max_i; i++)
    {
      if (costmap_[it] == NO_INFORMATION){
        it++;
        continue;
      }

      unsigned char old_cost = master_array[it];
      if (old_cost == NO_INFORMATION || old_cost < costmap_[it])
        master_array[it] = costmap_[it];
      it++;
    }
  }

  ROS_WARN("Heightmap Cost Updated");
  clock_t end = clock();
  const double time = static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000.0;
  std::cerr << "updateCosts : " << time << " [ms]" << std::endl;

}

} // end namespace
