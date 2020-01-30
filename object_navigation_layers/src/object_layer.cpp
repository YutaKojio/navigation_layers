#include <object_layers/object_layer.h>
#include <pluginlib/class_list_macros.h>

PLUGINLIB_EXPORT_CLASS(object_navigation_layers::ObjectLayer, costmap_2d::Layer)

using costmap_2d::LETHAL_OBSTACLE;
using costmap_2d::NO_INFORMATION;

namespace object_navigation_layers
{

ObjectLayer::ObjectLayer() {}

void ObjectLayer::onInitialize()
{
  ros::NodeHandle nh("~/" + name_);
  current_ = true;
  default_value_ = NO_INFORMATION;
  matchSize();

  dsrv_ = new dynamic_reconfigure::Server<object_navigation_layers::ObjectLayerConfig>(nh);
  dynamic_reconfigure::Server<object_navigation_layers::ObjectLayerConfig>::CallbackType cb = boost::bind(
      &ObjectLayer::reconfigureCB, this, _1, _2);
  dsrv_->setCallback(cb);

  cloud_sub_ = nh.subscribe("/objects_pointcloud/output", 1, &ObjectLayer::pointCloudCallback, this);
  global_frame_ = layered_costmap_->getGlobalFrameID();
}


void ObjectLayer::matchSize()
{
  Costmap2D* master = layered_costmap_->getCostmap();
  resizeMap(master->getSizeInCellsX(), master->getSizeInCellsY(), master->getResolution(),
            master->getOriginX(), master->getOriginY());
}


void ObjectLayer::reconfigureCB(object_navigation_layers::ObjectLayerConfig &config, uint32_t level)
{
  enabled_ = config.enabled;
  combination_method_ = config.combination_method;
}

void ObjectLayer::pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*msg, *cloud_);
  try {
    listener_.lookupTransform(msg->header.frame_id, global_frame_,
                              msg->header.stamp, transform_);
  }
  catch (tf2::LookupException &e)
  {
    ROS_ERROR("transform error: %s", e.what());
  }
}

void ObjectLayer::updateBounds(double robot_x, double robot_y, double robot_yaw, double* min_x,
                                           double* min_y, double* max_x, double* max_y)
{
  if (!enabled_)
    return;

  if (!cloud_)
    return;

  if (layered_costmap_->isRolling())
    updateOrigin(robot_x - getSizeInMetersX() / 2, robot_y - getSizeInMetersY() / 2);

  pcl::PointCloud<pcl::PointXYZ> cloud = *cloud_;
  pcl::PointXYZ pp;
  double mark_x, mark_y;
  unsigned int mx;
  unsigned int my;
  unsigned int index;

  costmap_2d::Costmap2D* costmap = layered_costmap_->getCostmap();

  tf::Vector3 original_coords;
  tf::Vector3 translation_vector;
  tf::Matrix3x3 rotation_matrix;
  tf::Vector3 target_coords;

  try {
    translation_vector = transform_.getOrigin();
    rotation_matrix = transform_.getBasis();

    // Overwrite
    if (combination_method_ == 0 && cloud.size() > 0) {
      costmap_ = new unsigned char[size_x_ * size_y_];
      for (int i = 0; i < size_x_ * size_y_; ++i) {
        costmap_[i] = NO_INFORMATION;
      }
    }

    for (int i = 0; i < cloud.size(); i++) {
      pp = cloud.points[i];

      original_coords.setValue(pp.x, pp.y, 0.0);
      target_coords = rotation_matrix.inverse() * (original_coords - translation_vector);

      mark_x = target_coords.x();
      mark_y = target_coords.y();

      if(costmap->worldToMap(mark_x, mark_y, mx, my)) {
        index = my * size_x_ + mx;
        costmap_[index] = LETHAL_OBSTACLE;
      }

      *min_x = std::min(*min_x, mark_x);
      *min_y = std::min(*min_y, mark_y);
      *max_x = std::max(*max_x, mark_x);
      *max_y = std::max(*max_y, mark_y);
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

void ObjectLayer::updateCosts(costmap_2d::Costmap2D& master_grid, int min_i, int min_j, int max_i,
                                          int max_j)
{
  if (!enabled_)
    return;

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

}

} // end namespace
