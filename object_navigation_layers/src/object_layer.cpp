#include <object_layers/object_layer.h>
#include <pluginlib/class_list_macros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <jsk_recognition_utils/pcl_conversion_util.h>
#include <sensor_msgs/point_cloud2_iterator.h>

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

  cloud_sub_.subscribe(nh, "/multisense_local/image_points2_color", 1);
  label_sub_.subscribe(nh, "/object_label_image", 1);

  async_ = boost::make_shared<message_filters::Synchronizer<ApproximateSyncPolicy> >(3);
  async_->connectInput(cloud_sub_, label_sub_);
  async_->registerCallback(
    boost::bind(&ObjectLayer::labelCloudCallback, this, _1, _2));

  ROS_WARN("Object Layer Initialized!!!");
  global_frame_ = layered_costmap_->getGlobalFrameID();

  pointcloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/debug_pointcloud", 1);
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

void ObjectLayer::labelCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg, const sensor_msgs::Image::ConstPtr& label_msg)
{
  ROS_WARN("Sync Label Cloud Callback");
  cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*cloud_msg, *cloud_);
  try {
    listener_.waitForTransform(cloud_msg->header.frame_id, global_frame_,
                               cloud_msg->header.stamp, ros::Duration(1.0));
    listener_.lookupTransform(cloud_msg->header.frame_id, global_frame_,
                              cloud_msg->header.stamp, transform_);
  }
  catch (tf2::LookupException &e)
  {
    ROS_ERROR("transform error: %s", e.what());
  }
  catch(tf::ConnectivityException& e) {
    ROS_ERROR("Connectivity Error: %s\n", e.what());
  }
  catch(tf::ExtrapolationException& e) {
    ROS_ERROR("Extrapolation Error: %s\n", e.what());
  }

  cv_bridge::CvImagePtr cv_ptr;
  cv_ptr  = cv_bridge::toCvCopy(label_msg, sensor_msgs::image_encodings::MONO8);
  label_  = cv_ptr->image;
  height_ = label_msg->height;
  width_  = label_msg->width;
  header_ = label_msg->header;
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
  cv::Mat label = label_;
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

    // ====== for debug =================================================
    sensor_msgs::PointCloud2::Ptr cloud_msg(new sensor_msgs::PointCloud2);

    cloud_msg->header = header_;
    cloud_msg->header.frame_id = global_frame_;
    // cloud_msg->header.frame_id = "left_camera_optical_frame";
    cloud_msg->height = height_;
    cloud_msg->width  = width_;
    cloud_msg->is_dense = false;
    cloud_msg->is_bigendian = false;

    sensor_msgs::PointCloud2Modifier pcd_modifier(*cloud_msg);
    pcd_modifier.setPointCloud2FieldsByString(1, "xyz");

    sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_msg, "z");
    // ==================================================================

    int u, v;
    for (int i = 0; i < cloud.size(); i++) {
      u = i % width_;
      v = i / width_;
      if (label.at<unsigned char>(v, u) == 0) continue;

      pp = cloud.points[i];

      original_coords.setValue(pp.x, pp.y, pp.z);
      target_coords = rotation_matrix.inverse() * (original_coords - translation_vector);

      mark_x = target_coords.x();
      mark_y = target_coords.y();

      if(costmap->worldToMap(mark_x, mark_y, mx, my)) {
        index = my * size_x_ + mx;
        costmap_[index] = std::min(label.at<unsigned char>(v, u), LETHAL_OBSTACLE);
      }

      *iter_x = target_coords.x();
      *iter_y = target_coords.y();
      *iter_z = target_coords.z();
      ++iter_x;
      ++iter_y;
      ++iter_z;

      *min_x = std::min(*min_x, mark_x);
      *min_y = std::min(*min_y, mark_y);
      *max_x = std::max(*max_x, mark_x);
      *max_y = std::max(*max_y, mark_y);
    }
    pointcloud_pub_.publish(cloud_msg);
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
