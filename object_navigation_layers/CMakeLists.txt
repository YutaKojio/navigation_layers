cmake_minimum_required(VERSION 2.8.3)
project(object_navigation_layers)

find_package(catkin REQUIRED COMPONENTS
  angles
  costmap_2d
  dynamic_reconfigure
  pluginlib
  roscpp
  sensor_msgs
  geometry_msgs
  jsk_recognition_msgs
  cv_bridge
)

# generate_dynamic_reconfigure_options(
#   cfg/ProxemicLayer.cfg
# )

catkin_package(
    CATKIN_DEPENDS angles costmap_2d dynamic_reconfigure pluginlib roscpp
    INCLUDE_DIRS include
    LIBRARIES object_layers
)

include_directories(
  include ${catkin_INCLUDE_DIRS}
)

## add cpp library
add_library(object_layers
        src/simple_layer.cpp
        src/grid_layer.cpp
        src/object_layer.cpp
        src/gradient_layer.cpp
)
# add_dependencies(object_layers ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(
    object_layers ${catkin_LIBRARIES}
)

install(FILES costmap_plugins.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(TARGETS object_layers
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(
    DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
