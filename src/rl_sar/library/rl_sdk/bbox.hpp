#ifndef BBOX_HPP_
#define BBOX_HPP_

#include <ros/ros.h>
#include <urdf/model.h>
#include <yaml-cpp/yaml.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <Eigen/Dense>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <filesystem>
#include <fstream>

class BoundingBoxManager {
public:
    struct BoxConfig {
        std::string link_name;
        std::vector<double> size;     // x, y, z dimensions
        std::vector<double> offset;   // x, y, z offset from link origin
    };

    BoundingBoxManager(ros::NodeHandle& nh);
    ~BoundingBoxManager() = default;

    // Main functions
    bool initializeFromURDF(const std::string& urdf_path);
    void publishVisualization();
    
    // Getters
    const std::vector<std::string>& getLinkNames() const { return link_names_; }
    const std::vector<BoxConfig>& getBoxConfigs() const { return box_configs_; }

    // OOBB computation helpers
    std::vector<Eigen::Vector3d> computeWorldVertices(
        const std::string& link_name,
        const Eigen::Vector3d& position,
        const Eigen::Quaterniond& orientation) const;

private:
    // ROS related
    ros::NodeHandle& nh_;
    ros::Publisher marker_pub_;
    std::string robot_name_;

    // Data storage
    std::vector<std::string> link_names_;
    std::vector<BoxConfig> box_configs_;

    // Helper functions
    bool loadFromYAML(const std::string& filepath);
    bool saveToYAML(const std::string& filepath) const;
    BoxConfig processURDFLink(const urdf::LinkSharedPtr& link, const std::string& package_path);
    visualization_msgs::MarkerArray createVisualizationMarkers() const;

    // OOBB computation helpers
    Eigen::Vector3d quatRotate(const Eigen::Quaterniond& q, const Eigen::Vector3d& v) const;
    std::vector<Eigen::Vector3d> transformVertices(
        const std::vector<Eigen::Vector3d>& local_vertices,
        const Eigen::Vector3d& position,
        const Eigen::Quaterniond& orientation) const;
};

#endif // BBOX_HPP_ 