#include "bbox.hpp"
#include <ros/package.h>

BoundingBoxManager::BoundingBoxManager(ros::NodeHandle& nh) : nh_(nh) {
    marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("bbox_visualization", 1, true);
}

bool BoundingBoxManager::initializeFromURDF(const std::string& urdf_path) {
    // Get robot name from URDF path
    size_t last_slash = urdf_path.find_last_of("/");
    size_t last_dot = urdf_path.find_last_of(".");
    robot_name_ = urdf_path.substr(last_slash + 1, last_dot - last_slash - 1);

    // Try to load existing bbox config
    std::string bbox_yaml = urdf_path.substr(0, last_slash + 1) + robot_name_ + "_bbox.yaml";
    if (std::filesystem::exists(bbox_yaml)) {
        if (loadFromYAML(bbox_yaml)) {
            ROS_INFO("Loaded existing bounding box configuration from %s", bbox_yaml.c_str());
            return true;
        }
    }

    // If no existing config, generate from URDF
    urdf::Model urdf_model;
    if (!urdf_model.initFile(urdf_path)) {
        ROS_ERROR("Failed to parse URDF file");
        return false;
    }

    // Clear existing data
    link_names_.clear();
    box_configs_.clear();

    // Get package path for mesh file resolution
    std::string package_path = ros::package::getPath("go2_description_xzh");

    // Process each link
    for (const auto& link_pair : urdf_model.links_) {
        const auto& link = link_pair.second;
        if (!link->collision || !link->collision->geometry) {
            continue;
        }

        BoxConfig config = processURDFLink(link, package_path);
        if (!config.link_name.empty()) {
            link_names_.push_back(config.link_name);
            box_configs_.push_back(config);
        }
    }

    // Save the configuration
    if (!box_configs_.empty()) {
        saveToYAML(bbox_yaml);
        ROS_INFO("Generated and saved new bounding box configuration to %s", bbox_yaml.c_str());
    }

    return !box_configs_.empty();
}

BoundingBoxManager::BoxConfig BoundingBoxManager::processURDFLink(
    const urdf::LinkSharedPtr& link,
    const std::string& package_path) {
    
    BoxConfig config;
    config.link_name = link->name;

    urdf::GeometrySharedPtr geom = link->collision->geometry;
    urdf::Pose collision_origin = link->collision->origin;

    // Set offset from collision origin
    config.offset = {
        collision_origin.position.x,
        collision_origin.position.y,
        collision_origin.position.z
    };

    switch (geom->type) {
        case urdf::Geometry::BOX: {
            auto box = std::static_pointer_cast<urdf::Box>(geom);
            config.size = {box->dim.x, box->dim.y, box->dim.z};
            break;
        }
        case urdf::Geometry::CYLINDER: {
            auto cylinder = std::static_pointer_cast<urdf::Cylinder>(geom);
            config.size = {
                cylinder->radius * 2,
                cylinder->radius * 2,
                cylinder->length
            };
            break;
        }
        case urdf::Geometry::SPHERE: {
            auto sphere = std::static_pointer_cast<urdf::Sphere>(geom);
            double dim = sphere->radius * 2;
            config.size = {dim, dim, dim};
            break;
        }
        case urdf::Geometry::MESH: {
            auto mesh = std::static_pointer_cast<urdf::Mesh>(geom);
            std::string mesh_path = mesh->filename;
            
            // Convert mesh path
            if (mesh_path.find("package://") == 0) {
                mesh_path = mesh_path.substr(10);
                size_t pos = mesh_path.find("/");
                if (pos != std::string::npos) {
                    mesh_path = package_path + mesh_path.substr(pos);
                }
            }

            // Load and process mesh
            Assimp::Importer importer;
            const aiScene* scene = importer.ReadFile(mesh_path, 
                aiProcess_Triangulate | aiProcess_JoinIdenticalVertices);

            if (scene && scene->mMeshes[0]) {
                aiVector3D min(1e10f, 1e10f, 1e10f);
                aiVector3D max(-1e10f, -1e10f, -1e10f);
                aiMesh* mesh_data = scene->mMeshes[0];

                for(unsigned int i = 0; i < mesh_data->mNumVertices; i++) {
                    aiVector3D& vertex = mesh_data->mVertices[i];
                    min.x = std::min(min.x, vertex.x);
                    min.y = std::min(min.y, vertex.y);
                    min.z = std::min(min.z, vertex.z);
                    max.x = std::max(max.x, vertex.x);
                    max.y = std::max(max.y, vertex.y);
                    max.z = std::max(max.z, vertex.z);
                }

                config.size = {
                    (max.x - min.x) * mesh->scale.x,
                    (max.y - min.y) * mesh->scale.y,
                    (max.z - min.z) * mesh->scale.z
                };

                std::vector<double> mesh_center = {
                    (max.x + min.x) * 0.5f * mesh->scale.x,
                    (max.y + min.y) * 0.5f * mesh->scale.y,
                    (max.z + min.z) * 0.5f * mesh->scale.z
                };

                config.offset = {
                    config.offset[0] + mesh_center[0],
                    config.offset[1] + mesh_center[1],
                    config.offset[2] + mesh_center[2]
                };
            }
            break;
        }
        default:
            config.link_name = "";  // Mark as invalid
            break;
    }

    return config;
}

bool BoundingBoxManager::loadFromYAML(const std::string& filepath) {
    try {
        YAML::Node config = YAML::LoadFile(filepath);
        
        link_names_ = config["link_names"].as<std::vector<std::string>>();
        
        const YAML::Node& boxes = config["box_configs"];
        box_configs_.clear();
        for (const auto& box : boxes) {
            BoxConfig config;
            config.link_name = box["link_name"].as<std::string>();
            config.size = box["size"].as<std::vector<double>>();
            config.offset = box["offset"].as<std::vector<double>>();
            box_configs_.push_back(config);
        }
        
        return true;
    } catch (const std::exception& e) {
        ROS_ERROR("Failed to load bounding box configuration: %s", e.what());
        return false;
    }
}

bool BoundingBoxManager::saveToYAML(const std::string& filepath) const {
    try {
        YAML::Node config;
        config["link_names"] = link_names_;
        
        YAML::Node boxes;
        for (const auto& box_config : box_configs_) {
            YAML::Node box;
            box["link_name"] = box_config.link_name;
            box["size"] = box_config.size;
            box["offset"] = box_config.offset;
            boxes.push_back(box);
        }
        config["box_configs"] = boxes;
        
        std::ofstream fout(filepath);
        fout << config;
        return true;
    } catch (const std::exception& e) {
        ROS_ERROR("Failed to save bounding box configuration: %s", e.what());
        return false;
    }
}

visualization_msgs::MarkerArray BoundingBoxManager::createVisualizationMarkers() const {
    visualization_msgs::MarkerArray marker_array;
    
    for (size_t i = 0; i < box_configs_.size(); ++i) {
        const auto& config = box_configs_[i];
        
        visualization_msgs::Marker marker;
        marker.header.frame_id = config.link_name;
        marker.header.stamp = ros::Time::now();
        marker.ns = "bounding_boxes";
        marker.id = i;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;
        
        // Set position (offset)
        marker.pose.position.x = config.offset[0];
        marker.pose.position.y = config.offset[1];
        marker.pose.position.z = config.offset[2];
        
        // Set orientation (identity quaternion)
        marker.pose.orientation.w = 1.0;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        
        // Set scale (size)
        marker.scale.x = config.size[0];
        marker.scale.y = config.size[1];
        marker.scale.z = config.size[2];
        
        // Set color (semi-transparent)
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 0.3;
        
        marker_array.markers.push_back(marker);
    }
    
    return marker_array;
}

void BoundingBoxManager::publishVisualization() {
    auto markers = createVisualizationMarkers();
    marker_pub_.publish(markers);
}

Eigen::Vector3d BoundingBoxManager::quatRotate(
    const Eigen::Quaterniond& q, 
    const Eigen::Vector3d& v) const {
    return q * v;
}

std::vector<Eigen::Vector3d> BoundingBoxManager::transformVertices(
    const std::vector<Eigen::Vector3d>& local_vertices,
    const Eigen::Vector3d& position,
    const Eigen::Quaterniond& orientation) const {
    
    std::vector<Eigen::Vector3d> world_vertices;
    world_vertices.reserve(local_vertices.size());
    
    for (const auto& local_vertex : local_vertices) {
        world_vertices.push_back(quatRotate(orientation, local_vertex) + position);
    }
    
    return world_vertices;
}

std::vector<Eigen::Vector3d> BoundingBoxManager::computeWorldVertices(
    const std::string& link_name,
    const Eigen::Vector3d& position,
    const Eigen::Quaterniond& orientation) const {
    
    // 查找对应的box配置
    auto it = std::find_if(box_configs_.begin(), box_configs_.end(),
        [&link_name](const BoxConfig& config) {
            return config.link_name == link_name;
        });
    
    if (it == box_configs_.end()) {
        ROS_WARN_STREAM("No box config found for link: " << link_name);
        return std::vector<Eigen::Vector3d>();
    }

    // 构造局部OOBB顶点
    std::vector<Eigen::Vector3d> local_vertices = {
        Eigen::Vector3d(-1, -1, -1), Eigen::Vector3d(1, -1, -1),
        Eigen::Vector3d(-1, 1, -1), Eigen::Vector3d(1, 1, -1),
        Eigen::Vector3d(-1, -1, 1), Eigen::Vector3d(1, -1, 1),
        Eigen::Vector3d(-1, 1, 1), Eigen::Vector3d(1, 1, 1)
    };

    // 应用尺寸和偏移
    Eigen::Vector3d size_vec(it->size[0]/2, it->size[1]/2, it->size[2]/2);
    Eigen::Vector3d offset_vec(it->offset[0], it->offset[1], it->offset[2]);
    
    for (auto& vertex : local_vertices) {
        vertex = vertex.cwiseProduct(size_vec) + offset_vec;
    }

    return transformVertices(local_vertices, position, orientation);
}