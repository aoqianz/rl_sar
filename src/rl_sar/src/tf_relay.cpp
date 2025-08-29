#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf/tf.h>

int main(int argc, char** argv) {
    ros::init(argc, argv, "tf_republisher");
    ros::NodeHandle node;

    tf::TransformListener listener;
    tf::TransformBroadcaster broadcaster;

    ros::Rate rate(100.0);
    while (node.ok()) {
        // geometry_msgs::TransformStamped world_to_dog;
        // geometry_msgs::TransformStamped world_to_stick;
        tf::StampedTransform world_to_dog;
        tf::StampedTransform world_to_stick;

        try {
            // Subscribe to the transforms
            listener.lookupTransform("world", "dog", ros::Time(0), world_to_dog);
            listener.lookupTransform("world", "stick", ros::Time(0), world_to_stick);

            // Create new transforms for republishing
            geometry_msgs::TransformStamped tf1;
            tf1.header.stamp = ros::Time::now();
            tf1.header.frame_id = "world";
            tf1.child_frame_id = "dog/ball_5";
            tf1.transform.translation.x = world_to_dog.getOrigin().x();
            tf1.transform.translation.y = world_to_dog.getOrigin().y();
            tf1.transform.translation.z = world_to_dog.getOrigin().z();
            tf1.transform.rotation.x = world_to_dog.getRotation().x();
            tf1.transform.rotation.y = world_to_dog.getRotation().y();
            tf1.transform.rotation.z = world_to_dog.getRotation().z();
            tf1.transform.rotation.w = world_to_dog.getRotation().w();

            geometry_msgs::TransformStamped tf2;
            tf2.header.stamp = ros::Time::now();
            tf2.header.frame_id = "world";
            tf2.child_frame_id = "stick/stick_ball_1";
            tf2.transform.translation.x = world_to_stick.getOrigin().x();
            tf2.transform.translation.y = world_to_stick.getOrigin().y();
            tf2.transform.translation.z = world_to_stick.getOrigin().z();
            tf2.transform.rotation.x = world_to_stick.getRotation().x();
            tf2.transform.rotation.y = world_to_stick.getRotation().y();
            tf2.transform.rotation.z = world_to_stick.getRotation().z();
            tf2.transform.rotation.w = world_to_stick.getRotation().w();

            // Publish the new transforms
            broadcaster.sendTransform(tf1);
            broadcaster.sendTransform(tf2);
        } catch (tf::TransformException &ex) {
            ROS_WARN("%s", ex.what());
        }

        rate.sleep();
    }
    return 0;
}
