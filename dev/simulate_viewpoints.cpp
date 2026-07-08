#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <filesystem>
#include <random>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

struct CameraModel {
    string name;
    Mat K;
    Mat D;
    int width;
    int height;
};

int main(int argc, char** argv) {
    const String keys =
        "{@image    | | Pattern image path }"
        "{@outdir   | | Output directory }"
        "{N         | 100 | Number of images to generate }"
        "{min_dist  | 0.15| Minimum distance (relative to pattern height) }"
        "{max_dist  | 4.0 | Maximum distance }"
        "{min_elev  | 0.0 | Minimum elevation angle (degrees) }"
        "{max_elev  | 80.0| Maximum elevation angle (degrees) }"
        "{min_yaw   | 0.0 | Minimum azimuth/yaw angle (degrees) }"
        "{max_yaw   | 360.0| Maximum azimuth/yaw angle (degrees) }"
        "{min_roll  |-180.0| Minimum camera roll angle (degrees) }"
        "{max_roll  | 180.0| Maximum camera roll angle (degrees) }"
        "{help h    | | Print help message }";

    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help") || !parser.check()) {
        parser.printMessage();
        return 0;
    }

    String img_path = parser.get<String>("@image");
    String out_dir = parser.get<String>("@outdir");
    if (img_path.empty() || out_dir.empty()) {
        cout << "Error: Pattern image and output directory must be provided." << endl;
        parser.printMessage();
        return -1;
    }

    int num_images = parser.get<int>("N");
    double min_dist = parser.get<double>("min_dist");
    double max_dist = parser.get<double>("max_dist");
    double min_elev = parser.get<double>("min_elev");
    double max_elev = parser.get<double>("max_elev");
    double min_yaw = parser.get<double>("min_yaw");
    double max_yaw = parser.get<double>("max_yaw");
    double min_roll = parser.get<double>("min_roll");
    double max_roll = parser.get<double>("max_roll");

    Mat pattern = imread(img_path, IMREAD_COLOR);
    if (pattern.empty()) {
        cout << "Could not read pattern image: " << img_path << endl;
        return -1;
    }

    if (!fs::exists(out_dir)) {
        fs::create_directories(out_dir);
    }

    // Pattern physical size (assume height is 1.0, width is proportional)
    double H_p = 1.0;
    double W_p = (double)pattern.cols / pattern.rows;

    // Define 4K Camera models
    int W_out = 3840;
    int H_out = 2160;

    vector<CameraModel> cameras;
    
    // Narrow FOV
    CameraModel cam_narrow;
    cam_narrow.name = "Narrow";
    cam_narrow.width = W_out; cam_narrow.height = H_out;
    cam_narrow.K = (Mat_<double>(3,3) << 3500, 0, W_out/2.0, 0, 3500, H_out/2.0, 0, 0, 1);
    cam_narrow.D = (Mat_<double>(1,5) << -0.01, 0.001, 0, 0, 0);
    cameras.push_back(cam_narrow);

    // Normal FOV
    CameraModel cam_normal;
    cam_normal.name = "Normal";
    cam_normal.width = W_out; cam_normal.height = H_out;
    cam_normal.K = (Mat_<double>(3,3) << 2500, 0, W_out/2.0, 0, 2500, H_out/2.0, 0, 0, 1);
    cam_normal.D = (Mat_<double>(1,5) << -0.02, 0.005, 0, 0, 0);
    cameras.push_back(cam_normal);

    // Wide FOV
    CameraModel cam_wide;
    cam_wide.name = "Wide";
    cam_wide.width = W_out; cam_wide.height = H_out;
    cam_wide.K = (Mat_<double>(3,3) << 1500, 0, W_out/2.0, 0, 1500, H_out/2.0, 0, 0, 1);
    cam_wide.D = (Mat_<double>(1,5) << -0.05, 0.01, 0, 0, 0);
    cameras.push_back(cam_wide);

    mt19937 rng(42); // Fixed seed for reproducibility
    uniform_real_distribution<double> dist_dist(min_dist, max_dist);
    uniform_real_distribution<double> dist_elev(min_elev * CV_PI / 180.0, max_elev * CV_PI / 180.0);
    uniform_real_distribution<double> dist_yaw(min_yaw * CV_PI / 180.0, max_yaw * CV_PI / 180.0);
    uniform_real_distribution<double> dist_roll(min_roll * CV_PI / 180.0, max_roll * CV_PI / 180.0);

    for (int i = 0; i < num_images; ++i) {
        double r = dist_dist(rng);
        double theta = dist_elev(rng); // Elevation from Z axis (0 is straight down)
        double phi = dist_yaw(rng);    // Azimuth
        double roll = dist_roll(rng);  // Camera roll

        // Camera position
        double cx = r * sin(theta) * cos(phi);
        double cy = r * sin(theta) * sin(phi);
        double cz = r * cos(theta);
        Vec3d C_w(cx, cy, cz);

        // Forward vector (looking at origin)
        Vec3d Z_c = -C_w;
        double norm_Z = norm(Z_c);
        if (norm_Z < 1e-6) Z_c = Vec3d(0,0,-1);
        else Z_c /= norm_Z;

        // Up vector (try to align with Y axis of world, but avoid singularity)
        Vec3d up(0, 1, 0);
        if (abs(Z_c.dot(up)) > 0.99) {
            up = Vec3d(1, 0, 0);
        }
        Vec3d X_c = Z_c.cross(up);
        X_c /= norm(X_c);
        Vec3d Y_c = Z_c.cross(X_c);

        // Apply roll around Z_c
        Matx33d R_roll;
        Rodrigues(Z_c * roll, R_roll);
        X_c = R_roll * X_c;
        Y_c = R_roll * Y_c;

        // Rotation matrix from world to camera
        Matx33d R;
        R(0,0) = X_c[0]; R(0,1) = X_c[1]; R(0,2) = X_c[2];
        R(1,0) = Y_c[0]; R(1,1) = Y_c[1]; R(1,2) = Y_c[2];
        R(2,0) = Z_c[0]; R(2,1) = Z_c[1]; R(2,2) = Z_c[2];

        Matx33d R_T = R.t();

        for (const auto& cam : cameras) {
            // Generate rays for each pixel
            Mat map_x(cam.height, cam.width, CV_32FC1, Scalar(-1));
            Mat map_y(cam.height, cam.width, CV_32FC1, Scalar(-1));

            vector<Point2f> dst_pts;
            dst_pts.reserve(cam.width * cam.height);
            for (int y = 0; y < cam.height; ++y) {
                for (int x = 0; x < cam.width; ++x) {
                    dst_pts.push_back(Point2f(x, y));
                }
            }

            vector<Point2f> undist_pts;
            undistortPoints(dst_pts, undist_pts, cam.K, cam.D);

            for (int p = 0; p < dst_pts.size(); ++p) {
                int px_x = p % cam.width;
                int px_y = p / cam.width;

                Vec3d v_c(undist_pts[p].x, undist_pts[p].y, 1.0);
                Vec3d v_w = R_T * v_c;

                // Intersect with Z=0 plane
                if (abs(v_w[2]) < 1e-6) continue;
                double lambda = -C_w[2] / v_w[2];
                if (lambda <= 0) continue; // Looking away or behind

                Vec3d P_w = C_w + lambda * v_w;

                // Map to pattern pixel
                double px = (P_w[0] + W_p / 2.0) / W_p * pattern.cols;
                // Note: Image Y is usually down, world Y is up in our definition if we want right-handed
                // Let's assume pattern Y corresponds directly (standard coordinate map)
                // World: Z is up. X and Y are pattern plane.
                double py = (P_w[1] + H_p / 2.0) / H_p * pattern.rows;

                if (px >= 0 && px < pattern.cols && py >= 0 && py < pattern.rows) {
                    map_x.at<float>(px_y, px_x) = px;
                    map_y.at<float>(px_y, px_x) = py;
                }
            }

            Mat sim_img;
            remap(pattern, sim_img, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(128, 128, 128));

            char filename[256];
            snprintf(filename, sizeof(filename), "sim_%03d_%s_d%.2f_e%.1f_y%.1f_r%.1f.jpg", 
                     i, cam.name.c_str(), r, theta * 180.0 / CV_PI, phi * 180.0 / CV_PI, roll * 180.0 / CV_PI);
            
            imwrite((fs::path(out_dir) / filename).string(), sim_img);
        }
        
        if ((i + 1) % 10 == 0) {
            cout << "Generated " << (i + 1) << " / " << num_images << " viewpoints (" << (i+1)*cameras.size() << " total images)" << endl;
        }
    }

    cout << "Simulation complete." << endl;
    return 0;
}
