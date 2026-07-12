#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <filesystem>

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
        "{N_dist    | 20  | Number of distance steps }"
        "{min_dist  | 0.5| Minimum distance (relative to pattern height) }"
        "{max_dist  | 100.0 | Maximum distance }"
        "{N_angle   | 9   | Number of angle steps }"
        "{min_angle | 0.0 | Minimum angle (degrees) }"
        "{max_angle | 80.0| Maximum angle (degrees) }"
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

    int n_dist = parser.get<int>("N_dist");
    double min_dist = parser.get<double>("min_dist");
    double max_dist = parser.get<double>("max_dist");
    
    int n_angle = parser.get<int>("N_angle");
    double min_angle = parser.get<double>("min_angle");
    double max_angle = parser.get<double>("max_angle");

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

    // Normal FOV
    CameraModel cam_normal;
    cam_normal.name = "Normal";
    cam_normal.width = W_out; cam_normal.height = H_out;
    cam_normal.K = (Mat_<double>(3,3) << 2500, 0, W_out/2.0, 0, 2500, H_out/2.0, 0, 0, 1);
    cam_normal.D = (Mat_<double>(1,5) << -0.02, 0.005, 0, 0, 0);
    cameras.push_back(cam_normal);

    int total_images = n_dist * n_angle * cameras.size();
    int count = 0;

    for (int i_d = n_dist-1; i_d >=0 ; --i_d) {
        double r = n_dist > 1 ? min_dist + i_d * (max_dist - min_dist) / (n_dist - 1) : min_dist;
        
        for (int i_a = 0; i_a < n_angle; ++i_a) {
            double angle_deg = n_angle > 1 ? min_angle + i_a * (max_angle - min_angle) / (n_angle - 1) : min_angle;
            double theta = angle_deg * CV_PI / 180.0; // Elevation from Z axis
            
            // Fix yaw and roll to 0 for simplicity
            double phi = 0.0;
            double roll = 0.0;

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

            // Up vector (try to align with Y axis of world)
            Vec3d up(0, 1, 0);
            if (abs(Z_c.dot(up)) > 0.99) {
                up = Vec3d(1, 0, 0);
            }
            Vec3d X_c = Z_c.cross(up);
            X_c /= norm(X_c);
            Vec3d Y_c = Z_c.cross(X_c);

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
                    double py = (H_p / 2.0 - P_w[1]) / H_p * pattern.rows;

                    if (px >= 0 && px < pattern.cols && py >= 0 && py < pattern.rows) {
                        map_x.at<float>(px_y, px_x) = px;
                        map_y.at<float>(px_y, px_x) = py;
                    }
                }

                Mat sim_img;
                remap(pattern, sim_img, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(128, 128, 128));

                char filename[256];
                snprintf(filename, sizeof(filename), "sim_%s_dist%.2f_angle%.1f.jpg", 
                         cam.name.c_str(), r, angle_deg);
                
                imwrite((fs::path(out_dir) / filename).string(), sim_img);
                
                count++;
            }
        }
        cout << "Generated " << count << " / " << total_images << " images..." << endl;
    }
    
    cout << "Simulation complete." << endl;
    return 0;
}
