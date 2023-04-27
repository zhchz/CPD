
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <mutex>
#include <string>
#include <cmath>
#include <random>
#include <unistd.h>
#include <dirent.h>
#include <sys/io.h>
#include <omp.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include "csv2/reader.hpp"
#include "matplot/matplot.h"
#include "ShapeRegistration.h"


int main(int argc, char* argv[])
{

#if 1  // test CPD 2d

    std::string csvx2dpath = "cpd_data2D_fish_X.csv";
    std::string csvy2dpath = "cpd_data2D_fish_Y.csv";

    std::vector<cv::Point2d> _x2d, _y2d;

    csv2::Reader<csv2::delimiter<','>, csv2::quote_character<'"'>, csv2::first_row_is_header<false>> csv;

    if (csv.mmap(csvx2dpath))
    {
        for (const auto row : csv)
        {
            int read = 0;
            int index = 0;
            double point[2];
            for (const auto cell : row)
            {
                std::string value;
                cell.read_value(value);

                double dbl = std::atof(value.c_str());

                point[index] = dbl;
                index++;

                read++;
            }

            if (read > 0)
            {
                _x2d.emplace_back(cv::Point2d(point[0], point[1]));
            }
        }
    }

    if (csv.mmap(csvy2dpath))
    {
        for (const auto row : csv)
        {
            int read = 0;
            int index = 0;
            double point[2];
            for (const auto cell : row)
            {
                std::string value;
                cell.read_value(value);

                double dbl = std::atof(value.c_str());

                point[index] = dbl;
                index++;

                read++;
            }

            if (read > 0)
            {
                _y2d.emplace_back(cv::Point2d(point[0], point[1]));
            }
        }
    }

    for (int mt = 0; mt < 4; ++mt)
    {
        std::vector<cv::Point2d> x2d, y2d;
        x2d = _x2d;
        if (mt < 2)
        {
            if (mt == 0)
            {
                y2d.resize(x2d.size());

                cv::Mat eye = cv::Mat::eye(2, 2, CV_64F);

                cv::Mat rmat(2, 2, CV_64F);
                cv::RNG rng((unsigned)time(NULL));
                rng.fill(rmat, cv::RNG::UNIFORM, 0., 1., false);

                cv::Mat afn = eye + rmat;

                for (size_t k = 0; k < x2d.size(); ++k)
                {
                    cv::Mat pt = (cv::Mat_<double>(2, 1) << x2d[k].x, x2d[k].y);
                    cv::Mat rpt = afn * pt;
                    y2d[k].x = rpt.at<double>(0, 0);
                    y2d[k].y = rpt.at<double>(0, 1);
                }
            }
            else if (mt == 1)
            {
                y2d.resize(x2d.size());

                cv::Mat R = cv::Mat::eye(2, 2, CV_64F);

                cv::RNG rng((unsigned)time(NULL));
                double a = rng.uniform(-CV_PI/2, CV_PI/2);
                R.at<double>(0, 0) = std::cos(a);
                R.at<double>(0, 1) = - std::sin(a);
                R.at<double>(1, 0) = std::sin(a);
                R.at<double>(1, 1) = std::cos(a);

                cv::Mat t = cv::Mat::zeros(2, 1, CV_64F);
                rng.fill(t, cv::RNG::UNIFORM, 0., 1., false);

                for (size_t k = 0; k < x2d.size(); ++k)
                {
                    cv::Mat pt = (cv::Mat_<double>(2, 1) << x2d[k].x, x2d[k].y);
                    cv::Mat rpt = R * pt + t;
                    y2d[k].x = rpt.at<double>(0, 0);
                    y2d[k].y = rpt.at<double>(0, 1);
                }
            }
        }
        else
        {
            y2d = _y2d;
        }

        for (int ft = 0; ft < 3; ++ft)
        {
            if (mt == 3)
            {
                for (int lr = 0; lr < 2; ++lr)
                {
                    std::vector<cv::Point2d> ry2d;
                    std::vector<int> correspondence;

                    Parameters para;
                    para.method = mt;
                    para.fgt_mode = ft;
                    if (lr == 0)
                    {
                        para.nonrigid_lowrank_eigen_fgt = false;
                    }
                    else
                    {
                        para.nonrigid_lowrank_eigen_fgt = true;
                    }

auto t0 = std::chrono::steady_clock::now();
                    ShapeRegistration::transform(x2d, y2d, ry2d, para, correspondence);
auto t1 = std::chrono::steady_clock::now();
auto tt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
std::cout << "time: " << tt << "ms" << std::endl;

                    std::vector<double> xu(x2d.size()), xv(x2d.size());
                    std::vector<double> ryu(ry2d.size()), ryv(ry2d.size());

                    for (size_t r = 0; r < x2d.size(); ++r)
                    {
                        xu[r] = x2d[r].x;
                        xv[r] = x2d[r].y;
                    }
                    
                    for (size_t r = 0; r < ry2d.size(); ++r)
                    {
                        ryu[r] = ry2d[r].x;
                        ryv[r] = ry2d[r].y;

                        // std::cout << ry2d[r] << std::endl;
                    }

                    matplot::axes_handle ax = matplot::gca();

                    matplot::plot(ax, xu, xv, "r*");
                    matplot::hold(true);
                    matplot::plot(ax, ryu, ryv, "bo");
                    matplot::hold(true);
                    for (size_t i = 0; i<correspondence.size(); ++i)
                    {
                        std::vector<double> u{y2d[i].x};
                        std::vector<double> v{y2d[i].y};
                        matplot::plot(ax, u, v, "go");
                        matplot::hold(true);

                        u.emplace_back(x2d[correspondence[i]].x);
                        v.emplace_back(x2d[correspondence[i]].y);

                        matplot::plot(ax, u, v, "b");
                        matplot::hold(true);
                    }

                    matplot::hold(false);
                    std::cout << "method_" + std::to_string(mt) + "_fgt_" + std::to_string(ft) + "_lowrankfgt_" + std::to_string(lr) << std::endl;
                    matplot::show();
                }
            }
            else
            {
                std::vector<cv::Point2d> ry2d;
                std::vector<int> correspondence;

                Parameters para;
                para.method = mt;
                para.fgt_mode = ft;

auto t0 = std::chrono::steady_clock::now();
                ShapeRegistration::transform(x2d, y2d, ry2d, para, correspondence);
auto t1 = std::chrono::steady_clock::now();
auto tt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
std::cout << "time: " << tt << "ms" << std::endl;

                std::vector<double> xu(x2d.size()), xv(x2d.size());
                std::vector<double> ryu(ry2d.size()), ryv(ry2d.size());

                for (size_t r = 0; r < x2d.size(); ++r)
                {
                    xu[r] = x2d[r].x;
                    xv[r] = x2d[r].y;
                }
                
                for (size_t r = 0; r < ry2d.size(); ++r)
                {
                    ryu[r] = ry2d[r].x;
                    ryv[r] = ry2d[r].y;

                    // std::cout << ry2d[r] << std::endl;
                }

                matplot::axes_handle ax = matplot::gca();

                matplot::plot(ax, xu, xv, "r*");
                matplot::hold(true);
                matplot::plot(ax, ryu, ryv, "bo");
                matplot::hold(true);
                for (size_t i = 0; i<correspondence.size(); ++i)
                {
                    std::vector<double> u{y2d[i].x};
                    std::vector<double> v{y2d[i].y};
                    matplot::plot(ax, u, v, "go");
                    matplot::hold(true);

                    u.emplace_back(x2d[correspondence[i]].x);
                    v.emplace_back(x2d[correspondence[i]].y);

                    matplot::plot(ax, u, v, "b");
                    matplot::hold(true);
                }

                matplot::hold(false);
                std::cout << "method_" + std::to_string(mt) + "_fgt_" + std::to_string(ft) << std::endl;
                matplot::show();
            }
        }
    }

#endif

#if 0  // test CPD 3d

    std::string csvx3dpath = "cpd_data3D_face_X.csv";
    std::string csvy3dpath = "cpd_data3D_face_Y.csv";

    std::vector<cv::Point3d> x3d, y3d;

    csv2::Reader<csv2::delimiter<','>, csv2::quote_character<'"'>, csv2::first_row_is_header<false>> csv;

    if (csv.mmap(csvx3dpath))
    {
        for (const auto row : csv)
        {
            int read = 0;
            int index = 0;
            double point[3];
            for (const auto cell : row)
            {
                std::string value;
                cell.read_value(value);

                double dbl = std::atof(value.c_str());

                point[index] = dbl;
                index++;

                read++;
            }

            if (read > 0)
            {
                x3d.emplace_back(cv::Point3d(point[0], point[1], point[2]));
            }
        }
    }

    if (csv.mmap(csvy3dpath))
    {
        for (const auto row : csv)
        {
            int read = 0;
            int index = 0;
            double point[3];
            for (const auto cell : row)
            {
                std::string value;
                cell.read_value(value);

                double dbl = std::atof(value.c_str());

                point[index] = dbl;
                index++;

                read++;
            }

            if (read > 0)
            {
                y3d.emplace_back(cv::Point3d(point[0], point[1], point[2]));
            }
        }
    }

    std::vector<cv::Point3d> ry3d;
    std::vector<int> correspondence;

    Parameters para;
    para.method = 3;
    para.fgt_mode = 0;
    ShapeRegistration::transform(x3d, y3d, ry3d, para, correspondence);

    std::vector<double> xu(x3d.size()), xv(x3d.size()), xw(x3d.size());
    std::vector<double> ryu(ry3d.size()), ryv(ry3d.size()), ryw(ry3d.size());

    for (size_t r = 0; r < x3d.size(); ++r)
    {
        xu[r] = x3d[r].x;
        xv[r] = x3d[r].y;
        xw[r] = x3d[r].z;
    }
    
    for (size_t r = 0; r < ry3d.size(); ++r)
    {
        ryu[r] = ry3d[r].x;
        ryv[r] = ry3d[r].y;
        ryw[r] = ry3d[r].z;

        std::cout << ry3d[r] << std::endl;
    }

    matplot::figure();
    matplot::axes_handle ax = matplot::gca();

    matplot::plot3(ax, xu, xv, xw, "r*");
    matplot::hold(true);
    matplot::plot3(ax, ryu, ryv, ryw, "bo");
    matplot::hold(true);
    for (size_t i = 0; i<correspondence.size(); ++i)
    {
        std::vector<double> u{y3d[i].x};
        std::vector<double> v{y3d[i].y};
        std::vector<double> w{y3d[i].z};
        matplot::plot3(ax, u, v, w, "go");
        matplot::hold(true);

        u.emplace_back(x3d[correspondence[i]].x);
        v.emplace_back(x3d[correspondence[i]].y);
        w.emplace_back(x3d[correspondence[i]].z);

        matplot::plot3(ax, u, v, w, "b");
        matplot::hold(true);
    }

    matplot::show();

#endif

    return 0;
}
