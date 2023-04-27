#ifndef __SHAPEREGISTRATION_HPP__
#define __SHAPEREGISTRATION_HPP__


class Parameters
{

public:

    /**
     * method to choose
     * 0 -- using affine transformation
     * 1 -- using rigid transformation
     * 2 -- using non-rigid transformation
     * 3 -- using non-rigid transformation but with lowrank to compute the eigenvalues during the proccessing
     * 
     * [ 0: affine ]  [ 1: rigid ]  [ 2: nonrigid ]  [ 3: nonrigid_lowrank ]
     */
    int method = -1;
    
    /**
     * fast Gaussian transformation (fgt) mode to use
     * 0 -- using default method (no fgt) to compute Gaussian probability
     * 1 -- using fgt to approximate the Gaussian probability, with expected faster speed but lower accuracy
     * 2 -- using fgt and truncated strategy, with better accuracy than purely fgt, but still a balance between speed and accurcy
     * 
     * [ 0: default, no fgt ]  [ 1: fgt ]  [ 2: fgt and trunc ]
     */
    int fgt_mode = 0;

    /**
     * maximum iteration for convergency
    */
    int max_iteration = 150;

    /**
     * tolerance for convergency
    */
    double tolerance = 1e-5;

    /**
     * weight of outliers and noise
     * 
     * [0 ~ 1]
    */
    double outlier_weight = 0.1;

    /**
     * sigma2 == 0.0 means it will be adaptively determined, otherwise maually determined
     * 
     * [ > 0]
    */
    double sigma2 = 0.0;
    
    /**
     * true -- allow rigid rotation when using rigid method
     * false -- opposite
    */
    bool rigid_rotation = true;

    /**
     * true -- allow rigid scale when using rigid method
     * false -- opposite
    */
    bool rigid_scale = true;

    /**
     * Gaussian smoothing filter size when using non-rigid (or non-rigid lowrank) method, default 2
     * 
     * [ > 0 ]
    */
    double nonrigid_beta = 2;

    /**
     * regularization factor weight when using non-rigid (or non-rigid lowrank) method, default 3
     * 
     * [ > 0 ]
    */
    double nonrigid_lambda = 3;

    /**
     * the request largest eigen values and vectors' number when using non-rigid lowrand method to compute eigens, 
     * default 0, which mean it will be adaptively determined by M^(1/2), M is the length of point set y
     * 
     * [n ~ M], n means a constant bigger than 0, but show determined properly
    */
    int nonrigid_lowrank_number_eigen = 0;

    /**
     * true -- allow fgt fast compute eigens when using non-rigid lowrank method for large points set
     * false -- opposite
    */
    bool nonrigid_lowrank_eigen_fgt = false;
};

class ShapeRegistration
{

public:

    ShapeRegistration();
    ~ShapeRegistration();

    static bool transform(const std::vector<cv::Point2d>& x, const std::vector<cv::Point2d>& y, std::vector<cv::Point2d>& ry, Parameters para);
    static bool transform(const std::vector<cv::Point3d>& x, const std::vector<cv::Point3d>& y, std::vector<cv::Point3d>& ry, Parameters para);
    static bool transform(const std::vector<cv::Point2d>& x, const std::vector<cv::Point2d>& y, std::vector<cv::Point2d>& ry, Parameters para, std::vector<int>& corr);
    static bool transform(const std::vector<cv::Point3d>& x, const std::vector<cv::Point3d>& y, std::vector<cv::Point3d>& ry, Parameters para, std::vector<int>& corr);
};


/**
 * usage
*/
// {
//     // 1. creating data, read from file or create
//     std::vector<cv::Point2d> x2d;
//     std::vector<cv::Point2d> y2d;   
//     // ....
//
//     // 2. setting parameters
//     Parameters para;
//     para.method = 0;
//     para.fgt_mode = 0;
//     // some other setting to add here if needed (see `Parameters`)
//     // param....
//
//     // 3. call registration
//     std::vector<cv::Point2d> ry2d;
//     std::vector<int> correspondence;
//     ShapeRegistration::transform(x2d, y2d, ry2d, para, correspondence);
//     // result `ry2d` is transformed to `x2d`
//     // `correspondence` shows correspondence index from `ry2d` to `x2d`, e.g. the i-th point of `ry2d` correspondence to `correspondence[i]`-th point of x2d after tranformation
// }

#endif
