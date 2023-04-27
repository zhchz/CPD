#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "headers.h"
#include "ShapeRegistration.h"
#include "Spectra/KrylovSchurEigsSolver.h"
#include "Spectra/SymEigsSolver.h"


#ifdef __cplusplus
extern "C"
{
#include "auxiliary.h"
}
#endif

#define pi 3.1415926535897
#define eps 1e-10

#define max(A, B) ((A) > (B) ? (A) : (B))
#define min(A, B) ((A) < (B) ? (A) : (B))


typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;
typedef Eigen::Matrix<Matrix::Index, Eigen::Dynamic, 1> IndexVector;
typedef Eigen::ArrayXd Array;


static void call_fgt_model(const Matrix& _xt, const Matrix& _wt, double _sigma, double _e, int _K, int _p, Matrix& _xc_t, Matrix& _A_k_t)
{
    const double* x = _xt.data();
    const double* w = _wt.data();

    double sigma = _sigma;
    double e = _e;

    int d, Nx;

    d = _xt.rows();
    Nx = _xt.cols();

    int K = min(Nx, _K);
    assert(K <= Nx);

    int p = _p;
    int pd = fgt_nchoosek(p + d - 1, d);

    _xc_t.setZero(d, K);
    _A_k_t.setZero(pd, K);

    double* xc = _xc_t.data();
    double* A_k = _A_k_t.data();

    double* dist_C = (double*)malloc(Nx * sizeof(double));
    double* C_k = (double*)malloc(pd * sizeof(double));
    double* dx = (double*)malloc(d * sizeof(double));
    double* prods = (double*)malloc(pd * sizeof(double));

    int* indxc = (int*)malloc(K * sizeof(int));
    int* indx = (int*)malloc(Nx * sizeof(int));
    int* xhead = (int*)malloc(K * sizeof(int));
    int* xboxsz = (int*)malloc(K * sizeof(int));
    int* heads = (int*)malloc((d + 1) * sizeof(int));
    int* cinds = (int*)malloc(pd * sizeof(int));

    fgt_model(x, w, sigma, p, K, e, xc, A_k, d, Nx, indxc, indx, xhead, xboxsz, dist_C, C_k, heads, cinds, dx, prods, pd);

    free(indxc);
    free(indx);
    free(xhead);
    free(xboxsz);
    free(dist_C);
    free(C_k);
    free(heads);
    free(cinds);
    free(dx);
    free(prods);

    return;
}

static void call_fgt_predict(const Matrix& _yt, const Matrix& _xc_t, const Matrix& _A_k_t, double _sigma, double _e, Matrix& _vt)
{
    const double* y = _yt.data();
    const double* xc = _xc_t.data();
    const double* A_k = _A_k_t.data();
    double sigma = _sigma;
    double e = _e;

    int d, pd, K, Ny;

    d = _yt.rows();
    Ny = _yt.cols();

    assert(d == _xc_t.rows());
    K = _xc_t.cols();

    pd = _A_k_t.rows();
    assert(K == _A_k_t.cols());

    _vt.setZero(1, Ny);
    double* v = _vt.data();

    double* dx = (double*)malloc(d * sizeof(double));
    double* prods = (double*)malloc(pd * sizeof(double));
    int* heads = (int*)malloc((d + 1) * sizeof(int));

    fgt_predict(y, xc, A_k, Ny, sigma, K, e, d, pd, v, dx, prods, heads);

    free(heads);
    free(prods);
    free(dx);

    return;
}

static void call_cpd_p(const Matrix& _x, const Matrix& _y, double _sigma2, double _outlier, Matrix& _Px, Vector& _P1, Vector& _Pt1, double* _E)
{
    const double* x = _x.data();
    const double* y = _y.data();
    const double* sigma2 = &_sigma2;
    const double* outlier = &_outlier;

    int N, M, D;
    N = _x.rows();
    M = _y.rows();
    D = _x.cols();
    assert(D == _y.cols());
    
    _Px.setZero(M, D);
    _P1.setZero(M);
    _Pt1.setZero(N);

    double* Px = _Px.data();
    double* P1 = _P1.data();
    double* Pt1 = _Pt1.data();

    double* E = _E;
    *E = 0;

    cpd_comp(x, y, sigma2, outlier, P1, Pt1, Px, E, N, M, D);

    return;
}

static void call_cpd_p_trunc(const Matrix& _x, const Matrix& _y, double _sigma2, double _outlier, Matrix& _Px, Vector& _P1, Vector& _Pt1, double* _truncate, double* _E)
{
    const double* x = _x.data();
    const double* y = _y.data();
    const double* sigma2 = &_sigma2;
    double* outlier = &_outlier;
    double* truncate = _truncate;

    int N, M, D;
    N = _x.rows();
    M = _y.rows();
    D = _x.cols();
    assert(D == _y.cols());

    _Px.setZero(M, D);
    _P1.setZero(M);
    _Pt1.setZero(N);

    double* Px = _Px.data();
    double* P1 = _P1.data();
    double* Pt1 = _Pt1.data();

    double* E = _E;
    *E = 0;

    /* Do the actual computations in a subroutine */
    cpd_comp_trunc(x, y, sigma2, outlier, P1, Pt1, Px, E, N, M, D, truncate);

    return;
}

static void call_cpd_p_fgt(const Matrix& _x, const Matrix& _y, double _sigma2, double _outlier, double _sigma2_init, Vector& _Pt1, Vector& _P1, Matrix& _Px, double* L)
{
    int N = _x.rows();
    int D = _x.cols();
    int M = _y.rows();
    assert(D == _y.cols());

    double hsigma = std::sqrt(2 * _sigma2);

    double e = 9;
    int K = std::round(min(N, min(M, 50 + _sigma2_init / _sigma2)));
    int p = 6;

    Matrix _xt = _x.transpose();
    Matrix _yt = _y.transpose();
    Matrix _wt = Matrix::Ones(1, M);

    Matrix xc, A_k;
    call_fgt_model(_yt, _wt, hsigma, e, K, p, xc, A_k);

    Matrix Kt1;
    call_fgt_predict(_xt, xc, A_k, hsigma, e, Kt1);

    double ndi = _outlier / (1 - _outlier) * M / N * std::pow((2 * pi * _sigma2), 0.5 * D);
    Matrix denomP = Kt1 + Matrix::Ones(Kt1.rows(), Kt1.cols()) * (ndi);

    Matrix __Pt1 = Matrix::Ones(denomP.rows(), denomP.cols()) - ndi * denomP.cwiseInverse();
    _Pt1 = __Pt1.row(0);

    call_fgt_model(_xt, 1 * denomP.cwiseInverse(), hsigma, e, K, p, xc, A_k);

    Matrix __P1;
    call_fgt_predict(_yt, xc, A_k, hsigma, e, __P1);
    _P1 = __P1.row(0);

    _Px.setZero(M, D);
    for (int i = 0; i < D; ++i)
    {
        call_fgt_model(_xt, _xt.row(i).cwiseQuotient(denomP), hsigma, e, K, p, xc, A_k);
        
        Matrix v;
        call_fgt_predict(_yt, xc, A_k, hsigma, e, v);

        _Px.col(i) = v.row(0);
    }

    *L = -(denomP.array().log().sum()) + D * N * std::log(_sigma2) / 2;

    return;
}

static void call_cpd_p_fast(const Matrix& _x, const Matrix& _y, double _sigma2, double _outlier, double _sigma2_init, int fgt_mode, Vector& _Pt1, Vector& _P1, Matrix& _Px, double* L)
{
    if (fgt_mode == 1)
    {
        if (_sigma2 < 0.05)
        {
            _sigma2 = 0.05;
        }

        call_cpd_p_fgt(_x, _y, _sigma2, _outlier, _sigma2_init, _Pt1, _P1, _Px, L);
    }
    else if (fgt_mode == 2)
    {
        if (_sigma2 > 0.015 * _sigma2_init)
        {
            call_cpd_p_fgt(_x, _y, _sigma2, _outlier, _sigma2_init, _Pt1, _P1, _Px, L);
        }
        else
        {
            double truncate = 0.001;
            call_cpd_p_trunc(_x, _y, _sigma2, _outlier, _Px, _P1, _Pt1, &truncate, L);
        }
    }
    else
    {
        std::cout << "fgt mode here should be 0, 1 or 2." << std::endl;
    }

    return;
}

static void call_cpd_p_correspondence(const Matrix& _x, const Matrix& _y, double _sigma2, double _outlier, std::vector<int>& _Pc)
{
    const double* x = _x.data();
    const double* y = _y.data();
    const double* sigma2 = &_sigma2;
    const double* outlier = &_outlier;

    int N, M, D;
    N = _x.rows();
    M = _y.rows();
    D = _x.cols();
    assert(D == _y.cols());

    _Pc.resize(M, 0);
    int* Pc = &_Pc[0];

    cpd_correpondence(x, y, sigma2, outlier, Pc, N, M, D);

    return;
}

static void call_fgt_model_predict_qs(const double* _xt, int d, int Nx, const double* _wt, double _sigma, double _e, int _K, int _p, double* _vt)
{
    /////// model ///////

    const double* x = _xt;
    const double* w = _wt;

    double sigma = _sigma;
    double e = _e;

    int K = min(Nx, _K);
    assert(K <= Nx);

    int p = _p;
    int pd = fgt_nchoosek(p + d - 1, d);

    // must be allocated outside
    double* xc = (double*)malloc((d * K) * sizeof(double));
    double* A_k = (double*)malloc((pd * K) * sizeof(double));

    double* dist_C = (double*)malloc(Nx * sizeof(double));
    double* C_k = (double*)malloc(pd * sizeof(double));
    double* dx = (double*)malloc(d * sizeof(double));
    double* prods = (double*)malloc(pd * sizeof(double));

    int* indxc = (int*)malloc(K * sizeof(int));
    int* indx = (int*)malloc(Nx * sizeof(int));
    int* xhead = (int*)malloc(K * sizeof(int));
    int* xboxsz = (int*)malloc(K * sizeof(int));
    int* heads = (int*)malloc((d + 1) * sizeof(int));
    int* cinds = (int*)malloc(pd * sizeof(int));

    fgt_model(x, w, sigma, p, K, e, xc, A_k, d, Nx, indxc, indx, xhead, xboxsz, dist_C, C_k, heads, cinds, dx, prods, pd);

    free(indxc);
    free(indx);
    free(xhead);
    free(xboxsz);
    free(dist_C);
    free(C_k);
    free(heads);
    free(cinds);
    free(dx);
    free(prods);

    /////// preict ///////

    const double* y = _xt;

    int Ny = Nx;

    // _vt = (double*)malloc(Ny * sizeof(double));
    double* v = _vt;

    double* dx_ = (double*)malloc(d * sizeof(double));
    double* prods_ = (double*)malloc(pd * sizeof(double));
    int* heads_ = (int*)malloc((d + 1) * sizeof(int));

    fgt_predict(y, xc, A_k, Ny, sigma, K, e, d, pd, v, dx_, prods_, heads_);

    free(heads_);
    free(prods_);
    free(dx_);
    free(xc);
    free(A_k);

    return;
}

class TransformMatrix
{

private:

    const double* yt;
    double hsigma, e;
    int K, p;
    int d, Nyt;

    int TM_rows, TM_cols;

public:

    TransformMatrix(const Matrix& _yt, double _hsigma, double _e, int _K, int _p)
    {
        TM_rows = TM_cols = _yt.cols();
        yt = _yt.data();

        d = _yt.rows();
        Nyt = _yt.cols();
        hsigma = _hsigma;
        e = _e;
        K = _K;
        p = _p;
    }

    using Scalar = double;

    int rows() const
    {
        return TM_rows;
    }

    int cols() const
    {
        return TM_cols;
    }

    void perform_op(const double* _in, double* _out) const
    {
        call_fgt_model_predict_qs(yt, d, Nyt, _in, hsigma, e, K, p, _out);  
    }

};

class Result
{

public:

    class Normalization
    {

    public:

        Vector x_mean;
        Matrix x;
        double x_scale;

        Vector y_mean;
        Matrix y;
        double y_scale;

        // if `linked = true`, apply the same scaling to both sets of points.
        // that means data that should not be scaled,
        // if `linked = false`, each point set is scaled seperately.
        // Myronenko's original implementation only had `linked = false` logic.
        void operator()(const Matrix& _x, const Matrix& _y, bool _linked = false)
        {
            x_mean = _x.colwise().mean();
            x = _x - x_mean.transpose().replicate(_x.rows(), 1);
            x_scale = std::sqrt(x.array().pow(2).sum() / x.rows());

            y_mean = _y.colwise().mean(),
            y = _y - y_mean.transpose().replicate(_y.rows(), 1);
            y_scale = std::sqrt(y.array().pow(2).sum() / y.rows());

            if (_linked)
            {
                double scale = max(x_scale, y_scale);
                x_scale = scale;
                y_scale = scale;
            }
            x /= x_scale;
            y /= y_scale;

            return;
        }
    };
    
    Normalization normalizer;
    
    int iteration;
    double n_tolerance;
    double sigma2;
    double L;

    Matrix probability_Px;
    Vector probability_P1;
    Vector probability_Pt1;

    Matrix fix_points;
    Matrix points;
    IndexVector correspondence;

    void normalize(const Matrix& x, const Matrix& y, Matrix& nx, Matrix& ny)
    {
        normalizer(x, y);
        nx = normalizer.x;
        ny = normalizer.y;
        return;
    };

    virtual void denormalize()
    {
        points = points * normalizer.x_scale + normalizer.x_mean.transpose().replicate(points.rows(), 1);
        return;
    }

};

class AffineResult : public Result
{

public:

    Matrix rotation_B;
    Vector translation_t;

    void denormalize()
    {
        translation_t = normalizer.x_scale * translation_t + normalizer.x_mean - rotation_B * normalizer.y_mean;
        rotation_B = rotation_B * normalizer.x_scale / normalizer.y_scale;
        Result::denormalize();
        return;
    }
};

class RigidResult : public Result
{

public:

    Matrix rotation_R;
    Vector translation_t;
    double scale_s;

    void denormalize()
    {
        scale_s = scale_s * (normalizer.x_scale / normalizer.y_scale);
        translation_t = normalizer.x_scale * translation_t + normalizer.x_mean - scale_s * (rotation_R * normalizer.y_mean);
        rotation_R = rotation_R * scale_s;
        Result::denormalize();
        return;
    }
};

class NonrigidResult : public Result
{

public:

    Matrix transform_W;
};

template <class RegisterResult>
class Register
{

protected:

    bool done;
    Matrix ox;
    int Nx, My, Dm;
    Matrix onx, onxt, ony, onyt, onx2, ony2;

public:

    Parameters parameter;
    RegisterResult result;

    virtual void init(const Matrix& x, const Matrix& y, Parameters& p)
    {
        ox = x;
        done = false;

        Nx = x.rows();
        My = y.rows();
        Dm = x.cols();
        assert(Dm == y.cols());

        result.normalize(x, y, onx, ony);
        onxt = onx.transpose();
        onyt = ony.transpose();
        onx2 = onx.cwiseProduct(onx);
        ony2 = ony.cwiseProduct(ony);

        Matrix& nx = onx;
        Matrix& ny = ony;

        if (std::abs(p.sigma2) < 1e-6)
        {
            p.sigma2 = (My * (nx.transpose() * nx).trace() + Nx * (ny.transpose() * ny).trace() - 2 * nx.colwise().sum() * ny.colwise().sum().transpose()) / (My * Nx * Dm);
        }

        if (p.nonrigid_lowrank_number_eigen == 0)
        {
            p.nonrigid_lowrank_number_eigen = (int) std::sqrt(My);
        }

        parameter = p;

        result.iteration = 0;
        result.n_tolerance = parameter.tolerance + 10;
        result.sigma2 = parameter.sigma2;
        
        result.L = 1;
        result.points = ny;

        return;
    }

    virtual void step() = 0;

    void run()
    {
        while (result.iteration < parameter.max_iteration && result.n_tolerance > parameter.tolerance && result.sigma2 > 10 * eps)
        {
            step();
        }

        result.denormalize();

        done = true;

        return;
    }

    void correspondence(std::vector<int>& corr)
    {
        if (!done)
        {
            std::cout << "Not converged! " << std::endl;
            return;
        }

        call_cpd_p_correspondence(ox, result.points, result.sigma2, parameter.outlier_weight, corr);
    }
};

class AffineRegister : public Register<AffineResult>
{

public:

    AffineRegister() :
        Register<AffineResult>() {}

    void init(const Matrix& x, const Matrix& y, Parameters& p)
    {
        Register<AffineResult>::init(x, y, p);
        result.translation_t = Vector::Zero(Dm);
        result.rotation_B = Matrix::Identity(Dm, Dm);
        return;
    }

    void step()
    {
        Matrix& Px = result.probability_Px;
        Vector& P1 = result.probability_P1;
        Vector& Pt1 = result.probability_Pt1;
        double& L = result.L;

        double L_old = result.L;

        if (parameter.fgt_mode == 0)
        {
            call_cpd_p(onx, result.points, result.sigma2, parameter.outlier_weight, Px, P1, Pt1, &L);
        }
        else
        {
            call_cpd_p_fast(onx, result.points, result.sigma2, parameter.outlier_weight, parameter.sigma2, parameter.fgt_mode, Pt1, P1, Px, &L);
        }

        result.n_tolerance = std::abs((L - L_old) / L);

        std::cout << "result: " << std::endl;
        std::cout << "iteration: " << result.iteration << std::endl;
        std::cout << "n_tolerance: " << result.n_tolerance << std::endl;
        std::cout << "sigma2: " << result.sigma2 << std::endl << std::endl;

        double Np = Pt1.sum();
        Vector mu_x = onxt * Pt1 / Np;
        Vector mu_y = onyt * P1 / Np;

        Matrix B1 = Px.transpose() * ony - Np * (mu_x * mu_y.transpose());
        Matrix B2 = (ony.cwiseProduct(P1.replicate(1, Dm))).transpose() * ony - Np * (mu_y * mu_y.transpose());

        result.rotation_B = B1 * B2.inverse();
        result.translation_t = mu_x - result.rotation_B * mu_y;

        result.sigma2 = std::abs((onx2.cwiseProduct(Pt1.replicate(1, Dm))).sum() - Np * mu_x.transpose() * mu_x - (B1 * result.rotation_B.transpose()).trace()) / (Np * Dm);

        result.points = ony * result.rotation_B.transpose() + (result.translation_t.transpose()).replicate(My, 1);

        result.iteration += 1;

        return;
    }
};

class RigidRegister : public Register<RigidResult>
{

public:

    RigidRegister() :
        Register<RigidResult>() {}

    void init(const Matrix& x, const Matrix& y, Parameters& p)
    {
        Register<RigidResult>::init(x, y, p);
        result.scale_s = 1;
        result.translation_t = Vector::Zero(Dm);
        result.rotation_R = Matrix::Identity(Dm, Dm);
        result.L = 0;
        return;
    }

    void step()
    {
        Matrix& Px = result.probability_Px;
        Vector& P1 = result.probability_P1;
        Vector& Pt1 = result.probability_Pt1;
        double& L = result.L;

        double L_old = result.L;

        if (parameter.fgt_mode == 0)
        {
            call_cpd_p(onx, result.points, result.sigma2, parameter.outlier_weight, Px, P1, Pt1, &L);
        }
        else
        {
            call_cpd_p_fast(onx, result.points, result.sigma2, parameter.outlier_weight, parameter.sigma2, parameter.fgt_mode, Pt1, P1, Px, &L);
        }

        result.n_tolerance = std::abs((L - L_old) / L);

        std::cout << "result: " << std::endl;
        std::cout << "iteration: " << result.iteration << std::endl;
        std::cout << "n_tolerance: " << result.n_tolerance << std::endl;
        std::cout << "sigma2: " << result.sigma2 << std::endl << std::endl;

        double Np = Pt1.sum();
        Vector mu_x = onxt * Pt1 / Np;
        Vector mu_y = onyt * P1 / Np;

        Matrix A = Px.transpose() * ony - Np * (mu_x * mu_y.transpose());
        Eigen::JacobiSVD<Matrix> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Matrix C = Matrix::Identity(Dm, Dm);
        Matrix U = svd.matrixU();
        Matrix V = svd.matrixV();
        Matrix S = U.inverse() * A * V.transpose().inverse();

        if (parameter.rigid_rotation)
        {
            C(Dm - 1, Dm - 1) = (U * V.transpose()).determinant();
        }

        result.rotation_R = U * C * V.transpose();


        if (parameter.rigid_scale)
        {
            result.scale_s = (S * C).trace() / ((ony2.cwiseProduct(P1.replicate(1, Dm))).sum() - Np * mu_y.transpose() * mu_y);
            result.sigma2 = std::abs((onx2.cwiseProduct(Pt1.replicate(1, Dm))).sum() - Np * mu_x.transpose() * mu_x - result.scale_s * (S * C).trace()) / (Np * Dm);
        }
        else
        {
            result.sigma2 = std::abs((onx2.cwiseProduct(Pt1.replicate(1, Dm))).sum() - Np * mu_x.transpose() * mu_x + (ony2.cwiseProduct(P1.replicate(1, Dm))).sum() - Np * mu_y.transpose() * mu_y - 2 * (S * C).trace()) / (Np * Dm);
        }

        result.translation_t = mu_x - result.scale_s * result.rotation_R * mu_y;

        result.points = result.scale_s * ony * result.rotation_R.transpose() + (result.translation_t.transpose()).replicate(My, 1);

        result.iteration += 1;

        return;
    }
};

class NonrigidRegister : public Register<NonrigidResult>
{

private:

    Matrix affinity_G;

    void computeAffinityG(const Matrix& x, const Matrix& y, double beta, Matrix& G)
    {
        double k = -2 * beta * beta;
        int N = x.rows();
        int M = y.rows();
        int D = x.cols();
        assert(D == y.cols());

        G.setZero(N, M);
        for (int m = 0; m < M; ++m)
        {
            Matrix _diff = x - y.row(m).replicate(N, 1);
            Matrix diff = _diff.cwiseProduct(_diff);
            G.col(m) = (diff.rowwise().sum() / k).array().exp();
        }

        return;
    }

public:

    NonrigidRegister() :
        Register<NonrigidResult>() {}

    void init(const Matrix& x, const Matrix& y, Parameters& p)
    {
        Register<NonrigidResult>::init(x, y, p);
        result.transform_W = Matrix::Zero(My, Dm);
        computeAffinityG(ony, ony, parameter.nonrigid_beta, affinity_G);
        return;
    }

    void step()
    {
        Matrix& Px = result.probability_Px;
        Vector& P1 = result.probability_P1;
        Vector& Pt1 = result.probability_Pt1;
        double& L = result.L;

        double L_old = result.L;

        if (parameter.fgt_mode == 0)
        {
            call_cpd_p(onx, result.points, result.sigma2, parameter.outlier_weight, Px, P1, Pt1, &L);
        }
        else
        {
            call_cpd_p_fast(onx, result.points, result.sigma2, parameter.outlier_weight, parameter.sigma2, parameter.fgt_mode, Pt1, P1, Px, &L);
        }

        L = L + parameter.nonrigid_lambda / 2. * (result.transform_W.transpose() * affinity_G * result.transform_W).trace();

        result.n_tolerance = std::abs((L - L_old) / L);

        std::cout << "result: " << std::endl;
        std::cout << "iteration: " << result.iteration << std::endl;
        std::cout << "n_tolerance: " << result.n_tolerance << std::endl;
        std::cout << "sigma2: " << result.sigma2 << std::endl << std::endl;

        Matrix dP = Matrix::Identity(My, My);
        for (int m = 0; m < My; ++m)
        {
            dP(m, m) = P1(m, 0);
        }

        result.transform_W = (dP * affinity_G + parameter.nonrigid_lambda * result.sigma2 * Matrix::Identity(My, My)).inverse() * (Px - dP * ony);

        result.points = ony + affinity_G * result.transform_W;

        double Np = P1.sum();

        Matrix points2 = result.points.cwiseProduct(result.points);
        result.sigma2 = std::abs(((onx2.cwiseProduct(Pt1.replicate(1, Dm))).sum() + (points2.cwiseProduct(P1.replicate(1, Dm))).sum() - 2 * (Px.transpose() * result.points).trace()) / (Np * Dm));

        result.iteration += 1;

        return;
    }
};

class NonrigidLowrankRegister : public Register<NonrigidResult>
{

private:

    Matrix affinity_G;
    Matrix lowrank_Q, lowrank_S;
    Matrix lowrank_invS;

    void computeAffinityG(const Matrix& x, const Matrix& y, double beta, Matrix& G)
    {
        double k = -2 * beta * beta;
        int N = x.rows();
        int M = y.rows();
        int D = x.cols();
        assert(D == y.cols());

        G.setZero(N, M);
        for (int m = 0; m < M; ++m)
        {
            Matrix _diff = x - y.row(m).replicate(N, 1);
            Matrix diff = _diff.cwiseProduct(_diff);
            G.col(m) = (diff.rowwise().sum() / k).array().exp();
        }

        return;
    }

    void comouteLowrankQS(const Matrix& y, double beta, int lowrank_number_eigen, bool lowrank_eigen_fgt, Matrix& Q, Matrix& S)
    {
        if (!lowrank_eigen_fgt)
        {
            computeAffinityG(y, y, beta, affinity_G);
            Eigen::EigenSolver<Matrix> solver(affinity_G);
            const Matrix eigenValues = solver.eigenvalues().real();
            const Matrix eigenVectors = solver.eigenvectors().real();

            std::vector<std::pair<double, Vector>> eigenValuesVectors;
            for (int i = 0; i < eigenValues.rows(); ++i)
            {
                eigenValuesVectors.emplace_back(std::make_pair(eigenValues(i), eigenVectors.col(i)));
            }

            std::sort(eigenValuesVectors.begin(), eigenValuesVectors.end(), [](const std::pair<double, Vector>& a, const std::pair<double, Vector>& b) { return a.first > b.first; });
            eigenValuesVectors.resize(lowrank_number_eigen);

            S.setZero(lowrank_number_eigen, lowrank_number_eigen);
            Q.setZero(y.rows(), lowrank_number_eigen);

            for (int i = 0; i < lowrank_number_eigen; ++i)
            {
                S(i, i) = eigenValuesVectors[i].first;
                Q.col(i) = eigenValuesVectors[i].second;
            }

            return;
        }

        Matrix yt = y.transpose();
        int K = (int)std::round(min(std::sqrt(My), 100));
        int p = 6;
        double e = 8;
        double hsigma = 1.41421356 * beta;

        TransformMatrix op(yt, hsigma, e, K, p);
        Spectra::KrylovSchurEigsSolver<TransformMatrix> eigs(op, lowrank_number_eigen, min(lowrank_number_eigen * 2, yt.cols()));
        // Spectra::SymEigsSolver<TransformMatrix> eigs(op, lowrank_number_eigen, min(lowrank_number_eigen * 2, yt.cols()));

        eigs.init();
        int nconv = eigs.compute(Spectra::SortRule::LargestReal);
        std::cout << "Converged Eigen Values: " << nconv << std::endl;

        if (eigs.info() == Spectra::CompInfo::Successful)
        {
            S = eigs.eigenvalues().asDiagonal();
            Q = eigs.eigenvectors();
        }
 
        return;
    }

public:

    NonrigidLowrankRegister() :
        Register<NonrigidResult>() {}

    void init(const Matrix& x, const Matrix& y, Parameters& p)
    {
        Register<NonrigidResult>::init(x, y, p);
        result.transform_W = Matrix::Zero(My, Dm);
        comouteLowrankQS(ony, parameter.nonrigid_beta, parameter.nonrigid_lowrank_number_eigen, parameter.nonrigid_lowrank_eigen_fgt, lowrank_Q, lowrank_S);
        
        assert(lowrank_S.rows() != 0);
        assert(lowrank_S.rows() == lowrank_S.cols());

        lowrank_invS.setZero(lowrank_S.rows(), lowrank_S.cols());
        for (int d = 0; d < lowrank_S.rows(); ++d)
        {
            lowrank_invS(d, d) = 1 / lowrank_S(d, d);
        }

        return;
    }

    void step()
    {
        Matrix& Px = result.probability_Px;
        Vector& P1 = result.probability_P1;
        Vector& Pt1 = result.probability_Pt1;
        double& L = result.L;

        double L_old = result.L;

        Matrix QtW = lowrank_Q.transpose() * result.transform_W;

        if (parameter.fgt_mode == 0)
        {
            call_cpd_p(onx, result.points, result.sigma2, parameter.outlier_weight, Px, P1, Pt1, &L);
        }
        else
        {
            call_cpd_p_fast(onx, result.points, result.sigma2, parameter.outlier_weight, parameter.sigma2, parameter.fgt_mode, Pt1, P1, Px, &L);
        }

        L = L + parameter.nonrigid_lambda / 2. * (QtW.transpose() * lowrank_S * QtW).trace();

        result.n_tolerance = std::abs((L - L_old) / L);

        std::cout << "result: " << std::endl;
        std::cout << "iteration: " << result.iteration << std::endl;
        std::cout << "n_tolerance: " << result.n_tolerance << std::endl;
        std::cout << "sigma2: " << result.sigma2 << std::endl << std::endl;

        Matrix dP = Matrix::Identity(My, My);
        for (int m = 0; m < My; ++m)
        {
            dP(m, m) = P1(m, 0);
        }

        Matrix dPQ = dP * lowrank_Q;
        Matrix F = Px - dP * ony;

        Matrix temp = ((parameter.nonrigid_lambda * result.sigma2 * lowrank_invS + lowrank_Q.transpose() * dPQ).inverse()) * (lowrank_Q.transpose() * F);
        result.transform_W = 1 / (parameter.nonrigid_lambda * result.sigma2) * (F - dPQ * temp);

        result.points = ony + (lowrank_Q * (lowrank_S * (lowrank_Q.transpose() * result.transform_W)));

        double Np = P1.sum();

        Matrix points2 = result.points.cwiseProduct(result.points);
        result.sigma2 = std::abs((onx2.cwiseProduct(Pt1.replicate(1, Dm))).sum() + (points2.cwiseProduct(P1.replicate(1, Dm))).sum() - 2 * (Px.transpose() * result.points).trace()) / (Np * Dm);

        result.iteration += 1;

        return;
    }
};

bool ShapeRegistration::transform(const std::vector<cv::Point2d>& x, const std::vector<cv::Point2d>& y, std::vector<cv::Point2d>& ry, Parameters para)
{
    if (x.size() <= 0)
    {
        std::cout << "x is empty." << std::endl;
        return false;
    }

    if (y.size() <= 0)
    {
        std::cout << "y is empty." << std::endl;
        return false;
    }

    Matrix mx(x.size(), 2);
    Matrix my(y.size(), 2);

    for (size_t r = 0; r < x.size(); ++r)
    {
        mx(r, 0) = x[r].x;
        mx(r, 1) = x[r].y;
    }

    for (size_t r = 0; r < y.size(); ++r)
    {
        my(r, 0) = y[r].x;
        my(r, 1) = y[r].y;
    }

    if (para.method == 0)
    {
        AffineRegister affine;

        affine.init(mx, my, para);
        affine.run();

        ry.clear();
        ry.resize(affine.result.points.rows());
        for (int r = 0; r < affine.result.points.rows(); ++r)
        {
            ry[r].x = affine.result.points(r, 0);
            ry[r].y = affine.result.points(r, 1);
        }
    }
    else if (para.method == 1)
    {
        RigidRegister rigid;

        rigid.init(mx, my, para);
        rigid.run();

        ry.clear();
        ry.resize(rigid.result.points.rows());
        for (int r = 0; r < rigid.result.points.rows(); ++r)
        {
            ry[r].x = rigid.result.points(r, 0);
            ry[r].y = rigid.result.points(r, 1);
        }
    }
    else if (para.method == 2)
    {
        NonrigidRegister nonrigid;

        nonrigid.init(mx, my, para);
        nonrigid.run();

        ry.clear();
        ry.resize(nonrigid.result.points.rows());
        for (int r = 0; r < nonrigid.result.points.rows(); ++r)
        {
            ry[r].x = nonrigid.result.points(r, 0);
            ry[r].y = nonrigid.result.points(r, 1);
        }
    }
    else if (para.method == 3)
    {
        NonrigidLowrankRegister nonrigidlowrank;

        nonrigidlowrank.init(mx, my, para);
        nonrigidlowrank.run();

        ry.clear();
        ry.resize(nonrigidlowrank.result.points.rows());
        for (int r = 0; r < nonrigidlowrank.result.points.rows(); ++r)
        {
            ry[r].x = nonrigidlowrank.result.points(r, 0);
            ry[r].y = nonrigidlowrank.result.points(r, 1);
        }
    }

    return true;
}

bool ShapeRegistration::transform(const std::vector<cv::Point2d>& x, const std::vector<cv::Point2d>& y, std::vector<cv::Point2d>& ry, Parameters para, std::vector<int>& corr)
{
    if (x.size() <= 0)
    {
        std::cout << "x is empty." << std::endl;
        return false;
    }

    if (y.size() <= 0)
    {
        std::cout << "y is empty." << std::endl;
        return false;
    }

    Matrix mx(x.size(), 2);
    Matrix my(y.size(), 2);

    for (size_t r = 0; r < x.size(); ++r)
    {
        mx(r, 0) = x[r].x;
        mx(r, 1) = x[r].y;
    }

    for (size_t r = 0; r < y.size(); ++r)
    {
        my(r, 0) = y[r].x;
        my(r, 1) = y[r].y;
    }

    if (para.method == 0)
    {
        AffineRegister affine;

        affine.init(mx, my, para);
        affine.run();

        ry.clear();
        ry.resize(affine.result.points.rows());
        for (int r = 0; r < affine.result.points.rows(); ++r)
        {
            ry[r].x = affine.result.points(r, 0);
            ry[r].y = affine.result.points(r, 1);
        }

        affine.correspondence(corr);
    }
    else if (para.method == 1)
    {
        RigidRegister rigid;

        rigid.init(mx, my, para);
        rigid.run();

        ry.clear();
        ry.resize(rigid.result.points.rows());
        for (int r = 0; r < rigid.result.points.rows(); ++r)
        {
            ry[r].x = rigid.result.points(r, 0);
            ry[r].y = rigid.result.points(r, 1);
        }

        rigid.correspondence(corr);
    }
    else if (para.method == 2)
    {
        NonrigidRegister nonrigid;

        nonrigid.init(mx, my, para);
        nonrigid.run();

        ry.clear();
        ry.resize(nonrigid.result.points.rows());
        for (int r = 0; r < nonrigid.result.points.rows(); ++r)
        {
            ry[r].x = nonrigid.result.points(r, 0);
            ry[r].y = nonrigid.result.points(r, 1);
        }

        nonrigid.correspondence(corr);
    }
    else if (para.method == 3)
    {
        NonrigidLowrankRegister nonrigidlowrank;

        nonrigidlowrank.init(mx, my, para);
        nonrigidlowrank.run();

        ry.clear();
        ry.resize(nonrigidlowrank.result.points.rows());
        for (int r = 0; r < nonrigidlowrank.result.points.rows(); ++r)
        {
            ry[r].x = nonrigidlowrank.result.points(r, 0);
            ry[r].y = nonrigidlowrank.result.points(r, 1);
        }

        nonrigidlowrank.correspondence(corr);
    }

    return true;
}

bool ShapeRegistration::transform(const std::vector<cv::Point3d>& x, const std::vector<cv::Point3d>& y, std::vector<cv::Point3d>& ry, Parameters para)
{
    if (x.size() <= 0)
    {
        std::cout << "x is empty." << std::endl;
        return false;
    }

    if (y.size() <= 0)
    {
        std::cout << "y is empty." << std::endl;
        return false;
    }

    Matrix mx(x.size(), 3);
    Matrix my(y.size(), 3);

    for (size_t r = 0; r < x.size(); ++r)
    {
        mx(r, 0) = x[r].x;
        mx(r, 1) = x[r].y;
        mx(r, 2) = x[r].z;
    }

    for (size_t r = 0; r < y.size(); ++r)
    {
        my(r, 0) = y[r].x;
        my(r, 1) = y[r].y;
        my(r, 2) = y[r].z;
    }

    if (para.method == 0)
    {
        AffineRegister affine;

        affine.init(mx, my, para);
        affine.run();

        ry.clear();
        ry.resize(affine.result.points.rows());
        for (int r = 0; r < affine.result.points.rows(); ++r)
        {
            ry[r].x = affine.result.points(r, 0);
            ry[r].y = affine.result.points(r, 1);
            ry[r].z = affine.result.points(r, 2);
        }
    }
    else if (para.method == 1)
    {
        RigidRegister rigid;

        rigid.init(mx, my, para);
        rigid.run();

        ry.clear();
        ry.resize(rigid.result.points.rows());
        for (int r = 0; r < rigid.result.points.rows(); ++r)
        {
            ry[r].x = rigid.result.points(r, 0);
            ry[r].y = rigid.result.points(r, 1);
            ry[r].z = rigid.result.points(r, 2);
        }
    }
    else if (para.method == 2)
    {
        NonrigidRegister nonrigid;

        nonrigid.init(mx, my, para);
        nonrigid.run();

        ry.clear();
        ry.resize(nonrigid.result.points.rows());
        for (int r = 0; r < nonrigid.result.points.rows(); ++r)
        {
            ry[r].x = nonrigid.result.points(r, 0);
            ry[r].y = nonrigid.result.points(r, 1);
            ry[r].z = nonrigid.result.points(r, 2);
        }
    }
    else if (para.method == 3)
    {
        NonrigidLowrankRegister nonrigidlowrank;

        nonrigidlowrank.init(mx, my, para);
        nonrigidlowrank.run();

        ry.clear();
        ry.resize(nonrigidlowrank.result.points.rows());
        for (int r = 0; r < nonrigidlowrank.result.points.rows(); ++r)
        {
            ry[r].x = nonrigidlowrank.result.points(r, 0);
            ry[r].y = nonrigidlowrank.result.points(r, 1);
            ry[r].z = nonrigidlowrank.result.points(r, 2);
        }
    }

    return true;
}

bool ShapeRegistration::transform(const std::vector<cv::Point3d>& x, const std::vector<cv::Point3d>& y, std::vector<cv::Point3d>& ry, Parameters para, std::vector<int>& corr)
{
    if (x.size() <= 0)
    {
        std::cout << "x is empty." << std::endl;
        return false;
    }

    if (y.size() <= 0)
    {
        std::cout << "y is empty." << std::endl;
        return false;
    }

    Matrix mx(x.size(), 3);
    Matrix my(y.size(), 3);

    for (size_t r = 0; r < x.size(); ++r)
    {
        mx(r, 0) = x[r].x;
        mx(r, 1) = x[r].y;
        mx(r, 2) = x[r].z;
    }

    for (size_t r = 0; r < y.size(); ++r)
    {
        my(r, 0) = y[r].x;
        my(r, 1) = y[r].y;
        my(r, 2) = y[r].z;
    }

    if (para.method == 0)
    {
        AffineRegister affine;

        affine.init(mx, my, para);
        affine.run();

        ry.clear();
        ry.resize(affine.result.points.rows());
        for (int r = 0; r < affine.result.points.rows(); ++r)
        {
            ry[r].x = affine.result.points(r, 0);
            ry[r].y = affine.result.points(r, 1);
            ry[r].z = affine.result.points(r, 2);
        }

        affine.correspondence(corr);
    }
    else if (para.method == 1)
    {
        RigidRegister rigid;

        rigid.init(mx, my, para);
        rigid.run();

        ry.clear();
        ry.resize(rigid.result.points.rows());
        for (int r = 0; r < rigid.result.points.rows(); ++r)
        {
            ry[r].x = rigid.result.points(r, 0);
            ry[r].y = rigid.result.points(r, 1);
            ry[r].z = rigid.result.points(r, 2);
        }

        rigid.correspondence(corr);
    }
    else if (para.method == 2)
    {
        NonrigidRegister nonrigid;

        nonrigid.init(mx, my, para);
        nonrigid.run();

        ry.clear();
        ry.resize(nonrigid.result.points.rows());
        for (int r = 0; r < nonrigid.result.points.rows(); ++r)
        {
            ry[r].x = nonrigid.result.points(r, 0);
            ry[r].y = nonrigid.result.points(r, 1);
            ry[r].z = nonrigid.result.points(r, 2);
        }

        nonrigid.correspondence(corr);
    }
    else if (para.method == 3)
    {
        NonrigidLowrankRegister nonrigidlowrank;

        nonrigidlowrank.init(mx, my, para);
        nonrigidlowrank.run();

        ry.clear();
        ry.resize(nonrigidlowrank.result.points.rows());
        for (int r = 0; r < nonrigidlowrank.result.points.rows(); ++r)
        {
            ry[r].x = nonrigidlowrank.result.points(r, 0);
            ry[r].y = nonrigidlowrank.result.points(r, 1);
            ry[r].z = nonrigidlowrank.result.points(r, 2);
        }

        nonrigidlowrank.correspondence(corr);
    }

    return true;
}

ShapeRegistration::ShapeRegistration()
{
}

ShapeRegistration::~ShapeRegistration()
{
}
