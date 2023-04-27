// Author: Zhenhui Zhou <zhouzhenhui_68@outlook.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.


#ifndef SPECTRA_KRYLOVSCHUR_EIGS_SOLVER_H
#define SPECTRA_KRYLOVSCHUR_EIGS_SOLVER_H

#include "KrylovSchurEigsBase.h"
#include "Util/SelectionRule.h"
#include "MatOp/DenseSymMatProd.h"


namespace Spectra {

///
/// \defgroup KrylovSchur Generalized Eigen Solvers
///
/// Generalized eigen solvers for different types of problems.
///

///
/// \ingroup KrylovSchur
///
/// This class implements the generalized eigen solver for real symmetric
/// matrices, i.e., to solve \f$Ax=\lambda Bx\f$ where \f$A\f$ is symmetric and
/// \f$B\f$ is positive definite.
///
/// There are two modes of this solver, specified by the template parameter `Mode`.
/// See the pages for the specialized classes for details.
/// - The Cholesky mode assumes that \f$B\f$ can be factorized using Cholesky
///   decomposition, which is the preferred mode when the decomposition is
///   available. (This can be easily done in Eigen using the dense or sparse
///   Cholesky solver.)
///   See \ref KrylovSchurEigsSolver<OpType, BOpType, GEigsMode::Cholesky> "KrylovSchurEigsSolver (Cholesky mode)" for more details.
/// - The regular inverse mode requires the matrix-vector product \f$Bv\f$ and the
///   linear equation solving operation \f$B^{-1}v\f$. This mode should only be
///   used when the Cholesky decomposition of \f$B\f$ is hard to implement, or
///   when computing \f$B^{-1}v\f$ is much faster than the Cholesky decomposition.
///   See \ref KrylovSchurEigsSolver<OpType, BOpType, GEigsMode::RegularInverse> "KrylovSchurEigsSolver (Regular inverse mode)" for more details.

template <typename OpType = DenseSymMatProd<double>>
class KrylovSchurEigsSolver :
    public KrylovSchurEigsBase<OpType, IdentityBOp>
{
private:
    using Scalar = typename OpType::Scalar;
    using Index = Eigen::Index;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

public:
    ///
    /// Constructor to create a solver object.
    ///
    /// \param op   The \f$A\f$ matrix operation object that implements the matrix-vector
    ///             multiplication operation of \f$A\f$:
    ///             calculating \f$Av\f$ for any vector \f$v\f$. Users could either
    ///             create the object from the wrapper classes such as DenseSymMatProd, or
    ///             define their own that implements all the public members
    ///             as in DenseSymMatProd.
    /// \param nev  Number of eigenvalues requested. This should satisfy \f$1\le nev \le n-1\f$,
    ///             where \f$n\f$ is the size of matrix.
    /// \param ncv  Parameter that controls the convergence speed of the algorithm.
    ///             Typically a larger `ncv` means faster convergence, but it may
    ///             also result in greater memory use and more matrix operations
    ///             in each iteration. This parameter must satisfy \f$nev < ncv \le n\f$,
    ///             and is advised to take \f$ncv \ge 2\cdot nev\f$.
    ///
    KrylovSchurEigsSolver(OpType& op, Index nev, Index ncv) :
        KrylovSchurEigsBase<OpType, IdentityBOp>(op, IdentityBOp(), nev, ncv)
    {}
};

}  // namespace Spectra

#endif  // SPECTRA_KRYLOVSCHUR_GEIGS_SOLVER_H
