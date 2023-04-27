#pragma once

#include <stdlib.h>
#include <math.h>

void cpd_comp(const double* x, const double* y, const double* sigma2, const double* outlier, double* P1, double* Pt1, double* Px, double* E, int N, int M, int D);
void cpd_comp_trunc(const double* x, const double* y, const double* sigma2, double* outlier, double* P1, double* Pt1, double* Px, double* E, int N, int M, int D, double* truncate);
void cpd_correpondence(const double* x, const double* y, const double* sigma2, const double* outlier, int* Pc, int N, int M, int D);

int fgt_nchoosek(int n, int k);
void fgt_model(const double* x, const double* w, double sigma, int p, int K, double e, double* xc, double* A_k, int d, int Nx, int* indxc, int* indx, int* xhead, int* xboxsz, double* dist_C, double* C_k, int* heads, int* cinds, double* dx, double* prods, int pd);
void fgt_predict(const double* y, const double* xc, const double* A_k, int Ny, double sigma, int K, double e, int d, int pd, double* v, double* dy, double* prods, int* heads);
