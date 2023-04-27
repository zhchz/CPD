#include "auxiliary.h"

#define max(A, B) ((A) > (B) ? (A) : (B))
#define min(A, B) ((A) < (B) ? (A) : (B))

// cpd aux //

void cpd_comp(const double* x, const double* y, const double* sigma2, const double* outlier, double* P1, double* Pt1, double* Px, double* E, int N, int M, int D)
{
    int n, m, d;
    double ksig, diff, razn, outlier_tmp, sp;
    double *P, *temp_x;

    P = (double*)calloc(M, sizeof(double));
    temp_x = (double*)calloc(D, sizeof(double));

    ksig = -2.0 * *sigma2;
    outlier_tmp = (*outlier * M * pow(-ksig * 3.14159265358979, 0.5 * D)) / ((1 - *outlier) * N);
    /* printf ("ksig = %lf\n", *sigma2);*/
    /* outlier_tmp=*outlier*N/(1- *outlier)/M*(-ksig*3.14159265358979); */

    for (n = 0; n < N; n++)
    {

        sp = 0;
        for (m = 0; m < M; m++)
        {
            razn = 0;
            for (d = 0; d < D; d++)
            {
                diff = *(x + n + d * N) - *(y + m + d * M);
                diff = diff * diff;
                razn += diff;
            }

            *(P + m) = exp(razn / ksig);
            sp += *(P + m);
        }

        sp += outlier_tmp;
        *(Pt1 + n) = 1 - outlier_tmp / sp;

        for (d = 0; d < D; d++)
        {
            *(temp_x + d) = *(x + n + d * N) / sp;
        }

        for (m = 0; m < M; m++)
        {

            *(P1 + m) += *(P + m) / sp;

            for (d = 0; d < D; d++)
            {
                *(Px + m + d * M) += *(temp_x + d) * *(P + m);
            }
        }

        *E += -log(sp);
    }
    *E += D * N * log(*sigma2) / 2;

    free((void*)P);
    free((void*)temp_x);

    return;
}

void cpd_comp_trunc(const double* x, const double* y, const double* sigma2, double* outlier, double* P1, double* Pt1, double* Px, double* E, int N, int M, int D, double* truncate)
{
    int n, m, d;
    double ksig, diff, razn, outlier_tmp, sp;
    double* P;

    P = (double*)calloc(M, sizeof(double));

    ksig = -2.0 * *sigma2;
    *truncate = log(*truncate);
    if (*outlier == 0)
    {
        *outlier = 1e-8;
    }

    outlier_tmp = (*outlier * M * pow(-ksig * 3.14159265358979, 0.5 * D)) / ((1 - *outlier) * N);
    /* printf ("ksig = %lf\n", *sigma2);*/
    /* outlier_tmp=*outlier*N/(1- *outlier)/M*(-ksig*3.14159265358979); */

    for (n = 0; n < N; n++)
    {
        sp = 0;

        for (m = 0; m < M; m++)
        {
            razn = 0;
            for (d = 0; d < D; d++)
            {
                diff = *(x + n + d * N) - *(y + m + d * M);
                diff = diff * diff;
                razn += diff;
            };

            razn = razn / ksig;

            if (razn < *truncate)
            {
                *(P + m) = 0;
            }
            else
            {
                *(P + m) = exp(razn);
                sp += *(P + m);
            };
        }

        sp += outlier_tmp;
        *(Pt1 + n) = 1 - outlier_tmp / sp;

        for (m = 0; m < M; m++)
        {

            if (*(P + m) == 0)
            {
            }
            else
            {
                *(P1 + m) += *(P + m) / sp;
                for (d = 0; d < D; d++)
                {
                    *(Px + m + d * M) += *(x + n + d * N) / sp * *(P + m);
                }
            }
        }

        *E += -log(sp);
    }
    *E += D * N * log(*sigma2) / 2;

    free((void*)P);

    return;
}

void cpd_correpondence(const double* x, const double* y, const double* sigma2, const double* outlier, int* Pc, int N, int M, int D)
{
    int n, m, d;
    double ksig, diff, razn, outlier_tmp, temp_x, sp;
    double *P, *P1;

    P = (double*)calloc(M, sizeof(double));
    P1 = (double*)calloc(M, sizeof(double));

    ksig = -2.0 * (*sigma2 + 1e-3);
    outlier_tmp = (*outlier * M * pow(-ksig * 3.14159265358979, 0.5 * D)) / ((1 - *outlier) * N);
    if (outlier_tmp == 0)
    {
        outlier_tmp = 1e-10;
    }

    /* printf ("ksig = %lf\n", *sigma2);*/

    for (n = 0; n < N; n++)
    {
        sp = 0;
        for (m = 0; m < M; m++)
        {
            razn = 0;
            for (d = 0; d < D; d++)
            {
                diff = *(x + n + d * N) - *(y + m + d * M);
                diff = diff * diff;
                razn += diff;
            }

            *(P + m) = exp(razn / ksig);
            sp += *(P + m);
        }

        sp += outlier_tmp;


        for (m = 0; m < M; m++)
        {

            temp_x = *(P + m) / sp;

            if (n == 0)
            {
                *(P1 + m) = *(P + m) / sp;
                *(Pc + m) = n;
            };

            if (temp_x > *(P1 + m))
            {
                *(P1 + m) = *(P + m) / sp;
                *(Pc + m) = n;
            }
        }
    }

    free((void*)P);
    free((void*)P1);

    return;
}

// fgt aux //

int fgt_nchoosek(int n, int k)
{
    int i, n_k = n - k, nchsk = 1;
    if (k < n_k)
    {
        k = n_k;
        n_k = n - k;
    }
    for (i = 1; i <= n_k; i++)
    {
        nchsk *= (++k);
        nchsk /= i;
    }
    return nchsk;
}

int fgt_idmax(const double* x, int N)
{
    int i, k = 0;
    double t = -1.0;

    for (i = 0; i < N; i++)
    {
        if (t < x[i])
        {
            t = x[i];
            k = i;
        }
    }
    return k;
}

double fgt_ddist(const double* x, const double* y, int d)
{
    int i;
    register double t, s = 0.0;
    for (i = 0; i < d; i++)
    {
        t = (x[i] - y[i]);
        s += (t * t);
    }
    return s;
}

void fgt_Compute_A_k(const double* x, const double* w, const double* xc, const double* C_k, double sigma, int d, int Nx, int p, int K, int pd, double* A_k, int* indx, double* dx, double* prods, int* heads)
{

    int n, i, k, t, tail, j, head, ind;
    int nbase, ix2c, ix2cbase;
    register double sum, ctesigma = 1.0 / (sigma), temp, temp1;

    for (i = 0; i < pd * K; i++)
    {
        A_k[i] = 0.0;
    }

    for (n = 0; n < Nx; n++)
    {
        nbase = n * d;
        ix2c = indx[n];
        ix2cbase = ix2c * d;
        ind = ix2c * pd;
        temp = w[n];
        sum = 0.0;
        for (i = 0; i < d; i++)
        {
            dx[i] = (x[i + nbase] - xc[i + ix2cbase]) * ctesigma;
            sum += dx[i] * dx[i];
            heads[i] = 0;
        }
        prods[0] = exp(-sum);

        for (k = 1, t = 1, tail = 1; k < p; k++, tail = t)
        {
            for (i = 0; i < d; i++)
            {
                head = heads[i];
                heads[i] = t;
                temp1 = dx[i];

                for (j = head; j < tail; j++, t++)
                {
                    prods[t] = temp1 * prods[j];
                }
            }
        }
        for (i = 0; i < pd; i++)
        {
            A_k[i + ind] += temp * prods[i];
        }
    }
    for (k = 0; k < K; k++)
    {
        ind = k * pd;
        for (i = 0; i < pd; i++)
        {
            A_k[i + ind] *= C_k[i];
        }
    }
}

void fgt_Compute_C_k(int d, int p, double* C_k, int* heads, int* cinds)
{
    int i, k, t, j, tail, head;

    for (i = 0; i < d; i++)
    {
        heads[i] = 0;
    }

    heads[d] = __INT32_MAX__;
    cinds[0] = 0;
    C_k[0] = 1.0;

    for (k = 1, t = 1, tail = 1; k < p; k++, tail = t)
    {
        for (i = 0; i < d; i++)
        {
            head = heads[i];
            heads[i] = t;

            for (j = head; j < tail; j++, t++)
            {
                cinds[t] = (j < heads[i + 1]) ? cinds[j] + 1 : 1;
                C_k[t] = 2.0 * C_k[j];
                C_k[t] /= (double)cinds[t];
            }
        }
    }
}

void fgt_Kcenter(const double* x, int d, int Nx, int K, double* xc, int* indxc, int* indx, int* xboxsz, double* dist_C)
{
    const double *x_ind, *x_j;
    register double temp;
    int i, j, ind, nd, ibase;

    // randomly pick one node as the first center.
    //	srand( (unsigned)time( NULL ) );
    //	ind      = rand() % Nx;

    ind = 1;
    *indxc++ = ind;
    x_j = x;
    x_ind = x + ind * d;

    for (j = 0; j < Nx; x_j += d, j++)
    {
        dist_C[j] = (j == ind) ? 0.0 : fgt_ddist(x_j, x_ind, d);
        indx[j] = 0;
    }
    for (i = 1; i < K; i++)
    {
        ind = fgt_idmax(dist_C, Nx);
        *indxc++ = ind;
        x_j = x;
        x_ind = x + ind * d;
        for (j = 0; j < Nx; x_j += d, j++)
        {
            temp = (j == ind) ? 0.0 : fgt_ddist(x_j, x_ind, d);
            if (temp < dist_C[j])
            {
                dist_C[j] = temp;
                indx[j] = i;
            }
        }
    }

    for (i = 0; i < K; i++)
    {
        xboxsz[i] = 0;
    }
    for (i = 0; i < d * K; i++)
    {
        xc[i] = 0.0;
    }

    for (i = 0, nd = 0; i < Nx; i++, nd += d)
    {
        xboxsz[indx[i]]++;
        ibase = indx[i] * d;
        for (j = 0; j < d; j++)
        {
            xc[j + ibase] += x[j + nd];
        }
    }

    for (i = 0, ibase = 0; i < K; i++, ibase += d)
    {
        temp = 1.0 / xboxsz[i];
        for (j = 0; j < d; j++)
        {
            xc[j + ibase] *= temp;
        }
    }
}

void fgt_model(const double* x, const double* w, double sigma, int p, int K, double e, double* xc, double* A_k, int d, int Nx, int* indxc, int* indx, int* xhead, int* xboxsz, double* dist_C, double* C_k, int* heads, int* cinds, double* dx, double* prods, int pd)
{
    fgt_Kcenter(x, d, Nx, K, xc, indxc, indx, xboxsz, dist_C);

    fgt_Compute_C_k(d, p, C_k, heads, cinds);

    fgt_Compute_A_k(x, w, xc, C_k, sigma, d, Nx, p, K, pd, A_k, indx, dx, prods, heads);
}

int fgt_invnchoosek(int d, int cnk)
{
    int i, cted = 1, ctep, cte, p;

    for (i = 2; i <= d; i++)
    {
        cted *= i;
    }

    cte = cnk * cted;
    p = 2;
    ctep = p;
    for (i = p + 1; i < p + d; i++)
    {
        ctep *= i;
    }
    while (ctep != cte)
    {
        ctep = ((p + d) * ctep) / p;
        p++;
    }
    return p;
}

void fgt_predict(const double* y, const double* xc, const double* A_k, int Ny, double sigma, int K, double e, int d, int pd, double* v, double* dy, double* prods, int* heads)
{
    int p, i, j, m, k, t, tail, mbase, kn, xbase, head, ind;
    double sum2, ctesigma = 1.0 / (sigma), temp, temp1;

    p = fgt_invnchoosek(d, pd);

    for (m = 0; m < Ny; m++)
    {
        temp = 0.0;
        mbase = m * d;
        for (kn = 0; kn < K; kn++)
        {
            xbase = kn * d;
            ind = kn * pd;
            sum2 = 0.0;
            for (i = 0; i < d; i++)
            {
                dy[i] = (y[i + mbase] - xc[i + xbase]) * ctesigma;
                sum2 += dy[i] * dy[i];
                heads[i] = 0;
            }
            if (sum2 > e)
            {
                continue;  //skip to next kn
            }

            prods[0] = exp(-sum2);

            for (k = 1, t = 1, tail = 1; k < p; k++, tail = t)
            {
                for (i = 0; i < d; i++)
                {
                    head = heads[i];
                    heads[i] = t;
                    temp1 = dy[i];
                    for (j = head; j < tail; j++, t++)
                    {
                        prods[t] = temp1 * prods[j];
                    }
                }
            }

            for (i = 0; i < pd; i++)
            {
                temp += A_k[i + ind] * prods[i];
            }
        }
        v[m] = temp;
    }
}
