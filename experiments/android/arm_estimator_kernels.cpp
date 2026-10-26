// Inference kernels for unchanged exported tree and binary RBF-SVC parameters.
#include <cmath>
#include <cstdint>
#include <algorithm>

extern "C" void forest_probability(const float* x, int rows, int columns,
    const int64_t* roots, int trees, const int64_t* left, const int64_t* right,
    const int64_t* feature, const double* threshold, const double* probability,
    const uint8_t* missing_left, double* output) {
    for (int r=0;r<rows;r++) {
        double sum=0;
        for (int t=0;t<trees;t++) {
            int64_t node=roots[t];
            while (left[node]>=0) {
                float value=x[r*columns+feature[node]];
                bool take_left=std::isnan(value) ? missing_left[node]!=0 : value<=threshold[node];
                node=take_left ? left[node] : right[node];
            }
            sum+=probability[node];
        }
        output[r]=sum/trees;
    }
}

static double coupled_binary_probability(double decision, double A, double B) {
    // libsvm sigmoid calibration followed by its binary pairwise coupling.
    double f=decision*A+B;
    double p0=f>=0 ? std::exp(-f)/(1+std::exp(-f)) : 1/(1+std::exp(f));
    p0=std::min(std::max(p0,1e-7),1-1e-7);
    double r01=p0, r10=1-p0;
    double Q[2][2]={{r10*r10,-r10*r01},{-r10*r01,r01*r01}};
    double p[2]={0.5,0.5}, Qp[2];
    for (int iter=0;iter<100;iter++) {
        double pQp=0;
        for(int t=0;t<2;t++) {
            Qp[t]=0;
            for(int j=0;j<2;j++) Qp[t]+=Q[t][j]*p[j];
            pQp+=p[t]*Qp[t];
        }
        if(std::max(std::abs(Qp[0]-pQp),std::abs(Qp[1]-pQp))<0.0025) break;
        for(int t=0;t<2;t++) {
            double diff=(-Qp[t]+pQp)/Q[t][t];
            p[t]+=diff;
            pQp=(pQp+diff*(diff*Q[t][t]+2*Qp[t]))/((1+diff)*(1+diff));
            for(int j=0;j<2;j++) {
                Qp[j]=(Qp[j]+diff*Q[t][j])/(1+diff);
                p[j]/=(1+diff);
            }
        }
    }
    return p[1];
}

extern "C" void svc_probability(const double* x, int rows, int columns,
    const double* support, int vectors, const double* dual, double intercept,
    double gamma, double A, double B, double* output) {
    for(int r=0;r<rows;r++) {
        double value=intercept;
        for(int v=0;v<vectors;v++) {
            double distance=0;
            for(int c=0;c<columns;c++) {
                double diff=x[r*columns+c]-support[v*columns+c];
                distance+=diff*diff;
            }
            value+=dual[v]*std::exp(-gamma*distance);
        }
        output[r]=coupled_binary_probability(value,A,B);
    }
}
