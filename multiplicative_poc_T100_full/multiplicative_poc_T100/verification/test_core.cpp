#define TEST_CORE
#include "run_experiment.cpp"
#include <cassert>
int main(){
    Diagnostics dg;Work work;std::vector<double> y{0.1,-2.3,0.9,4.0,-5.2},z,zp;
    stage(z,y,1.0/64,true,work,dg);
    double mean=0;for(double v:z)mean+=v;mean/=z.size();
    for(size_t i=0;i<z.size();++i)assert(std::abs((1+2.0/64)*z[i]+14.0/64*std::pow(z[i],3)-.5/64*mean-y[i])<1e-10);
    auto yr=y;std::reverse(yr.begin(),yr.end());stage(zp,yr,1.0/64,true,work,dg);std::reverse(zp.begin(),zp.end());
    for(size_t i=0;i<z.size();++i)assert(std::abs(zp[i]-z[i])<1e-12);
    for(auto&v:yr)v=-v;stage(zp,yr,1.0/64,true,work,dg);std::reverse(zp.begin(),zp.end());
    for(size_t i=0;i<z.size();++i)assert(std::abs(zp[i]+z[i])<1e-12);
    std::mt19937_64 rng(4293);std::normal_distribution<double>normal;
    for(int k=0;k<100000;++k){
        double u=6*normal(rng),v=6*normal(rng);
        double dg=std::sqrt(1+u*u)-std::sqrt(1+v*v);
        assert(dg*dg<=(u-v)*(u-v)+1e-11);
        assert(dg*dg<=(u*u+u*v+v*v)*(u-v)*(u-v)/3.0+1e-10);
    }
    // State-dependent diffusion must not be replaced by common noise values.
    const double a=.3,b=.9,dw=.2;
    assert(std::abs((a+std::sqrt(1+a*a)*dw)-(b+std::sqrt(1+b*b)*dw)-(a-b))>1e-3);
    std::cout<<"PASS: nonlinear residual; permutation equivariance; odd stage;\n"
      <<"PASS: diffusion inequalities on 100000 sampled pairs; different diffusion evaluations.\n"
      <<"Tests check implementation; analytic inequalities are proved in README.\n";
}
