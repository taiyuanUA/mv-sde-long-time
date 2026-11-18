#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

// Exact model used in the current experiment:
// b_l(x,mu) = -2*x_l - 14*x_l^3 + 0.5*mean_l(mu),
// sigma(x)  = diag(sqrt(1+x_l^2)).
// All coordinates of all particles are simulated; no mean-centering,
// equilibrium reset, time skipping, or replacement of an observed error.
constexpr int D = 8;
constexpr double TOL = 1e-12;
using Paths = std::array<std::vector<double>,D>;
struct Diagnostics { int max_iter=0; double max_res=0, max_abs=0; };
struct Work { std::vector<double> invD,g; void resize(int n) {invD.resize(n);g.resize(n);} };

void stage(std::vector<double>& z,const std::vector<double>& y,double h,
           bool interacting,Work& w,Diagnostics& diag) {
    const int N=static_cast<int>(y.size());
    const double a=1.0+2.0*h, b=14.0*h, c=interacting ? 0.5*h : 0.0;
    z=y; w.resize(N);
    double scale=1.0;
    for(double x:y) scale=std::max(scale,1.0+std::abs(x));
    for(int it=0;it<=20;++it) {
        double sumz=0; if(interacting) for(double x:z)sumz+=x;
        const double mz=sumz/N;
        double res=0,sumg=0,sumv=0;
        for(int i=0;i<N;++i) {
            const double zz=z[i]*z[i];
            const double f=(a+b*zz)*z[i]-c*mz-y[i];
            if(!std::isfinite(f)) throw std::runtime_error("Nonfinite implicit residual");
            res=std::max(res,std::abs(f));
            w.invD[i]=1.0/(a+3.0*b*zz);
            w.g[i]=w.invD[i]*f;
            sumg+=w.g[i]; sumv+=w.invD[i];
        }
        if(res<=TOL*scale) {
            diag.max_iter=std::max(diag.max_iter,it);
            diag.max_res=std::max(diag.max_res,res/scale);
            return;
        }
        const double den=1.0-c*sumv/N;
        if(!(den>0))throw std::runtime_error("Invalid Newton denominator");
        const double shift=c*(sumg/N)/den;
        for(int i=0;i<N;++i) z[i]-=w.g[i]+w.invD[i]*shift;
    }
    throw std::runtime_error("Implicit solve did not converge");
}

std::array<double,5> errors(const Paths& y,const Paths& x) {
    std::array<double,5> ans{};double cumul=0;int p=0;
    const size_t N=y[0].size();
    for(int l=0;l<D;++l) {
        double ss=0;for(size_t i=0;i<N;++i){double e=y[l][i]-x[l][i];ss+=e*e;}
        cumul+=ss/N;
        if(l==1||l==2||l==3||l==4||l==7)ans[p++]=cumul;
    }
    return ans;
}
std::array<double,5> error_difference(const Paths& yc,const Paths& xc,
                                      const Paths& yf,const Paths& xf) {
    std::array<double,5> ans{};double cumul=0;int p=0;
    const size_t N=yc[0].size();
    for(int l=0;l<D;++l) {
        double ss=0;for(size_t i=0;i<N;++i){double e=(yc[l][i]-xc[l][i])-(yf[l][i]-xf[l][i]);ss+=e*e;}
        cumul+=ss/N;
        if(l==1||l==2||l==3||l==4||l==7)ans[p++]=cumul;
    }
    return ans;
}
void paired_step(Paths& y,Paths& x,const Paths& dW,double h,Work& work,Diagnostics& dg) {
    std::vector<double> zy,zx;
    for(int l=0;l<D;++l) {
        stage(zy,y[l],h,true,work,dg);stage(zx,x[l],h,false,work,dg);
        for(size_t i=0;i<zy.size();++i) {
            y[l][i]=zy[i]+std::sqrt(1.0+zy[i]*zy[i])*dW[l][i];
            x[l][i]=zx[i]+std::sqrt(1.0+zx[i]*zx[i])*dW[l][i];
            if(!std::isfinite(y[l][i])||!std::isfinite(x[l][i]))throw std::runtime_error("Nonfinite state");
            dg.max_abs=std::max({dg.max_abs,std::abs(y[l][i]),std::abs(x[l][i])});
        }
    }
}
struct Job{int N,rep;};
struct Result{std::vector<std::array<double,5>> coarse,fine,difference;Diagnostics dg;std::string error;};

#ifndef TEST_CORE
int main(int argc,char**argv) {
    int R=64,oneN=0;double T=100,h=1.0/64.0,save=0.25;bool refine=false;
    std::string out="data";
    for(int a=1;a<argc;++a) {
        const std::string arg=argv[a];
        if(arg=="--out")out=argv[++a];
        else if(arg=="--R")R=std::stoi(argv[++a]);
        else if(arg=="--N")oneN=std::stoi(argv[++a]);
        else if(arg=="--T")T=std::stod(argv[++a]);
        else if(arg=="--h")h=std::stod(argv[++a]);
        else if(arg=="--refine")refine=true;
        else throw std::runtime_error("Unknown argument: "+arg);
    }
    if(R<1||T<=0||h<=0)throw std::runtime_error("Invalid parameters");
    if(refine&&oneN==0)oneN=2560;
    std::vector<int> Ns=oneN?std::vector<int>{oneN}:std::vector<int>{40,80,160,320,640,1280,2560};
    int steps=std::lround(T/h),stride=std::lround(save/h);
    if(stride<1||std::abs(steps*h-T)>1e-10||std::abs(stride*h-save)>1e-10||steps%stride!=0)throw std::runtime_error("Invalid grid");
    const int nt=steps/stride+1;
    std::vector<Job>jobs;for(int n:Ns)for(int rep=0;rep<R;++rep)jobs.push_back({n,rep});
    // Largest jobs first for balanced parallel execution. Output order is restored below.
    std::vector<int>order(jobs.size());for(int k=0;k<(int)jobs.size();++k)order[k]=k;
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){return jobs[a].N>jobs[b].N;});
    std::vector<Result> result(jobs.size());std::atomic<int> completed{0};
    const auto start=std::chrono::steady_clock::now();
    #pragma omp parallel for schedule(dynamic,1)
    for(int jj=0;jj<(int)jobs.size();++jj) {
        int job=order[jj],N=jobs[job].N,rep=jobs[job].rep;Result& ans=result[job];
        try {
            // New streams for this model, independent across full systems and N.
            uint64_t seed=(refine?520260930ULL:20260930ULL)+1000003ULL*N+7919ULL*(rep+1);
            std::mt19937_64 rng(seed);std::normal_distribution<double>normal;
            Paths y,x,yf,xf,w,w1,w2;
            for(int l=0;l<D;++l){y[l].resize(N);x[l].resize(N);w[l].resize(N);
                for(int i=0;i<N;++i){y[l][i]=normal(rng);x[l][i]=y[l][i];ans.dg.max_abs=std::max(ans.dg.max_abs,std::abs(y[l][i]));}
                if(refine){yf[l]=y[l];xf[l]=x[l];w1[l].resize(N);w2[l].resize(N);}
            }
            Work work;work.resize(N);
            ans.coarse.resize(nt);ans.coarse[0].fill(0);
            if(refine){ans.fine.resize(nt);ans.difference.resize(nt);ans.fine[0].fill(0);ans.difference[0].fill(0);}
            int isave=1;
            for(int k=1;k<=steps;++k){
                if(refine){
                    for(int l=0;l<D;++l)for(int i=0;i<N;++i){w1[l][i]=std::sqrt(h/2)*normal(rng);w2[l][i]=std::sqrt(h/2)*normal(rng);w[l][i]=w1[l][i]+w2[l][i];}
                    paired_step(y,x,w,h,work,ans.dg);
                    paired_step(yf,xf,w1,h/2,work,ans.dg);
                    paired_step(yf,xf,w2,h/2,work,ans.dg);
                }else{
                    for(int l=0;l<D;++l)for(int i=0;i<N;++i)w[l][i]=std::sqrt(h)*normal(rng);
                    paired_step(y,x,w,h,work,ans.dg);
                }
                if(k%stride==0){ans.coarse[isave]=errors(y,x);
                    if(refine){ans.fine[isave]=errors(yf,xf);ans.difference[isave]=error_difference(y,x,yf,xf);}
                    ++isave;
                }
            }
        }catch(const std::exception&e){ans.error=e.what();}
        const int done=++completed;
        if(done%8==0||done==(int)jobs.size()){
            #pragma omp critical
            std::cerr<<done<<"/"<<jobs.size()<<" completed; elapsed="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"s\n";
        }
    }
    for(size_t k=0;k<result.size();++k)if(!result[k].error.empty())throw std::runtime_error("N="+std::to_string(jobs[k].N)+": "+result[k].error);
    std::ofstream f(out+(refine?"/raw_refinement.csv":"/raw_errors.csv"));
    if(!f)throw std::runtime_error("Cannot open output file");
    f<<"N,rep,t,";
    if(!refine)f<<"mse_d2,mse_d3,mse_d4,mse_d5,mse_d8\n";
    else f<<"c_d2,c_d3,c_d4,c_d5,c_d8,f_d2,f_d3,f_d4,f_d5,f_d8,diff_d2,diff_d3,diff_d4,diff_d5,diff_d8\n";
    f<<std::setprecision(17);
    std::ofstream dg(out+(refine?"/refinement_solver_diagnostics.csv":"/solver_diagnostics.csv"));
    dg<<"N,rep,max_newton_updates,max_scaled_residual,max_abs_state\n"<<std::setprecision(17);
    for(size_t k=0;k<jobs.size();++k){auto j=jobs[k];auto&a=result[k];
        dg<<j.N<<","<<j.rep+1<<","<<a.dg.max_iter<<","<<a.dg.max_res<<","<<a.dg.max_abs<<"\n";
        for(int q=0;q<nt;++q){f<<j.N<<","<<j.rep+1<<","<<q*stride*h;
            for(double v:a.coarse[q])f<<","<<v;
            if(refine){for(double v:a.fine[q])f<<","<<v;for(double v:a.difference[q])f<<","<<v;}
            f<<"\n";
        }
    }
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::ofstream meta(out+(refine?"/refinement_metadata.txt":"/run_metadata.txt"));
    meta<<std::setprecision(17)<<"Model: b=-2x-14x^[3]+0.5m(mu), sigma=diag(sqrt(1+x_l^2))\n"
        <<"Initial: iid N(0,I_8); no empirical centering. Dimension prefixes 2,3,4,5,8.\n"
        <<"T="<<T<<"\nh="<<h<<"\nR="<<R<<"\nsave_dt="<<save<<"\n"
        <<"N=";for(int n:Ns)meta<<n<<",";
    meta<<"\nMethod: fully coupled implicit-drift SSBE, separately evaluated multiplicative diffusion.\n"
        <<"Reference: noninteracting zero-mean limit equation discretized at the SAME h.\n"
        <<"No noise cancellation assumption; only Brownian increments are shared.\n"
        <<"RNG: mt19937_64, seed="<<(refine?520260930ULL:20260930ULL)<<"+1000003*N+7919*(rep+1)\n"
        <<"Runtime_seconds="<<seconds<<"\n";
    std::cerr<<"DONE; wall_seconds="<<seconds<<"\n";
}
#endif
