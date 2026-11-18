function out = multiplicative_poc_T100(action, folder, R)
% MULTIPLICATIVE_POC_T100  PoC of a cubic MV-SDE with multiplicative noise.
%
% Model (coordinate ell):
%   dX_ell=(-2X_ell-14X_ell^3+0.5*E[X_ell])dt
%           +sqrt(1+X_ell^2)dW_ell,    X0~N(0,I_d).
% The limiting mean is zero by symmetry, NOT by empirical centering.
% Both the IPS and noninteracting reference use fully coupled SSBE.
% Their Brownian increments agree, but their diffusion coefficients differ.
%
%   multiplicative_poc_T100('plot')
%   multiplicative_poc_T100('simulate','my_results',64)
%   multiplicative_poc_T100('refine','my_refinement',32)
%   multiplicative_poc_T100('analyze','my_results')
%   multiplicative_poc_T100('test')
%
% The distributed result tables were actually computed by the accompanying
% C++ implementation. This MATLAB implementation has not been executed by
% the authoring environment (no MATLAB/Octave there). MATLAB's and C++'s
% random-number generators differ, so new runs need not match bit for bit.
% No statistics toolbox is required. Parallelization is not needed.

if nargin<1 || isempty(action), action='plot'; end
root=fileparts(fileparts(mfilename('fullpath')));
if nargin<2 || isempty(folder)
    if any(strcmpi(action,{'plot','analyze','test'}))
        folder=root;
    else
        folder=fullfile(root,['matlab_' lower(action) '_results']);
    end
end
if nargin<3 || isempty(R)
    if strcmpi(action,'refine'), R=32; else, R=64; end
end
validateattributes(R,{'numeric'},{'scalar','integer','>=',2});
cfg=struct('T',100,'h',1/64,'save_dt',0.25,'D',8, ...
    'dims',[2,3,4,5,8],'Ns',[40,80,160,320,640,1280,2560], ...
    'R',R,'tol',1e-12,'bootstrap',5000);
if ~exist(folder,'dir'), mkdir(folder); end
datafolder=fullfile(folder,'data');
if ~exist(datafolder,'dir'), mkdir(datafolder); end
switch lower(action)
    case 'plot'
        S=readtable(fullfile(datafolder,'summary.csv'));
        out=plot_both(S,folder);
    case 'simulate'
        simulate_all(cfg,datafolder);
        out=analyze_main(cfg,datafolder);
        plot_both(out,folder);
    case 'analyze'
        if isfile(fullfile(datafolder,'raw_errors.csv'))
            out=analyze_main(cfg,datafolder);
            plot_both(out,folder);
        else
            out=[];
        end
        if isfile(fullfile(datafolder,'raw_refinement.csv'))
            analyze_refinement(cfg,datafolder);
        end
    case 'refine'
        refine_all(cfg,datafolder);
        out=analyze_refinement(cfg,datafolder);
    case 'test'
        test_core();out=true;
    otherwise
        error('Action must be plot, simulate, analyze, refine, or test.');
end
end

function simulate_all(cfg,folder)
steps=round(cfg.T/cfg.h);stride=round(cfg.save_dt/cfg.h);
assert(abs(steps*cfg.h-cfg.T)<1e-12 && mod(steps,stride)==0);
times=(0:stride:steps)'*cfg.h;nt=numel(times);
raw=zeros(numel(cfg.Ns)*cfg.R*nt,8);
diag=zeros(numel(cfg.Ns)*cfg.R,5);jr=0;row=0;
for N=cfg.Ns
    for rep=1:cfg.R
        rng(mod(20260930+1000003*N+7919*rep,2^32-1),'twister');
        Y=randn(N,cfg.D);X=Y;
        values=zeros(nt,numel(cfg.dims));isave=2;
        maxit=0;maxres=0;maxstate=max(abs(Y(:)));
        for n=1:steps
            dW=sqrt(cfg.h)*randn(N,cfg.D);
            [Y,X,it,res]=paired_step(Y,X,dW,cfg.h,cfg.tol);
            maxit=max(maxit,it);maxres=max(maxres,res);
            maxstate=max([maxstate;abs(Y(:));abs(X(:))]);
            if mod(n,stride)==0
                values(isave,:)=error_by_dimension(Y,X,cfg.dims);
                isave=isave+1;
            end
        end
        ii=row+(1:nt);
        raw(ii,:)=[N*ones(nt,1),rep*ones(nt,1),times,values];row=row+nt;
        jr=jr+1;diag(jr,:)=[N,rep,maxit,maxres,maxstate];
        fprintf('N=%d, independent system %d/%d finished\n',N,rep,cfg.R);
    end
end
writetable(array2table(raw,'VariableNames', ...
 {'N','rep','t','mse_d2','mse_d3','mse_d4','mse_d5','mse_d8'}), ...
 fullfile(folder,'raw_errors.csv'));
writetable(array2table(diag,'VariableNames', ...
 {'N','rep','max_newton_updates','max_scaled_residual','max_abs_state'}), ...
 fullfile(folder,'solver_diagnostics.csv'));
save(fullfile(folder,'settings.mat'),'cfg');
end

function [Ynew,Xnew,it,res]=paired_step(Y,X,dW,h,tol)
[Z,it1,res1]=implicit_stage(Y,h,true,tol);
[ZX,it2,res2]=implicit_stage(X,h,false,tol);
% IMPORTANT: only dW is shared. Evaluate diffusion at EACH intermediate state.
Ynew=Z+sqrt(1+Z.^2).*dW;
Xnew=ZX+sqrt(1+ZX.^2).*dW;
assert(all(isfinite(Ynew(:))) && all(isfinite(Xnew(:))), ...
    'Nonfinite state: the simulation was stopped.');
it=max(it1,it2);res=max(res1,res2);
end

function [Z,it,res]=implicit_stage(Y,h,interacting,tol)
% (1+2h)Z_i + 14h Z_i.^3 - (h/2)mean(Z) = Y_i.
% Each coordinate Jacobian is diagonal minus rank one. It is solved exactly
% within Newton by the Sherman--Morrison formula. No frozen previous mean.
a=1+2*h;b=14*h;c=0.5*h*double(interacting);Z=Y;
scale=1+max(abs(Y),[],1);
for it=0:20
    F=(a+b*Z.^2).*Z-Y-c*mean(Z,1);
    res=max(max(abs(F),[],1)./scale);
    if ~isfinite(res), error('Nonfinite implicit residual.'); end
    if res<=tol, return; end
    invD=1./(a+3*b*Z.^2);G=invD.*F;
    denominator=1-c*mean(invD,1);
    assert(all(denominator>0),'Invalid Newton denominator.');
    correction=c*mean(G,1)./denominator;
    Z=Z-G-invD.*correction;
end
error('Implicit solve failed, scaled residual=%g.',res);
end

function e=error_by_dimension(Y,X,dims)
% Coordinate prefixes are legitimate here: the drift and diffusion act
% coordinatewise (each mean is the mean of that coordinate only).
cumulative=cumsum(mean((Y-X).^2,1));e=cumulative(dims);
end

function S=analyze_main(cfg,folder)
A=readtable(fullfile(folder,'raw_errors.csv'));
Ns=unique(A.N);times=unique(A.t);reps=unique(A.rep);
R=numel(reps);nt=numel(times);D=numel(cfg.dims);nN=numel(Ns);
terminal=zeros(R,nN,D);late=zeros(R,nN,D);gridmax=zeros(nN,D);
rows=zeros(nN*D,9);row=0;meanrows=zeros(nN*nt,7);
for ni=1:nN
    block=sortrows(A(A.N==Ns(ni),:),{'rep','t'});
    V=permute(reshape(block{:,4:8},nt,R,D),[2,1,3]);
    assert(all(isfinite(V(:))) && all(V(:)>=0),'Invalid observed errors.');
    terminal(:,ni,:)=reshape(V(:,end,:),R,1,D);
    late(:,ni,:)=reshape(mean(V(:,times>=80 & times<=100,:),2),R,1,D);
    meanV=reshape(mean(V,1),nt,D);gridmax(ni,:)=max(meanV,[],1);
    ii=(ni-1)*nt+(1:nt);meanrows(ii,:)=[Ns(ni)*ones(nt,1),times,meanV];
    for q=1:D
        x=terminal(:,ni,q);l=late(:,ni,q);mu=mean(x);se=std(x)/sqrt(R);
        row=row+1;rows(row,:)=[cfg.dims(q),Ns(ni),mu,se,max(0,mu-1.96*se), ...
            mu+1.96*se,mean(l),std(l)/sqrt(R),gridmax(ni,q)];
    end
end
S=array2table(rows,'VariableNames',{'d','N','mse_T','se_T','ci_low_T', ...
    'ci_high_T','mse_late','se_late','max_grid_mean'});
writetable(S,fullfile(folder,'summary.csv'));
writetable(array2table(meanrows,'VariableNames', ...
 {'N','t','mse_d2','mse_d3','mse_d4','mse_d5','mse_d8'}), ...
 fullfile(folder,'time_mean.csv'));
% Whole-system bootstrap: do not resample particles as independent units.
rng(703092026,'twister');B=cfg.bootstrap;
bootT=zeros(B,nN,D);bootL=zeros(B,nN,D);
for ni=1:nN
    ind=randi(R,B,R);
    for q=1:D
        x=terminal(:,ni,q);l=late(:,ni,q);
        bootT(:,ni,q)=mean(x(ind),2);bootL(:,ni,q)=mean(l(ind),2);
    end
end
lx=log(Ns(:));weight=(lx-mean(lx))/sum((lx-mean(lx)).^2);
rates=zeros(D,8);
for q=1:D
    fitT=polyfit(lx,log(mean(terminal(:,:,q),1))',1);
    fitL=polyfit(lx,log(mean(late(:,:,q),1))',1);
    fitG=polyfit(lx,log(gridmax(:,q)),1);
    bt=log(bootT(:,:,q))*weight;bl=log(bootL(:,:,q))*weight;
    rates(q,:)=[cfg.dims(q),fitT(1),percentiles(bt),fitL(1),percentiles(bl),fitG(1)];
end
writetable(array2table(rates,'VariableNames', ...
 {'d','slope_T','ci_low_T','ci_high_T','slope_late','ci_low_late','ci_high_late','slope_max_grid'}), ...
 fullfile(folder,'rates.csv'));
disp(array2table(rates(:,1:4),'VariableNames',{'d','slope_T','ci_low','ci_high'}));
end

function refine_all(cfg,folder)
N=2560;h=cfg.h;steps=round(cfg.T/h);stride=round(cfg.save_dt/h);
times=(0:stride:steps)'*h;nt=numel(times);raw=zeros(cfg.R*nt,18);
for rep=1:cfg.R
    rng(mod(520260930+1000003*N+7919*rep,2^32-1),'twister');
    Yc=randn(N,cfg.D);Xc=Yc;Yf=Yc;Xf=Yc;
    V=zeros(nt,15);isave=2;
    for n=1:steps
        w1=sqrt(h/2)*randn(N,cfg.D);w2=sqrt(h/2)*randn(N,cfg.D);
        [Yc,Xc]=paired_step(Yc,Xc,w1+w2,h,cfg.tol);
        [Yf,Xf]=paired_step(Yf,Xf,w1,h/2,cfg.tol);
        [Yf,Xf]=paired_step(Yf,Xf,w2,h/2,cfg.tol);
        if mod(n,stride)==0
            V(isave,:)=[error_by_dimension(Yc,Xc,cfg.dims), ...
                error_by_dimension(Yf,Xf,cfg.dims), ...
                error_by_dimension(Yc-Xc,Yf-Xf,cfg.dims)];
            isave=isave+1;
        end
    end
    ii=(rep-1)*nt+(1:nt);raw(ii,:)=[N*ones(nt,1),rep*ones(nt,1),times,V];
    fprintf('Refinement independent system %d/%d finished\n',rep,cfg.R);
end
writetable(array2table(raw,'VariableNames', ...
 {'N','rep','t','c_d2','c_d3','c_d4','c_d5','c_d8','f_d2','f_d3','f_d4','f_d5','f_d8', ...
 'diff_d2','diff_d3','diff_d4','diff_d5','diff_d8'}),fullfile(folder,'raw_refinement.csv'));
save(fullfile(folder,'settings.mat'),'cfg');
end

function S=analyze_refinement(cfg,folder)
A=sortrows(readtable(fullfile(folder,'raw_refinement.csv')),{'rep','t'});
times=unique(A.t);R=numel(unique(A.rep));nt=numel(times);D=numel(cfg.dims);
V=permute(reshape(A{:,4:18},nt,R,15),[2,1,3]);
rng(807092026,'twister');ind=randi(R,cfg.bootstrap,R);rows=cell(2*D,10);row=0;
for k=1:2
    if k==1, mask=abs(times-100)<1e-12;name='T100';
    else, mask=times>=80 & times<=100;name='late80_100'; end
    B=reshape(mean(V(:,mask,:),2),R,15);
    for q=1:D
        c=B(:,q);f=B(:,q+D);diff=B(:,q+2*D);
        percent=100*(mean(c)/mean(f)-1);
        boot=100*(mean(c(ind),2)./mean(f(ind),2)-1);ci=percentiles(boot);
        drms=sqrt(mean(diff));row=row+1;
        rows(row,:)={cfg.dims(q),name,mean(c),mean(f),percent,ci(1),ci(2),drms,drms/sqrt(mean(f)),R};
    end
end
S=cell2table(rows,'VariableNames',{'d','window','coarse_mse','fine_mse','relative_percent', ...
    'ci_low_percent','ci_high_percent','coupling_difference_rms','difference_relative_rms','R'});
writetable(S,fullfile(folder,'refinement_summary.csv'));disp(S);
end

function values=percentiles(x)
x=sort(x(:));pos=1+(numel(x)-1)*[0.025,0.975];
lo=floor(pos);hi=ceil(pos);w=pos-lo;
values=(1-w).*x(lo)'+w.*x(hi)';
end

function fitsOut=plot_both(S,folder)
fitsOut=plot_results(S,folder,'mse_T','multiplicative_PoC_T100','PoC-Error');
plot_results(S,folder,'mse_late','multiplicative_PoC_window80_100','Window-averaged PoC-Error');
end

function fitsOut=plot_results(S,folder,field,stem,ylabelText)
figdir=fullfile(folder,'figures');if ~exist(figdir,'dir'),mkdir(figdir);end
allD=[2,3,4,5,8];groups={[2,3],4,[5,8]};
fitcolors=[1 .498 .055;.173 .627 .173;1 .22 .22;.50 .50 .50;1 .498 .055];
markcolors=[.55 .337 .294;.89 .467 .761;.50 .50 .50;.737 .741 .133;.122 .467 .706];
marks={'d','s','o','v','x'};
titles={'(a) PoC rates for $d\in\{2,3\}$','(b) PoC rates for $d=4$', ...
        '(c) PoC rates for $d\in\{5,8\}$'};
fig=figure('Color','w','Position',[60,100,1720,500]);
fitsOut=zeros(5,2);k=0;
for p=1:3
    ax=subplot(1,3,p,'Parent',fig);hold(ax,'on');box(ax,'on');
    set(ax,'XScale','log','YScale','log');ds=groups{p};
    handles=gobjects(numel(ds),1);labels=cell(1,numel(ds));lo=inf;hi=0;leftmax=0;
    for q=1:numel(ds)
        d=ds(q);idx=find(allD==d);A=sortrows(S(S.d==d,:),'N');N=A.N;E=A.(field);
        assert(numel(N)==7 && all(isfinite(E)) && all(E>0),'Invalid observed data.');
        coef=polyfit(log(N),log(E),1);dense=logspace(log10(N(1)),log10(N(end)),250);
        fitted=exp(polyval(coef,log(dense)));k=k+1;fitsOut(k,:)=[d,coef(1)];
        plot(ax,dense,fitted,'Color',fitcolors(idx,:),'LineWidth',1.5,'HandleVisibility','off');
        plot(ax,N,E,'LineStyle','none','Marker',marks{idx},'Color',markcolors(idx,:), ...
            'MarkerFaceColor',markcolors(idx,:),'MarkerSize',6,'HandleVisibility','off');
        handles(q)=plot(ax,nan,nan,'-','Color',fitcolors(idx,:),'Marker',marks{idx}, ...
            'MarkerFaceColor',markcolors(idx,:),'MarkerEdgeColor',markcolors(idx,:), ...
            'LineWidth',1.5,'MarkerSize',6);
        labels{q}=sprintf('d=%d, slope=%.3f',d,coef(1));
        leftmax=max(leftmax,exp(polyval(coef,log(N(1)))));
        lo=min(lo,min([E;fitted(:)]));hi=max(hi,max(E));
    end
    ref=1.8*leftmax*(dense/N(1)).^(-1);
    guide=plot(ax,dense,ref,'--','Color',[0,.65,.12],'LineWidth',2);
    hi=max(hi,max(ref));
    xlabel(ax,'Number of Particles');ylabel(ax,ylabelText);
    title(ax,titles{p},'Interpreter','latex','FontSize',16);
    legend(ax,[guide;handles],[{'Order 1'},labels],'Location','southwest','FontSize',10, ...
        'Interpreter','none');
    set(ax,'XScale','log','YScale','log','XLim',[32,3200],'YLim',[lo*.55,hi*1.35], ...
        'XTick',[100,1000],'FontSize',12,'LineWidth',.8,'TickDir','in');
    grid(ax,'off');if isprop(ax,'Toolbar'),ax.Toolbar.Visible='off';end
end
set(fig,'PaperUnits','inches','PaperPosition',[0,0,17.2,5],'PaperSize',[17.2,5]);
print(fig,fullfile(figdir,[stem,'.png']),'-dpng','-r300');
print(fig,fullfile(figdir,[stem,'.pdf']),'-dpdf','-painters');
savefig(fig,fullfile(figdir,[stem,'.fig']));
writetable(array2table(fitsOut,'VariableNames',{'d','slope'}), ...
    fullfile(folder,[stem,'_fitted_slopes.csv']));
end

function test_core()
y=reshape(linspace(-5,5,35),7,5);h=1/64;
[z,~,res]=implicit_stage(y,h,true,1e-12);
assert(res<=1e-12);
r=(1+2*h)*z+14*h*z.^3-h/2*mean(z,1)-y;
assert(max(abs(r(:)))<1e-9);
p=[7,2,4,1,6,5,3];zp=implicit_stage(y(p,:),h,true,1e-12);
zperm=z(p,:);assert(max(abs(zp(:)-zperm(:)))<1e-10);
zn=implicit_stage(-y,h,true,1e-12);assert(max(abs(zn(:)+z(:)))<1e-10);
% Finite differences check the fully implicit Jacobian.
n=7;v=sin((1:n)');q=z(:,1);epsFD=1e-6;
f=@(x)(1+2*h)*x+14*h*x.^3-h/2*mean(x);
fd=(f(q+epsFD*v)-f(q-epsFD*v))/(2*epsFD);
exact=(1+2*h+42*h*q.^2).*v-h/2*mean(v);
assert(norm(fd-exact)<1e-7);
fprintf('PASS: implicit residual, permutation, odd symmetry, and Jacobian.\n');
end
