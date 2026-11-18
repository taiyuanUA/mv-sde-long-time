"""Plot observed Monte Carlo errors, not theoretical replacement data."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
import fitz

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
S=pd.read_csv(ROOT/'data'/'summary.csv')
DIMS=[2,3,4,5,8]
GROUPS=[[2,3],[4],[5,8]]
FITCOLORS={2:'#ff7f0e',3:'#2ca02c',4:'#ff3838',5:'#808080',8:'#ff7f0e'}
MARKCOLORS={2:'#8c564b',3:'#e377c2',4:'#808080',5:'#bcbd22',8:'#1f77b4'}
MARKS={2:'D',3:'s',4:'o',5:'v',8:'x'}
GREEN='#00a61f'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,
 'axes.linewidth':0.9,'mathtext.fontset':'dejavusans','pdf.fonttype':42})
def plot_rates(field, stem, ylabel):
    panels=[];pdfs=[]
    for k,ds in enumerate(GROUPS):
        fig,ax=plt.subplots(figsize=(6.8,5.0))
        fig.subplots_adjust(left=.17,bottom=.15,right=.98,top=.895)
        ax.set_xscale('log');ax.set_yscale('log')
        handles=[];labels=[];lo=np.inf;hi=0.;leftmax=0.
        for d in ds:
            rows=S[S.d==d].sort_values('N');N=rows.N.to_numpy(float);E=rows[field].to_numpy(float)
            assert len(N)==7 and np.all(np.isfinite(E)) and np.all(E>0)
            coef=np.polyfit(np.log(N),np.log(E),1)
            dense=np.logspace(np.log10(N[0]),np.log10(N[-1]),250)
            fit=np.exp(np.polyval(coef,np.log(dense)))
            ax.plot(dense,fit,color=FITCOLORS[d],lw=1.5)
            ax.plot(N,E,ls='none',marker=MARKS[d],ms=5.5,mew=1.1,
                    color=MARKCOLORS[d],markerfacecolor=MARKCOLORS[d])
            handles.append(Line2D([],[],color=FITCOLORS[d],lw=1.5,marker=MARKS[d],
                        markerfacecolor=MARKCOLORS[d],markeredgecolor=MARKCOLORS[d],ms=5.5))
            labels.append(f'd={d}, slope={coef[0]:.3f}')
            leftmax=max(leftmax,float(fit[0]));lo=min(lo,float(E.min()),float(fit.min()));hi=max(hi,float(E.max()))
        reference=1.8*leftmax*(dense/N[0])**(-1)
        ax.plot(dense,reference,ls='--',color=GREEN,lw=2.0)
        hi=max(hi,float(reference.max()))
        handles.insert(0,Line2D([],[],ls='--',color=GREEN,lw=2.0));labels.insert(0,'Order 1')
        if k==0:title=r'(a) PoC rates for $d\in\{2,3\}$'
        elif k==1:title=r'(b) PoC rates for $d=4$'
        else:title=r'(c) PoC rates for $d\in\{5,8\}$'
        ax.set_title(title,fontsize=15,pad=12)
        ax.set_xlabel('Number of Particles',fontsize=13)
        ax.set_ylabel(ylabel,fontsize=13)
        ax.set_xlim(32,3200);ax.set_ylim(lo*.55,hi*1.35);ax.set_xticks([100,1000])
        ax.tick_params(axis='both',which='both',direction='in',top=True,right=True)
        ax.legend(handles,labels,loc='lower left',fontsize=10,framealpha=1,edgecolor='#bfbfbf')
        ax.grid(False)
        path=OUT/f'{stem}_panel_{chr(97+k)}.png';pdf=OUT/f'{stem}_panel_{chr(97+k)}.pdf'
        fig.savefig(path,dpi=220);fig.savefig(pdf);plt.close(fig)
        panels.append(path);pdfs.append(pdf)
    # Assemble the separately generated panels; do not re-draw or interpolate data.
    images=[Image.open(x).convert('RGB') for x in panels]
    combined=Image.new('RGB',(sum(im.width for im in images),max(im.height for im in images)),'white')
    x=0
    for im in images:combined.paste(im,(x,0));x+=im.width
    combined.save(OUT/(stem+'.png'))
    out=fitz.open();docs=[fitz.open(p) for p in pdfs]
    width=sum(doc[0].rect.width for doc in docs);height=max(doc[0].rect.height for doc in docs)
    p=out.new_page(width=width,height=height);x=0
    for doc in docs:
        w=doc[0].rect.width;p.show_pdf_page(fitz.Rect(x,0,x+w,height),doc,0);x+=w
    out.save(OUT/(stem+'.pdf'))
    for doc in docs:doc.close()
    out.close()

plot_rates('mse_T','multiplicative_PoC_T100','PoC-Error')
plot_rates('mse_late','multiplicative_PoC_window80_100','Window-averaged PoC-Error')
# Auxiliary long-time curve: one distinct figure, no altered observations.
A=pd.read_csv(ROOT/'data'/'time_mean.csv')
fig,ax=plt.subplots(figsize=(8.2,5.0))
for N in sorted(A.N.unique()):
    q=A[A.N==N].sort_values('t')
    ax.plot(q.t,q.mse_d4,label=f'N={N}',lw=1.0)
ax.set_xlabel('Time t');ax.set_ylabel('Mean-square coupling error')
ax.set_title('Multiplicative-noise PoC error: d=4')
ax.legend(ncol=2,fontsize=9);ax.grid(alpha=.2);fig.tight_layout()
fig.savefig(OUT/'long_time_d4.png',dpi=200);fig.savefig(OUT/'long_time_d4.pdf');plt.close(fig)
print('Figures plotted from the new raw simulations. Slopes freely fitted to all seven N values.')
