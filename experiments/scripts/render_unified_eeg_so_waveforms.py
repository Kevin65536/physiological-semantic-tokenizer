"""Export matched measured waveforms from the completed unified-EEG S/O pilot."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import evaluate_unified_eeg_so_pilot as pilot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pymupdf
from PIL import Image

RUN = pilot.OUT
NAMES = {'eeg_fnirs_single_trial':'Single-Trial', 'simultaneous_eeg_nirs':'Simultaneous',
         'refed':'REFED', 'visual_cognitive_motivation':'Visual'}
X, S, O = '#27394e', '#007a72', '#ae552a'
BEFORE, AFTER = '#335f82', '#b2492d'
STYLES = ['-', (0,(5,2)), (0,(1.3,1.6))]
DPI = 210
ZOOM = (40,44)


def axis_style(ax,lim):
    ax.spines[['top','right']].set_visible(False)
    ax.set_ylim(*lim)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3,steps=[1,2,2.5,5,10]))
    ax.axhline(0,linewidth=.55,color='#b8bfc7',zorder=0)
    ax.grid(axis='y',linewidth=.35,color='#e0e4e6')
    ax.tick_params(labelsize=8,length=2.3,pad=2)


def limits(a):
    a=np.asarray(a); lo=min(0.,float(a.min()));hi=max(0.,float(a.max()))
    pad=max(hi-lo,1e-12)*.07
    return lo-pad,hi+pad


def extract(i):
    wd=RUN/'inputs'/str(i)
    target=RUN/'targets'/str(i)/'unified'
    fsdir=RUN/'fits'/str(i)/'unified'
    ident=pilot.read_json(wd/'identity.json')
    g=ident['group']
    assert g['split']=='validation'
    meta=pilot.read_json(pilot.SO/'windows'/(g['key']+'.json'))
    assert meta['source']==g and meta['EEG_channels']==g['EEG_channels']
    assert all(pilot.read_json(target/(v+'.json'))['qualified'] for v in ('baseline','ocular1'))
    j=next(k for k,c in enumerate(g['EEG_channels']) if c.upper().startswith(('FP','AFP')))
    x0=np.load(wd/'unified.npy',mmap_mode='r')[:,j].copy()
    s0=np.load(target/'baseline_S.npy',mmap_mode='r')[:,j].copy()
    with np.load(fsdir/'baseline.npz') as f:
        h=f['target'][:,0,1:]*ident['co']['hb_factor']
        hs=f['prediction'][:,0,1:]*ident['co']['hb_factor']
    hb_input=np.load(wd/'hb.npy')[:,0]*ident['co']['hb_factor']
    np.testing.assert_allclose(h,hb_input,rtol=1e-12,atol=1e-12)
    _,x1,rate,de,_,_,_=pilot.known_inputs(i,'unified','ocular1')
    assert rate==200 and len(x0)==18000 and h.shape==(900,2)
    with np.load(fsdir/'ocular1.npz') as f: proxy=f['prediction'][:,:,0]
    with np.load(RUN/'references'/(g['dataset']+'.npz')) as f:
        model=pilot.bind_reference(dict(f),np.load(wd/'projection.npy'),len(proxy))
    s1=pilot.voltage_power_cap_target(x1,model,proxy,cap=1.,audit_final=False,frame_batch_size=8)['s'][:,j].copy()
    x1=x1[:,j].copy()
    delta=de[:,j].copy()
    np.testing.assert_allclose(x1,x0+delta,rtol=1e-13,atol=1e-15)
    with np.load(target/'ocular1_example.npz') as saved:
        samples={'x':x1,'s':s1,'base_s':s0,'delta':delta}
        errors=[]
        for key,val in samples.items():
            err=float(np.max(abs(val[::2]-saved[key])))
            np.testing.assert_allclose(val[::2],saved[key],rtol=1e-10,atol=1e-14)
            errors.append(err)
    baseline=np.stack([x0,s0,x0-s0])
    ocular=np.stack([x1,s1,x1-s1])
    hb=np.stack([h,hs,h-hs])
    for arr in [baseline,ocular,hb]:
        assert np.isfinite(arr).all()
        np.testing.assert_allclose(arr[1]+arr[2],arr[0],rtol=1e-12,atol=1e-12)
    rel=meta['EEG_unit']=='unknown_REFED_EEG_export'
    assert rel or meta['EEG_unit']=='uV'
    return dict(index=i,dataset=NAMES[g['dataset']],subject=g['subject'],record=g['native_record'],
                start=g['start_sample']/g['rate'],electrode=g['EEG_channels'][j],region=ident['positions'][0],
                rel=rel,unit='相对单位（×10⁻⁵）' if rel else 'μV',factor=1e5 if rel else 1.,
                baseline=baseline,ocular=ocular,hb=hb,delta=delta,replay_max=max(errors))


def baseline_plate(cases,out):
    fig=plt.figure(figsize=(17,17.3),facecolor='white')
    fig.text(.06,.975,'四数据集的 EEG 与血红蛋白波形',fontsize=22,weight='bold',va='top')
    fig.text(.06,.948,'每库取面板首名被试。EEG 先看全窗，再看 40–44 秒的细节。各信号均给出 X、S、O 叠加与分行波形。',fontsize=11)
    fig.legend([Line2D([],[],color=c,ls=l,lw=1.8) for c,l in zip((X,S,O),STYLES)],
               ['观测 X','条件分量 S','残差 O'],loc='upper right',bbox_to_anchor=(.977,.976),ncol=3,frameon=False)
    left,right=.10,.987
    for j,c in enumerate(cases):
        middle=left+(right-left)*(j+.5)/4
        fig.text(middle,.918,c['dataset'],ha='center',fontsize=15,weight='bold')
        fig.text(middle,.900,f"{c['subject']}  ·  {c['record']}",ha='center',fontsize=10)
        fig.text(middle,.883,f"EEG {c['electrode']}   /   Hb {c['region']}",ha='center',fontsize=9)
    specs=[('EEG 电压','全窗 90 秒，阴影为 40–44 秒放大段 · 200 Hz',.847,.599),
           ('氧合血红蛋白 HbO','完整 90 秒窗口 · 10 Hz · 被试校准单位',.562,.347),
           ('脱氧血红蛋白 HbR','完整 90 秒窗口 · 10 Hz · 被试校准单位',.308,.093)]
    t_eeg=np.arange(800)/200+40
    t_hb=np.arange(900)/10
    hb_limits=[limits(np.concatenate([c['hb'][:,:,j].ravel() for c in cases])) for j in range(2)]
    full_uv_limit=limits(np.concatenate([c['baseline'].ravel() for c in cases if not c['rel']]))
    uv_limit=limits(np.concatenate([c['baseline'][:,8000:8800].ravel() for c in cases if not c['rel']]))
    for k,(name,span,top,bottom) in enumerate(specs):
        fig.text(.06,top+.014,name,fontsize=13,weight='bold')
        fig.text(.258,top+.014,span,fontsize=10,color='#405368')
        labels=('全窗叠加','放大叠加','观测 X','分量 S','分量 O') if k==0 else ('叠加','观测 X','分量 S','分量 O')
        grid=fig.add_gridspec(len(labels),4,left=left,right=right,top=top,bottom=bottom,
                              height_ratios=([1.25,1.25,1,1,1] if k==0 else [1.35,1,1,1]),wspace=.19,hspace=.22)
        for j,c in enumerate(cases):
            if k==0:
                wave=c['baseline'][:,8000:8800]*c['factor']; t=t_eeg
                lim=limits(wave) if c['rel'] else uv_limit
            else:
                wave=c['hb'][:,:,k-1];t=t_hb;lim=hb_limits[k-1]
            for r,label in enumerate(labels):
                ax=fig.add_subplot(grid[r,j])
                full=(k==0 and r==0)
                data=c['baseline']*c['factor'] if full else wave
                time=np.arange(18000)/200 if full else t
                if full:
                    ax.axvspan(*ZOOM,color='#f2deb2',alpha=.8)
                layer=r-1 if k==0 else r
                for q in (range(3) if layer<=0 else [layer-1]):
                    ax.plot(time,data[q],color=(X,S,O)[q],ls=STYLES[q] if layer<=0 else '-',
                            lw=.65 if full else .88,zorder=2+q)
                scale=(limits(data) if c['rel'] else full_uv_limit) if full else lim
                axis_style(ax,scale)
                short=k==0 and not full
                ax.set_xlim(*(ZOOM if short else (0,90)))
                ax.set_xticks(range(40,45) if short else (0,15,30,45,60,75,90))
                last=(r==len(labels)-1)
                ax.tick_params(labelbottom=last or full,bottom=last or full)
                if j==0: ax.set_ylabel(label,rotation=0,labelpad=24,va='center',fontsize=9)
                if k==0 and r==0: ax.text(.99,.98,c['unit'],ha='right',va='top',transform=ax.transAxes,fontsize=7.5,
                                          bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=.5))
    fig.text(.06,.057,'横轴均为距窗口起点的秒数。三个已标定数据集的 EEG 共用 μV 刻度；REFED 保留发布相对单位。HbO、HbR 分别跨库共用被试校准刻度。',fontsize=10)
    fig.text(.06,.039,'X 为统一预处理后的观测；S 来自完整窗口的条件拟合与目标构建，O = X − S。Hb 已减窗口前 5 秒参考。',fontsize=10)
    path=out/'four_dataset_SO.png';fig.savefig(path,dpi=DPI);plt.close(fig);return path


def ocular_plate(c,out):
    fig=plt.figure(figsize=(15,9.0),facecolor='white')
    fig.text(.07,.965,f"{c['dataset']}   ·   {c['subject']}   |   眼电注入前后的 EEG 分解",va='top',fontsize=20,weight='bold')
    fig.text(.07,.902,f"{c['record']}   /   电极 {c['electrode']}   /   窗口自记录 {c['start']:.3f} 秒开始   /   纵轴 {c['unit']}",fontsize=11)
    t=np.arange(18000)/200
    full=slice(6000,10000); zoom=slice(8000,8800)
    ax=fig.add_axes([.115,.747,.85,.095]);delta=c['delta']*c['factor']
    ax.axvspan(*ZOOM,color='#f2deb2',alpha=.6);ax.plot(t[full],delta[full],color='#66534c',lw=.85)
    axis_style(ax,limits(delta[full]));ax.set_xlim(30,50);ax.set_xticks([30,32,34,36,38,40,42,44,46,48,50])
    ax.set_ylabel('已知注入量',fontsize=10);ax.set_title('实际进入模型的眼电增量   ·   阴影为下方 40–44 秒放大段',fontsize=10,loc='left')
    grid=fig.add_gridspec(3,3,left=.115,right=.966,top=.646,bottom=.132,hspace=.28,wspace=.20,width_ratios=[1.55,1,1])
    rows=('输入 EEG  X','拟合分量  S','残差分量  O')
    for r,label in enumerate(rows):
        a,b=c['baseline'][r]*c['factor'],c['ocular'][r]*c['factor']
        lim=limits(np.stack([a[full],b[full]]))
        for j in range(3):
            ax=fig.add_subplot(grid[r,j]);axis_style(ax,lim)
            if j==0:
                ax.axvspan(*ZOOM,color='#f2deb2',alpha=.6)
                ax.plot(t[full],a[full],color=BEFORE,lw=.95)
                ax.plot(t[full],b[full],color=AFTER,ls='--',lw=.85)
                ax.set_xlim(30,50);ax.set_xticks([30,35,40,45,50])
                ax.set_ylabel(label,fontsize=10)
            else:
                arr,color=(a,BEFORE) if j==1 else (b,AFTER)
                ax.plot(t[zoom],arr[zoom],color=color,lw=.95);ax.set_xlim(*ZOOM);ax.set_xticks(range(40,45))
            if r==0: ax.set_title(('前后叠加   ·   注入段 30–50 秒','注入前   ·   局部放大','注入后   ·   同段放大')[j],fontsize=12)
            ax.tick_params(labelbottom=r==2)
            if r==2: ax.set_xlabel('窗口内时间（秒）',fontsize=9)
    fig.legend([Line2D([],[],color=BEFORE),Line2D([],[],color=AFTER,ls='--')],
               ['注入前','注入后'],loc='upper right',bbox_to_anchor=(.975,.966),ncol=2,frameon=False)
    fig.text(.07,.055,'实测垂直眼电（VEOG）供体按额前较强的空间权重叠加，取 1 倍强度。两种输入均通过同一预处理，各自使用对应的 90 秒条件拟合结果。',fontsize=10)
    fig.text(.07,.030,'同一行三图共用纵轴，不同行按各分量的实际幅值取刻度。左右两种条件均满足 X = S + O。',fontsize=10)
    path=out/f"ocular_{c['dataset']}_{c['subject']}.png"
    fig.savefig(path,dpi=DPI);plt.close(fig);return path


def atlas(pages,out):
    doc=pymupdf.open();bookmarks=[]
    for idx,(path,title,caption) in enumerate(pages,1):
        with Image.open(path) as im: w,h=im.size
        width,height=w*72/DPI,h*72/DPI
        page=doc.new_page(width=width,height=height+26)
        page.insert_image(pymupdf.Rect(0,0,width,height),filename=str(path))
        page.insert_text((26,height+16),caption,fontname='china-s',fontsize=9)
        bookmarks.append([1,title,idx])
    doc.set_toc(bookmarks)
    path=out/'waveform_atlas.pdf'
    doc.save(path,deflate=True);doc.close()
    with pymupdf.open(path) as check:
        assert len(check)==len(pages)
        for page in check:
            assert len(page.get_images())==1 and page.get_text().strip()
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    assert pilot.read_json(RUN/'verification.json')['status']=='passed'
    plan=pilot.read_json(RUN/'plan.json');assert len(plan['cases'])==8
    out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':['Noto Sans CJK JP','DejaVu Sans'],'font.size':10,
                         'axes.unicode_minus':False,'axes.edgecolor':'#b2bbc2','axes.linewidth':.55,
                         'text.color':'#1c2e3e','xtick.color':'#3f4d58','ytick.color':'#3f4d58'})
    with ProcessPoolExecutor(max_workers=4) as pool: cases=list(pool.map(extract,range(8)))
    first={}
    for c in cases:first.setdefault(c['dataset'],c)
    assert len(first)==4
    paths=[(baseline_plate(list(first.values()),out),'四库基线波形','四个公开验证窗口的统一输入与条件分量。每列展示同一被试的 EEG、HbO 与 HbR。')]
    for c in cases:
        paths.append((ocular_plate(c,out),f"{c['dataset']} · {c['subject']}",f"{c['dataset']}  {c['subject']}  {c['electrode']}   统一输入，眼电注入前后，完整扰动段与局部放大。"))
    atlas(paths,out)
    pilot.write_json(out/'verification.json',dict(source=str(RUN),baseline_indices=[c['index'] for c in first.values()],
        ocular_variant='ocular1',display_eeg_hz=200,display_hb_hz=10,scope='validation',pdf_pages=len(paths),
        examples=[{k:c[k] for k in ('index','dataset','subject','record','start','electrode','region','unit','replay_max')} for c in cases]))
    print(out,flush=True)


if __name__=='__main__':main()
