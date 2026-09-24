#!/usr/bin/env python3
"""Render native track-finding scaling plots from a collated campaign."""
from __future__ import annotations
import argparse, contextlib, csv, io
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from campaign_util import read_yaml

METRICS=(("native_ari_signal","Signal ARI"),("native_ari_with_background","Inclusive ARI"),("native_matched_iou_mean","Matched IoU"),("native_track_purity_global","Global track purity"),("native_track_efficiency_global","Global track efficiency"),("native_fake_rate","Fake-track rate"),("native_background_rejection","Background rejection"),("native_signal_loss_to_background","Signal loss to background"))
FOCUS_METRICS=(("native_matched_iou_mean","matched_iou"),("native_track_efficiency_global","efficiency"),("native_signal_loss_to_background","signal_loss_to_background"))
PAPER_METRICS=(("native_ari_signal","Signal ARI"),("native_track_efficiency_global","Track efficiency"),("native_track_purity_global","Track purity"))
DNP_PAPER="#F7F6F2"
DNP_NAVY="#18344A"
DNP_RUST="#B85C3B"
DNP_TEXT="#202428"
DNP_MUTED="#6B7177"
DNP_SOFT="#E8E6E1"
OTHER_COLORS=("#C3C8C6","#B3BAB7","#A2ABA7","#929C98","#828E89")
def val(x):
    try: return float(x) if x not in (None,"","None","null") else None
    except ValueError: return None
def adapter(r): return r.get("backbone_run_id")=="adapteronly" or r.get("model_family")=="adapteronly"
def parse():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--campaign-dir",required=True);p.add_argument("--summary-name",default="summary");p.add_argument("--output-dir");return p.parse_args()
def model_labels(rs):
    models={}
    for r in rs:
        name=r.get("backbone_run_id") or r["run_id"]
        if not adapter(r):
            models[name]=(int(r["embed_dim"]),int(r["num_layers_backbone"]))
    ordered=sorted(models,key=lambda name:(models[name],name))
    return {name:f"m{i}" for i,name in enumerate(ordered,1)}
def draw(ax,rs,key,coat,labels=None):
    groups=defaultdict(list); base=[]
    for r in rs:
        x,y=val(r.get("labeled_events")),val(r.get(key))
        if x is not None and y is not None: groups[r.get("backbone_run_id",r["run_id"])].append((x,y))
        z=val(r.get(key.replace("native_","native_coatjava_")))
        if z is not None: base.append(z)
    for name,pts in sorted(groups.items()):
        pts.sort();ax.plot(*zip(*pts),marker="o",label=(labels or {}).get(name,"Adapter only" if name=="adapteronly" else name))
    if coat and base: ax.axhline(sum(base)/len(base),color="black",linestyle="--",label="COATJAVA")
    ax.set_xscale("log");ax.grid(True,which="both",alpha=.25)
def suite(rs,out,name,coat,labels=None):
    out.mkdir(parents=True,exist_ok=True)
    fig,axs=plt.subplots(3,3,figsize=(13,10),constrained_layout=True)
    for ax,(key,label) in zip(axs.flat,METRICS): draw(ax,rs,key,coat,labels);ax.set_title(label);ax.set_xlabel("Labeled events")
    for ax in list(axs.flat)[len(METRICS):]: ax.set_visible(False)
    h,l=axs.flat[0].get_legend_handles_labels()
    if h: fig.legend(h,l,loc="outside lower center",ncol=min(4,len(h)),fontsize="small")
    fig.savefig(out/f"{name}.png",dpi=160);plt.close(fig)
def focus_suite(rs,out,model_names):
    labels=dict(METRICS)
    for key,name in FOCUS_METRICS:
        fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
        draw(ax,rs,key,True,model_names)
        ax.set_title(labels[key]);ax.set_xlabel("Labeled events");ax.set_ylabel(labels[key])
        if ax.has_data(): ax.legend(fontsize="small")
        fig.savefig(out/f"{name}_vs_labeled_events.png",dpi=160);plt.close(fig)
def individual_suite(rs,out,model_names):
    out.mkdir(parents=True,exist_ok=True)
    for key,label in METRICS:
        fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
        draw(ax,rs,key,True,model_names)
        ax.set_title(label);ax.set_xlabel("Labeled events");ax.set_ylabel(label)
        if ax.has_data(): ax.legend(fontsize="small")
        fig.savefig(out/f"{key.removeprefix('native_')}_vs_labeled_events.png",dpi=200)
        plt.close(fig)
def paper_suite(rs,out,model_names):
    out.mkdir(parents=True,exist_ok=True)
    by_label={label:name for name,label in model_names.items()}
    with (out/"model_labels.csv").open("w",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(("plot_label","backbone_run_id","embed_dim","num_layers_backbone"))
        for label,name in sorted(by_label.items(),key=lambda item:int(item[0][1:])):
            row=next(r for r in rs if r.get("backbone_run_id")==name)
            writer.writerow((label,name,row["embed_dim"],row["num_layers_backbone"]))
    ordered=sorted(by_label,key=lambda label:int(label[1:]))
    has_m6_largest=bool(ordered) and ordered[-1]=="m6"
    other_names=[by_label[label] for label in ordered if label!="m6"]
    def points(rows,key):
        return sorted(
            (x,y) for r in rows
            if (x:=val(r.get("labeled_events"))) is not None
            and (y:=val(r.get(key))) is not None
        )
    def line(ax,rows,key,label,color,lw=2.4,alpha=1,zorder=3):
        pts=points(rows,key)
        if pts:
            ax.plot(*zip(*pts),label=label,color=color,lw=lw,alpha=alpha,
                    marker="o",ms=5.5,zorder=zorder)
    for subset in ("all_models","adapter_m6_coatjava","other_pretrained_lines","other_pretrained_envelope"):
        if subset!="all_models" and not has_m6_largest:
            continue
        for layout in ("3x1","1x3"):
            vertical=layout=="3x1"
            fig,axs=plt.subplots(3 if vertical else 1,1 if vertical else 3,figsize=(7.2,11) if vertical else (15,4.6),squeeze=False,constrained_layout=True)
            fig.patch.set_facecolor(DNP_PAPER)
            for ax,(key,title) in zip(axs.flat,PAPER_METRICS):
                ax.set_facecolor(DNP_PAPER)
                line(ax,[r for r in rs if adapter(r)],key,"Adapter only",DNP_NAVY)
                if subset=="all_models":
                    for i,label in enumerate(ordered):
                        if has_m6_largest and label=="m6": continue
                        rows=[r for r in rs if r.get("backbone_run_id")==by_label[label]]
                        line(ax,rows,key,label,OTHER_COLORS[i%len(OTHER_COLORS)],lw=1.7,alpha=.9,zorder=2)
                elif subset=="other_pretrained_lines":
                    for i,name in enumerate(other_names):
                        rows=[r for r in rs if r.get("backbone_run_id")==name]
                        line(ax,rows,key,"Other pretrained backbones" if i==0 else "_nolegend_",
                             "#AAB2AE",lw=1.6,alpha=.65,zorder=2)
                elif subset=="other_pretrained_envelope":
                    by_x=defaultdict(list)
                    for r in rs:
                        if r.get("backbone_run_id") in other_names:
                            x,y=val(r.get("labeled_events")),val(r.get(key))
                            if x is not None and y is not None: by_x[x].append(y)
                    xs=sorted(x for x,ys in by_x.items() if len(ys)>=2)
                    if xs:
                        lower=[min(by_x[x]) for x in xs]
                        upper=[max(by_x[x]) for x in xs]
                        ax.fill_between(xs,lower,upper,color="#B5BDB9",alpha=.55,
                                        label="Other pretrained backbones (range)",zorder=1)
                        ax.plot(xs,lower,color="#9CA7A2",lw=.8,zorder=2)
                        ax.plot(xs,upper,color="#9CA7A2",lw=.8,zorder=2)
                if has_m6_largest:
                    line(ax,[r for r in rs if r.get("backbone_run_id")==by_label["m6"]],key,
                         "Largest pretrained backbone (m6)",DNP_RUST,lw=2.7,zorder=4)
                baseline=[val(r.get(key.replace("native_","native_coatjava_"))) for r in rs]
                baseline=[x for x in baseline if x is not None]
                if baseline: ax.axhline(sum(baseline)/len(baseline),label="COATJAVA",color=DNP_TEXT,lw=1.8,ls="--",zorder=2)
                ax.set_title(title,fontsize=13,weight="semibold",pad=10,color=DNP_TEXT)
                ax.set_xscale("log");ax.set_xlabel("Labeled events",color=DNP_TEXT)
                ax.set_xlim(80,125000)
                ax.set_xticks((100,1000,10000,100000),labels=("100","1k","10k","100k"))
                values=[val(r.get(key)) for r in rs]+baseline
                values=[x for x in values if x is not None]
                if values: ax.set_ylim(max(0,min(values)-0.04),min(1,max(values)+0.04))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.grid(axis="y",color=DNP_SOFT,lw=.9)
                ax.spines[["top","right"]].set_visible(False)
                ax.spines[["left","bottom"]].set_color(DNP_MUTED)
                ax.tick_params(labelsize=10,colors=DNP_TEXT)
            handles,legend_labels=axs.flat[0].get_legend_handles_labels()
            fig.legend(handles,legend_labels,loc="outside lower center",
                       ncol=4 if subset=="all_models" else 2 if subset.startswith("other_pretrained") else 3,
                       frameon=False,fontsize=9.5,labelcolor=DNP_TEXT)
            for extension in ("png","pdf"):
                fig.savefig(out/f"track_finding_{subset}_{layout}.{extension}",dpi=300,facecolor=DNP_PAPER)
            plt.close(fig)
def nparams(r):
    if adapter(r): return None
    try:
        p=read_yaml(Path(r["model_yaml"]))[r["model_config"]]
        from fm4npp.models.mambagpt import Mamba1GPT
        from fm4npp.utils import count_parameters
        with contextlib.redirect_stdout(io.StringIO()): m=Mamba1GPT(embed_dim=int(p["embed_dim"]),num_layers=int(p["num_layers_backbone"]),d_state=int(p.get("d_state",16)),d_conv=int(p.get("d_conv",4)),expand=int(p.get("expand",2)),klen=int(p.get("klen",1)),embed_method=p.get("embed_method","pos_only"),pe_method=p.get("pe_method","nerf"))
        return int(count_parameters(m))
    except Exception:return None
def parameter_suite(rs,out):
    counts={r["backbone_run_id"]:nparams(r) for r in rs}
    for key,label in METRICS:
        fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True);groups=defaultdict(list)
        for r in rs:
            n,x,y=counts.get(r.get("backbone_run_id")),val(r.get("labeled_events")),val(r.get(key))
            if n is not None and x is not None and y is not None:groups[int(x)].append((n,y))
        for labels,pts in sorted(groups.items()):pts.sort();ax.plot(*zip(*pts),marker="o",label=f"{labels:g} labels")
        if groups:
            ax.set_xscale("log");ax.set_xlabel("Pretrained backbone parameters");ax.set_ylabel(label);ax.grid(True,which="both",alpha=.25);ax.legend(fontsize="small");fig.savefig(out/f"pretrained_{key}_vs_backbone_params.png",dpi=160)
        plt.close(fig)
def main():
    a=parse();root=Path(a.campaign_dir).resolve();summary=root/a.summary_name;out=Path(a.output_dir).resolve() if a.output_dir else summary/"plots";out.mkdir(parents=True,exist_ok=True)
    with (summary/"run_table.csv").open(newline="") as s:rs=[r for r in csv.DictReader(s) if r.get("summary_found","").lower()=="true"]
    if not rs:raise ValueError("No completed evaluations; collate first.")
    labels=model_labels(rs)
    for b in sorted({r.get("backbone_run_id") for r in rs}):suite([r for r in rs if r.get("backbone_run_id")==b],out/f"per_model",f"{b}_vs_labeled_events",True,labels)
    suite(rs,out,"adapter_vs_all_pretrained",False,labels);suite(rs,out,"adapter_pretrained_and_coatjava",True,labels)
    focus_suite(rs,out,labels);individual_suite(rs,out/"individual",labels);paper_suite(rs,out/"presentation",labels)
    parameter_suite([r for r in rs if not adapter(r)],out);print(f"Wrote plots to {out}")
if __name__=="__main__":main()
