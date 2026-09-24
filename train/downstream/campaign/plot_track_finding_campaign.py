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
MODEL_COLORS=("#4E79A7","#59A14F","#B07AA1","#76B7B2","#F28E2B","#D43F3A")
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
    colors={label:MODEL_COLORS[(int(label[1:])-1)%len(MODEL_COLORS)] for label in by_label}
    series=[("Adapter only",None,"#666666")]+[(label,name,colors[label]) for label,name in sorted(by_label.items(),key=lambda item:int(item[0][1:]))]
    for subset in ("all_models","adapter_m6_coatjava"):
        if subset=="adapter_m6_coatjava" and "m6" not in by_label:
            continue
        selected=series if subset=="all_models" else [series[0],("m6",by_label["m6"],colors["m6"])]
        for layout in ("3x1","1x3"):
            vertical=layout=="3x1"
            fig,axs=plt.subplots(3 if vertical else 1,1 if vertical else 3,figsize=(7.2,11) if vertical else (15,4.6),squeeze=False,constrained_layout=True)
            for ax,(key,title) in zip(axs.flat,PAPER_METRICS):
                for label,name,color in selected:
                    rows=[r for r in rs if adapter(r)] if name is None else [r for r in rs if r.get("backbone_run_id")==name]
                    pts=sorted((val(r.get("labeled_events")),val(r.get(key))) for r in rows if val(r.get("labeled_events")) is not None and val(r.get(key)) is not None)
                    if pts:
                        ax.plot(*zip(*pts),label=label,color=color,lw=2.2,marker="o",ms=5.5,zorder=3)
                baseline=[val(r.get(key.replace("native_","native_coatjava_"))) for r in rs]
                baseline=[x for x in baseline if x is not None]
                if baseline: ax.axhline(sum(baseline)/len(baseline),label="COATJAVA",color="#222222",lw=2,ls="--",zorder=2)
                ax.set_title(title,fontsize=13,weight="semibold",pad=10)
                ax.set_xscale("log");ax.set_xlabel("Labeled events")
                ax.set_xlim(80,125000)
                ax.set_xticks((100,1000,10000,100000),labels=("100","1k","10k","100k"))
                values=[val(r.get(key)) for r in rs]+baseline
                values=[x for x in values if x is not None]
                if values: ax.set_ylim(max(0,min(values)-0.04),min(1,max(values)+0.04))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.grid(axis="y",color="#D7DCE2",lw=.8)
                ax.spines[["top","right"]].set_visible(False)
                ax.tick_params(labelsize=10)
            handles,legend_labels=axs.flat[0].get_legend_handles_labels()
            fig.legend(handles,legend_labels,loc="outside lower center",ncol=4 if subset=="all_models" else 3,frameon=False,fontsize=10)
            for extension in ("png","pdf"):
                fig.savefig(out/f"track_finding_{subset}_{layout}.{extension}",dpi=300)
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
