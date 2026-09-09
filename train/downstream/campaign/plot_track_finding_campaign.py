#!/usr/bin/env python3
"""Render native track-finding scaling plots from a collated campaign."""
from __future__ import annotations
import argparse, contextlib, csv, io
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from campaign_util import read_yaml

METRICS=(("native_ari_signal","Signal ARI"),("native_ari_with_background","Inclusive ARI"),("native_track_purity_global","Global track purity"),("native_fake_rate","Fake-track rate"),("native_background_rejection","Background rejection"),("native_signal_loss_to_background","Signal loss to background"))
def val(x):
    try: return float(x) if x not in (None,"","None","null") else None
    except ValueError: return None
def adapter(r): return r.get("backbone_run_id")=="adapteronly" or r.get("model_family")=="adapteronly"
def parse():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--campaign-dir",required=True);p.add_argument("--summary-name",default="summary");p.add_argument("--output-dir");return p.parse_args()
def draw(ax,rs,key,coat):
    groups=defaultdict(list); base=[]
    for r in rs:
        x,y=val(r.get("labeled_events")),val(r.get(key))
        if x is not None and y is not None: groups[r.get("backbone_run_id",r["run_id"])].append((x,y))
        z=val(r.get(key.replace("native_","native_coatjava_")))
        if z is not None: base.append(z)
    for name,pts in sorted(groups.items()):
        pts.sort();ax.plot(*zip(*pts),marker="o",label=name)
    if coat and base: ax.axhline(sum(base)/len(base),color="black",linestyle="--",label="COATJAVA")
    ax.set_xscale("log");ax.grid(True,which="both",alpha=.25)
def suite(rs,out,name,coat):
    out.mkdir(parents=True,exist_ok=True)
    fig,axs=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for ax,(key,label) in zip(axs.flat,METRICS): draw(ax,rs,key,coat);ax.set_title(label);ax.set_xlabel("Labeled events")
    h,l=axs.flat[0].get_legend_handles_labels()
    if h: fig.legend(h,l,loc="outside lower center",ncol=min(4,len(h)),fontsize="small")
    fig.savefig(out/f"{name}.png",dpi=160);plt.close(fig)
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
    for b in sorted({r.get("backbone_run_id") for r in rs}):suite([r for r in rs if r.get("backbone_run_id")==b],out/f"per_model",f"{b}_vs_labeled_events",True)
    suite(rs,out,"adapter_vs_all_pretrained",False);suite(rs,out,"adapter_pretrained_and_coatjava",True);parameter_suite([r for r in rs if not adapter(r)],out);print(f"Wrote plots to {out}")
if __name__=="__main__":main()
