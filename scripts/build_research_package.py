"""Build aggregate research exhibits and a one-page note from immutable results."""
import argparse
from collections import defaultdict
from datetime import date, datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
POLICY = "config/research_package_v1.json"
METHODS = ("keyword", "tfidf", "embedding")
METHOD_NAMES = ("Keyword", "TF-IDF", "Embedding")
TARGETS = ("relevant", "threat", "disruption")
MODELS = ("M0", "M1", "M2", "M3")
NAVY, TEAL, GOLD, GRAY = "#142F43", "#087F8C", "#B97813", "#8795A1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_hashes(root, hashes):
    for name, expected in hashes.items():
        if digest(root / name) != expected:
            raise ValueError(f"Frozen input hash mismatch: {name}")


def load_inputs(root=ROOT):
    policy = json.loads((root / POLICY).read_text())
    inputs = {POLICY: digest(root / POLICY)}
    folders = {}
    for name, source in policy["sources"].items():
        folder = root / source["directory"]
        manifest_path = folder / "manifest.json"
        if digest(manifest_path) != source["manifest_sha256"]:
            raise ValueError(f"Frozen {name} manifest changed")
        manifest = json.loads(manifest_path.read_text())
        verify_hashes(root, manifest["input_sha256"])
        verify_hashes(folder, manifest["artifact_sha256"])
        inputs[str(manifest_path.relative_to(root))] = digest(manifest_path)
        folders[name] = folder
    paths = {"weekly": folders["weekly"] / "weekly_news.jsonl",
             "weekly_summary": folders["weekly"] / "summary.json",
             "nlp": folders["nlp"] / "summary.json", "finance": folders["finance"] / "summary.json"}
    inputs.update({str(path.relative_to(root)): digest(path) for path in paths.values()})
    data = {k: read_rows(p) if k == "weekly" else json.loads(p.read_text()) for k, p in paths.items()}
    return policy, data, inputs


def report_data(data):
    quarterly = defaultdict(lambda: {"ft": 0, "guardian": 0, "candidate_count": 0, "relevant": 0, "threat": 0, "disruption": 0, "weeks": 0})
    for r in data["weekly"]:
        if r["timing"] != "publication" or not r["news_features_available"] or not "2019-01-01" <= r["week_start"] <= "2025-12-31":
            continue
        d = date.fromisoformat(r["week_start"])
        q = f"{d.year}-Q{(d.month-1)//3+1}"
        item = quarterly[q]
        if r["view"] in ("ft", "guardian"):
            item[r["view"]] += r["candidate_count"]
        elif r["view"] == "combined":
            item["candidate_count"] += r["candidate_count"]
            item["weeks"] += 1
            for target in TARGETS:
                item[target] += r["counts"][target]
    coverage = []
    for quarter, r in sorted(quarterly.items()):
        if r["ft"] + r["guardian"] != r["candidate_count"]:
            raise ValueError("Quarterly source counts do not reconcile")
        coverage.append({"quarter": quarter, **r,
            "threat_share": r["threat"] / r["candidate_count"] if r["candidate_count"] else None,
            "disruption_share": r["disruption"] / r["candidate_count"] if r["candidate_count"] else None})
    nlp = {m: {"overall": data["nlp"]["methods"][m]["overall"],
               "without_dominant_component": data["nlp"]["methods"][m]["without_dominant_component"]} for m in METHODS}
    finance = data["finance"]
    if not finance["final_test_scored"] or finance["retuned_from_final_test"]:
        raise ValueError("Reporting requires completed frozen final evaluation")
    return {"coverage": coverage, "nlp": nlp,
            "documents": data["weekly_summary"]["classified_documents"],
            "source_documents": data["weekly_summary"]["source_documents"],
            "finance": {k: finance[k] for k in ("primary", "bootstrap", "bootstrap_primary_block_weeks",
                                                 "panels_common_weeks", "years", "crisis_exclusion")},
            "provenance": "Aggregates from frozen experiments; no article bodies or reference text exported."}


def configure_plotting():
    cache = ROOT / "tmp/reporting/matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": "#D4DCE2", "axes.labelcolor": NAVY, "text.color": NAVY,
        "xtick.color": NAVY, "ytick.color": NAVY, "axes.titleweight": "bold",
        "axes.titlepad": 15, "svg.hashsalt": "maritime-research-v1", "pdf.fonttype": 42})


def plot_coverage(data):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, MaxNLocator
    rows = data["coverage"]
    x = list(range(len(rows)))
    fig, (ax, shares) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    fig.subplots_adjust(left=.09, right=.98, top=.9, bottom=.21, hspace=.35)
    fig.suptitle("Observed coverage and maritime event indicators", x=.09, ha="left", fontsize=18, weight="bold")
    ax.bar(x, [r["ft"] for r in rows], color=GRAY, label="Financial Times", width=.72)
    ax.bar(x, [r["guardian"] for r in rows], bottom=[r["ft"] for r in rows], color=TEAL, label="Guardian", width=.72)
    ax.set_ylabel("Observed exact texts")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    ax.legend(loc="upper left", frameon=False, ncol=2)
    shares.plot(x, [r["threat_share"] for r in rows], color=TEAL, linewidth=2.4, label="Threat share")
    shares.plot(x, [r["disruption_share"] for r in rows], color=GOLD, linewidth=2.4, label="Disruption share")
    shares.set_ylim(0, 1.04)
    shares.yaxis.set_major_formatter(PercentFormatter(1))
    shares.set_ylabel("Share of candidates")
    shares.set_xticks([i for i, r in enumerate(rows) if r["quarter"].endswith("Q1")],
                      [r["quarter"][:4] for r in rows if r["quarter"].endswith("Q1")])
    shares.legend(loc="upper left", frameon=False, ncol=2)
    for axis in (ax, shares):
        axis.set_axisbelow(True); axis.grid(axis="y", alpha=.2)
        for year in ("2023", "2024"):
            index = next(i for i, r in enumerate(rows) if r["quarter"] == year + "-Q1")
            axis.axvline(index-.5, color=NAVY, linestyle="--", linewidth=.8, alpha=.6)
        axis.set_xlim(-.7,len(rows)-.3)
    fig.text(.09,.115,"Quarterly display of available full publication weeks. Models use weekly features.\nShares divide summed event counts by summed candidate texts. Threat and disruption can overlap.",fontsize=10)
    fig.text(.09,.045,"Known FT gaps: 2019, 2020 and 2024. Observed counts do not establish complete coverage.\nDashed boundaries: 2023 validation and 2024 final test. Source: weekly_finance_v1.",fontsize=9,color="#526776")
    return fig


def plot_nlp(data):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10.5,5.8))
    fig.subplots_adjust(left=.09,right=.98,top=.85,bottom=.25)
    fig.suptitle("Keyword rules lead on the frozen 2023 NLP holdout",x=.09,ha="left",fontsize=18,weight="bold")
    for j,(method,name,color) in enumerate(zip(METHODS,METHOD_NAMES,(TEAL,GRAY,GOLD))):
        values=[data["nlp"][method]["overall"]["targets"][t]["f1"] for t in TARGETS]
        bars=ax.bar([i+(j-1)*.24 for i in range(3)],values,width=.21,label=name,color=color)
        ax.bar_label(bars,labels=[f"{v:.3f}" for v in values],padding=5,fontsize=10)
    ax.set_xticks(range(3),["Relevance","Threat","Disruption"]);ax.set_ylim(0,1.12)
    ax.set_yticks([0,.25,.5,.75,1]);ax.set_ylabel("Positive-class F1")
    ax.legend(loc="upper center",bbox_to_anchor=(.5,1.16),frameon=False,ncol=3)
    ax.set_axisbelow(True);ax.grid(axis="y",alpha=.2)
    fig.text(.09,.13,"120 reserved articles. Scores measure agreement with AI references.\nKeyword mean F1: 0.803 overall and 0.584 after excluding the largest related-report component.",fontsize=10)
    fig.text(.09,.045,"All methods miss the three remaining disruption positives outside that component.\nSource: nlp_eval_2023_scoring_v1. No thresholds or text rules changed after evaluation.",fontsize=9,color="#526776")
    return fig


def plot_finance(data):
    import matplotlib.pyplot as plt
    f=data["finance"];models=f["primary"]["models"]
    interval=f["bootstrap"][str(f["bootstrap_primary_block_weeks"])]["mae_difference"]
    fig,(ax,ci)=plt.subplots(1,2,figsize=(11,4.8),gridspec_kw={"width_ratios":[1.15,1]})
    fig.subplots_adjust(left=.07,right=.98,top=.76,bottom=.25,wspace=.35)
    fig.suptitle("A small final-test gain with substantial uncertainty",x=.07,ha="left",fontsize=18,weight="bold")
    values=[models[m]["mae"] for m in MODELS]
    bars=ax.bar(MODELS,values,color=[GRAY,"#57748A","#70B8BE",TEAL],width=.62)
    ax.bar_label(bars,labels=[f"{v:.4f}" for v in values],padding=5,fontsize=11)
    ax.set_ylim(0,3.45);ax.set_ylabel("MAE, OVX points (lower is better)")
    ax.set_axisbelow(True);ax.grid(axis="y",alpha=.2)
    ci.axvline(0,color=GRAY,linestyle="--",linewidth=1)
    ci.errorbar([interval["point"]],[0],xerr=[[interval["point"]-interval["lower"]],[interval["upper"]-interval["point"]]],fmt="o",color=TEAL,capsize=7,linewidth=3,markersize=9)
    ci.set_xlim(-.19,.075);ci.set_ylim(-.5,.5);ci.set_yticks([])
    ci.set_xlabel("M3 minus M2 MAE, OVX points")
    ci.set_title("Primary comparison",fontsize=12)
    ci.text(.5,.16,"-0.0608  [95% interval: -0.1490, +0.0227]\nNegative favours M3. The interval includes zero.",transform=ci.transAxes,ha="center",fontsize=9)
    fig.text(.07,.13,"M0 persistence    M1 market history    M2 adds news volume    M3 adds threat/disruption shares",fontsize=9)
    fig.text(.07,.045,"104 common target weeks, 2024-2025. Four-week paired block bootstrap, 10,000 draws.\nM2 and M3 RMSE is almost identical. Source: financial_evaluation_v1; frozen settings, no retuning.",fontsize=9,color="#526776")
    return fig


def save_figure(fig, folder, name):
    import matplotlib.pyplot as plt
    stamp=datetime(2026,11,24,tzinfo=timezone.utc)
    fig.savefig(folder/(name+'.png'),dpi=180,facecolor='white')
    fig.savefig(folder/(name+'.pdf'),metadata={'CreationDate':stamp,'ModDate':stamp},facecolor='white')
    fig.savefig(folder/(name+'.svg'),metadata={'Date':'2026-11-24'},facecolor='white')
    plt.close(fig)


def executive_note(path, data, chart):
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import ParagraphStyle
    c=canvas.Canvas(str(path),pagesize=A4,invariant=1)
    c.setTitle('Maritime news and oil-market uncertainty - executive note')
    c.setAuthor('Maritime news research project')
    width,height=A4;left=44;usable=width-2*left;y=height-46
    def para(text,size=10.2,leading=14,color=NAVY,bold=False,gap=9):
        nonlocal y
        style=ParagraphStyle('note',fontName='Helvetica-Bold' if bold else 'Helvetica',fontSize=size,leading=leading,textColor=HexColor(color))
        p=Paragraph(text,style);_,h=p.wrap(usable,700)
        p.drawOn(c,left,y-h);y-=h+gap
    c.setFillColor(HexColor(TEAL));c.rect(left,y,56,4,fill=1,stroke=0);y-=22
    para('Maritime news and oil-market uncertainty',20,24,bold=True,gap=10)
    para('Executive note  /  24 November 2026',9,12,color='#526776',gap=16)
    para('A modest forecasting gain remains uncertain',14,18,bold=True,gap=8)
    para('Does distinguishing threats from actual disruption improve forecasts of oil-market uncertainty beyond past market conditions and news volume? The study examines the Strait of Hormuz and Red Sea/Bab el-Mandeb corridor. Threats could change supply expectations before operations stop.',10.2,14,gap=10)
    para('The study combines Financial Times and Guardian reports from 2019-2025 with weekly OVX, an options-implied measure of oil-market uncertainty. It fixes the text method and model settings before evaluating 2024-2025 forecasts.',10.2,14,gap=9)
    chart_height=usable*4.8/11
    # Hash image pixels, not the temporary build path, for reproducible PDF objects.
    c.drawImage(ImageReader(str(chart)),left,y-chart_height,width=usable,height=chart_height,mask='auto');y-=chart_height+13
    para('Three findings',12,16,bold=True,gap=7)
    para('<b>1. Small error reduction.</b> Adding threat/disruption shares lowers mean absolute error by <b>2.2%</b> relative to the news-volume model over 104 test weeks.',10.2,14,gap=7)
    para('<b>2. Uncertain incremental value.</b> The MAE difference is <b>-0.0608 OVX points</b>, with a 95% paired block-bootstrap interval of <b>[-0.1490, +0.0227]</b>. The interval includes zero and large-error performance is essentially unchanged.',10.2,14,gap=7)
    para('<b>3. A fragile measurement layer.</b> Keyword mean F1 is <b>0.803</b> against AI references, falling to <b>0.584</b> after removing the largest related-report component. All methods miss the three remaining disruption positives.',10.2,14,gap=12)
    para('Interpretation and next experiment',12,16,bold=True,gap=7)
    para('The results support further measurement work, but do not establish a reliable forecasting advantage or a trading application. Historical article versions and availability remain unverified, FT coverage has gaps, and only two test years constrain inference. Improve independently reviewed disruption labels and evaluate a newly frozen method on genuinely new observations.',10.2,14,gap=10)
    if y<48:raise ValueError(f'Executive note exceeds one-page content area: {y}')
    c.setStrokeColor(HexColor('#D4DCE2'));c.line(left,37,width-left,37)
    c.setFillColor(HexColor('#526776'));c.setFont('Helvetica',7)
    c.drawString(left,25,'Sources: NLP_EVALUATION.md and FINANCIAL_EVALUATION.md. AI-assisted implementation and working labels.')
    c.showPage();c.save()


def build(destination, root=ROOT):
    policy,raw,inputs=load_inputs(root)
    data=report_data(raw)
    configure_plotting()
    destination.mkdir(parents=True,exist_ok=True)
    figures=destination/'figures';figures.mkdir()
    for name,plot in [('coverage_indicators',plot_coverage),('nlp_comparison',plot_nlp),('forecast_comparison',plot_finance)]:
        save_figure(plot(data),figures,name)
    save_json(destination/'exhibit_data.json',data)
    executive_note(destination/'executive_note.pdf',data,figures/'forecast_comparison.png')
    inputs.update({name:digest(root/name) for name in ['scripts/build_research_package.py','requirements-reporting.txt']})
    return inputs


def run(root=ROOT,validate=False):
    policy=json.loads((root/POLICY).read_text());output=root/policy['output_directory']
    if not output.resolve().is_relative_to((root/'output').resolve()):raise ValueError('Unsafe reporting output')
    if output.exists() and not validate:raise ValueError('Research package exists; use --validate or a new output version')
    runtime={k:version(k) for k in ('matplotlib','reportlab','pypdf','pymupdf','numpy')}
    output.parent.mkdir(exist_ok=True,parents=True)
    with tempfile.TemporaryDirectory(prefix='.research_',dir=output.parent) as tmp:
        stage=Path(tmp);inputs=build(stage,root)
        artifacts={str(p.relative_to(stage)):digest(p) for p in sorted(stage.rglob('*')) if p.is_file()}
        verify_hashes(root,inputs)
        manifest={'version':policy['version'],'input_sha256':inputs,'artifact_sha256':artifacts,'runtime':runtime}
        if validate:
            saved=json.loads((output/'manifest.json').read_text())
            verify_hashes(output,saved['artifact_sha256'])
            if saved!=manifest:raise ValueError('Research package does not reproduce exactly')
        else:
            save_json(stage/'manifest.json',manifest);output.mkdir()
            for p in stage.iterdir():p.rename(output/p.name)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--validate',action='store_true')
    print(run(validate=parser.parse_args().validate))
