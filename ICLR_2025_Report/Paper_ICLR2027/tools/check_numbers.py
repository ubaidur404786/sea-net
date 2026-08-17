"""
Check every number printed in the paper's main tables against the CSV it came
from. Run this before submitting -- it is the mechanical half of the promise
made in the AI use statement ("every numeric value was traced to the specific
result file and column it came from").

    python ICLR_2025_Report/Paper_ICLR2027/tools/check_numbers.py

Prints one line per value and exits non-zero if anything disagrees.
"""
import csv
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parent.parent.parent.parent          # .../SEA_NET
LB = ROOT / "results" / "SEA_NET" / "leaderboard.csv"
PF = ROOT / "results" / "SEA_NET" / "profile.csv"

lb = {r["config"]: r for r in csv.DictReader(LB.open(newline="", encoding="utf-8"))}
# profile.csv keys on the long model id, so index it by the config's last part
pf = {}
for r in csv.DictReader(PF.open(newline="", encoding="utf-8")):
    pf[r["config"].split("/")[-1]] = r

fails = []


def check(label, printed, actual, decimals):
    """printed = what the .tex says; actual = the raw CSV string."""
    ok = round(float(actual), decimals) == printed
    print(f"  {'OK ' if ok else 'BAD'}  {label:<46} paper={printed}  csv={actual}")
    if not ok:
        fails.append(label)


print("Table 1 (tab:main) -- WebTraffic acc / AOPCR / NDCG / UCR-85 acc")
MAIN = {
    # config              acc     aopcr    ndcg    ucr85
    "fcn":                    (0.742,  3.83,  0.533, 0.814),
    "resnet":                 (0.772,  2.95,  0.554, 0.815),
    "millet":                 (0.887,  2.57,  0.677, 0.827),
    "millet_paper":           (0.920, 13.27,  0.661, 0.843),
    "seanet_classwise":       (0.946,  1.72,  0.686, 0.829),
    "seanet_gated_mean_topk": (0.955,  2.23,  0.750, 0.808),
    # NDCG is exactly 0.7725 -- a rounding tie. We print 0.772, i.e. we never
    # round a tie in the direction that flatters us.
    "seanet_bottleneck_topk": (0.947,  2.62,  0.772, 0.810),
}
for cfg, (acc, aopcr, ndcg, ucr) in MAIN.items():
    r = lb[cfg]
    check(f"{cfg} web_acc",   acc,   r["web_acc"],   3)
    check(f"{cfg} web_aopcr", aopcr, r["web_aopcr"], 2)
    check(f"{cfg} web_ndcg",  ndcg,  r["web_ndcg"],  3)
    check(f"{cfg} ucr85_acc", ucr,   r["ucr85_acc"], 3)

print("\nTable 2 (tab:cost) -- params / FLOPs / latency / peak memory")
COST = {
    # config                        params   mflops  lat_ms  mem_mb
    "millet":                     (423707,  847.6, 0.203, 112.3),
    "seanet_classwise":           (269164,  523.7, 0.359, 123.7),
    "seanet_gated_mean_topk":     ( 61740,  109.7, 0.128,  64.9),
    "seanet_bottleneck_topk":     ( 41324,   76.7, 0.131, 179.0),
    "seanet_bottleneck_shallow":  ( 21548,   40.0, 0.069, 178.9),
}
for cfg, (params, mflops, lat, mem) in COST.items():
    r = pf[cfg]
    check(f"{cfg} params",   params, r["params"],      0)
    check(f"{cfg} flops_m",  mflops, r["flops_m"],     1)
    check(f"{cfg} infer_ms", lat,    r["infer_ms"],    3)
    check(f"{cfg} peak_mem", mem,    r["peak_mem_mb"], 1)
    # weight memory is COMPUTED, not measured -- verify the arithmetic too
    kb32 = round(int(r["params"]) * 4 / 1024)
    print(f"       weights fp32 = {kb32} KB, int8 = {round(kb32/4)} KB (computed)")

print("\nTable 3 left (tab:ablation) -- all seed 0")
ABL = {
    "seanet_slim":          (0.916, 2.48, 0.719, 57580),
    "seanet_slim_topk":     (0.932, 2.69, 0.778, 57580),
    "seanet_bottleneck":    (0.902, 2.57, 0.733, 41324),
    # seanet_bottleneck_topk seed 0 is NOT in leaderboard.csv (that holds the
    # 3-seed mean); it is checked against results.csv below.
}
for cfg, (acc, aopcr, ndcg, params) in ABL.items():
    r = lb[cfg]
    check(f"{cfg} web_acc",   acc,    r["web_acc"],   3)
    check(f"{cfg} web_aopcr", aopcr,  r["web_aopcr"], 2)
    check(f"{cfg} web_ndcg",  ndcg,   r["web_ndcg"],  3)
    check(f"{cfg} params",    params, r["params"],    0)

print("\nTable 3 right (kappa sweep) -- all seed 0")
KAPPA = {
    "seanet_topk_k005":    (0.888, 2.29, 0.715),
    "seanet_topk_k025":    (0.940, 2.65, 0.733),
    "seanet_topk_k050":    (0.882, 3.11, 0.732),
    "seanet_topk_k100":    (0.908, 2.44, 0.722),
    "seanet_topk_nofocus": (0.950, 2.30, 0.765),
}
for cfg, (acc, aopcr, ndcg) in KAPPA.items():
    r = lb[cfg]
    check(f"{cfg} web_acc",   acc,   r["web_acc"],   3)
    check(f"{cfg} web_aopcr", aopcr, r["web_aopcr"], 2)
    check(f"{cfg} web_ndcg",  ndcg,  r["web_ndcg"],  3)

print("\nSeed-0 row of seanet_bottleneck_topk (kappa = 0.10, and Table 3 bottom-left)")
rc = ROOT / "results" / "SEA_NET" / (
    "seanet_bottleneck_topk__sea_mstcn_sep_bottleneck__sea_topk_conjunctive" ) / "results.csv"
seed0 = [r for r in csv.DictReader(rc.open(newline="", encoding="utf-8"))
         if r["dataset"] == "WebTraffic" and r["seed"] == "0"]
if not seed0:
    fails.append("seed-0 row not found")
    print("  BAD  could not find the WebTraffic seed-0 row")
else:
    r = seed0[0]
    check("bottleneck_topk seed0 acc",   0.938, r["test_acc"],   3)
    check("bottleneck_topk seed0 aopcr", 2.78,  r["test_aopcr"], 2)
    check("bottleneck_topk seed0 ndcg",  0.777, r["test_ndcg"],  3)

print("\nInterpretability cost (Section 3.1 / 6): head-only difference")
gap = int(pf["millet"]["params"]) - int(pf["conventional"]["params"])
share = 100 * gap / int(pf["millet"]["params"])
print(f"  {'OK ' if gap == 1041 else 'BAD'}  conjunctive head costs {gap} params "
      f"= {share:.2f} % (paper says 1,041 and 0.25 %)")
if gap != 1041:
    fails.append("head parameter cost")

print()
if fails:
    print(f"FAILED: {len(fails)} value(s) disagree -> {fails}")
    sys.exit(1)
print("All values match their source files.")
