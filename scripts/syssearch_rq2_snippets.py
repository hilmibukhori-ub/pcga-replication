"""Print evidence snippets per RQ2 field from extracted full text (aid for manual coding)."""
import re, sys, pathlib
PAT = {
 "norel": r"(LSTM|GRU|ALSTM|Rank_LSTM|Transformer|MLP)[^.]{0,80}(baseline|without|no relation|do(es)? not)|without (the )?(graph|relation)|w/o\.? ?(graph|relation|GNN|GAT|GCN|hyper)|remov\w+ (the )?(graph|relation)",
 "rand": r"random(ly)? (graph|relation|edge|adjacen|connect)|shuffl\w*|permut\w* (the )?(graph|relation|edge|node)|fully[- ]connected graph|complete graph|identity (matrix|graph)|placebo",
 "runs": r"(\d+|five|ten|three|twenty|fifteen) (independent )?(runs|times|trials|random seeds|seeds)|averag\w+ over|mean and standard deviation|± ?\d|std",
 "sig": r"significan\w+ (test|level)|t-test|p[- ]?value|wilcoxon|p ?< ?0\.\d|statistically significant|diebold|bootstrap",
 "val": r"validation (set|period|data|split)|valid(ation)? set|early stop\w*|selected (based )?on (the )?valid|dev(elopment)? set",
 "code": r"github\.com/\S+|code (is|are) (publicly )?available|available at",
}
for f in sys.argv[1:]:
    t = " ".join(pathlib.Path(f).read_text(errors="ignore").split())
    print(f"\n######## {f}  ({len(t)} chars)")
    for k, p in PAT.items():
        ms = list(re.finditer(p, t, re.I))
        snips = []
        for m in ms[:4]:
            snips.append(t[max(0, m.start()-90):m.end()+90])
        print(f"[{k}] n={len(ms)} :: " + " || ".join(snips))
