"""Write an interactive reply network to reports/network.html.

Open that file in a browser. Search or click an account to see only that account
and the accounts it exchanged replies with. Edge numbers are reply counts.
The graph uses reply edges from data/processed/interaction_edges.csv.

Example:

    python src/network_view.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd

SCORE_COLUMNS = {
    "new_account_7d",
    "new_account_30d",
    "median_interarrival_min",
    "max_subreddit_share",
    "activity_count",
}


def normalize_edges(frame):
    columns = set(frame.columns)
    if {"author", "parent_author", "weight"}.issubset(columns):
        edges = frame.rename(columns={"author": "source", "parent_author": "target"})
    elif {"source", "target", "weight"}.issubset(columns):
        edges = frame
    else:
        raise ValueError("Edges need source/target/weight or author/parent_author/weight columns.")
    edges = edges[["source", "target", "weight"]].copy()
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce").fillna(0)
    edges = edges[(edges["source"] != "") & (edges["target"] != "") & (edges["source"] != edges["target"])]
    return edges.groupby(["source", "target"], as_index=False)["weight"].sum()


def select_graph(edges, min_weight, max_nodes):
    """Keep edges at or above min_weight, then the highest-degree accounts."""
    kept = normalize_edges(edges)
    kept = kept[kept["weight"] >= min_weight]
    if kept.empty or max_nodes <= 0:
        return kept.iloc[0:0]
    degree = pd.concat([kept["source"], kept["target"]], ignore_index=True).value_counts()
    keep_nodes = set(degree.head(max_nodes).index)
    return kept[kept["source"].isin(keep_nodes) & kept["target"].isin(keep_nodes)].reset_index(drop=True)


def load_features(path):
    path = Path(path)
    if not path.exists():
        return None
    features = pd.read_csv(path, index_col=0)
    if SCORE_COLUMNS.issubset(features.columns):
        from report import coordination_frame
        features = coordination_frame(features)
    return features


def _feature_row(features, account):
    if features is None or account not in features.index:
        return None
    return features.loc[account]


def node_color(row):
    if row is None:
        return "#3d6f8f"
    new7 = row.get("new_account_7d") if hasattr(row, "get") else None
    if bool(new7) and pd.notna(new7):
        return "#d85a30"
    score = row.get("coordination_score") if hasattr(row, "get") else None
    if score is not None and pd.notna(score) and float(score) >= 0.5:
        return "#e0a106"
    return "#3d6f8f"


def node_tip(account, row, degree):
    lines = [str(account), f"connections: {degree}"]
    if row is None:
        return "\n".join(lines)
    labels = (
        ("activity", "activity_count", ".0f"),
        ("signal score", "coordination_score", ".3f"),
        ("new ≤7d", "new_account_7d", None),
        ("new ≤30d", "new_account_30d", None),
        ("max subreddit share", "max_subreddit_share", ".2f"),
    )
    for label, column, spec in labels:
        if column not in row.index:
            continue
        value = row[column]
        if pd.isna(value):
            continue
        shown = format(float(value), spec) if spec and not isinstance(value, (bool, str)) else value
        lines.append(f"{label}: {shown}")
    return "\n".join(lines)


def graph_payload(edges, features):
    degree = pd.concat([edges["source"], edges["target"]], ignore_index=True).value_counts()
    accounts = list(degree.index)
    index = {account: i for i, account in enumerate(accounts)}
    nodes = []
    for account in accounts:
        row = _feature_row(features, account)
        connections = int(degree[account])
        nodes.append({
            "id": account,
            "degree": connections,
            "r": min(22, 6 + connections),
            "color": node_color(row),
            "tip": node_tip(account, row, connections),
        })
    links = [
        {"s": index[row.source], "t": index[row.target], "w": float(row.weight)}
        for row in edges.itertuples(index=False)
    ]
    return {"nodes": nodes, "links": links}


def render_network_html(payload):
    blob = json.dumps(payload).replace("<", "\\u003c")
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Reply network</title>
<style>
body {{margin:0;font-family:Arial,sans-serif;background:#f4f1ea;color:#1c1a17}}
header {{display:flex;gap:16px;align-items:center;flex-wrap:wrap;padding:10px 14px;background:#fff;border-bottom:1px solid #ddd}}
.layout {{display:grid;grid-template-columns:300px 1fr;height:calc(100vh - 58px)}}
aside {{overflow:auto;background:#fff;border-right:1px solid #ddd;padding:10px}}
input {{font:inherit;padding:6px 8px;width:100%;box-sizing:border-box}}
button.acct {{display:block;width:100%;text-align:left;font:inherit;border:0;background:transparent;padding:6px 4px;cursor:pointer}}
button.acct.on {{background:#efe6d6}}
main {{overflow:auto;padding:16px 20px 32px}}
svg {{width:100%;height:auto;max-width:920px;display:block}}
table {{border-collapse:collapse;width:100%;max-width:720px;font-size:14px}}
th, td {{text-align:left;padding:6px 8px;border-bottom:1px solid #e2dcd2}}
.legend i {{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}}
.muted {{color:#5c564e;font-size:13px}}
</style></head>
<body>
<header>
  <strong>Reply network</strong>
  <span id="count"></span>
  <span class="legend"><i style="background:#d85a30"></i>first seen ≤7d</span>
  <span class="legend"><i style="background:#e0a106"></i>signal score ≥ 0.5</span>
  <span class="legend"><i style="background:#3d6f8f"></i>other accounts</span>
</header>
<div class="layout">
  <aside>
    <input id="q" placeholder="Find an account" autocomplete="off">
    <p class="muted">Click an account. The drawing shows only that account and who it replied with.</p>
    <div id="list"></div>
  </aside>
  <main>
    <h2 id="who"></h2>
    <p id="detail" class="muted"></p>
    <svg id="g" viewBox="0 0 900 560"></svg>
    <h3>Replied to</h3>
    <table id="pairs"><thead><tr><th>Account</th><th>Direction</th><th>Replies</th></tr></thead><tbody></tbody></table>
  </main>
</div>
<script>
const data = {blob};
const nodes = data.nodes;
const links = data.links;
document.getElementById("count").textContent = nodes.length + " accounts, " + links.length + " replies";
function esc(s) {{
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}}
const adj = nodes.map(() => []);
links.forEach(e => {{
  adj[e.s].push({{other: e.t, w: e.w, dir: "replied to"}});
  adj[e.t].push({{other: e.s, w: e.w, dir: "replied from"}});
}});
let focus = 0;
const list = document.getElementById("list");
const svg = document.getElementById("g");
function neighbors(i) {{
  return adj[i].slice().sort((a, b) => b.w - a.w || nodes[a.other].id.localeCompare(nodes[b.other].id));
}}
function drawList() {{
  const q = document.getElementById("q").value.trim().toLowerCase();
  list.replaceChildren();
  nodes.forEach((n, i) => {{
    if (q && !n.id.toLowerCase().includes(q)) return;
    const b = document.createElement("button");
    b.className = "acct" + (i === focus ? " on" : "");
    b.textContent = n.id + "  ·  " + n.degree;
    b.onclick = () => {{ focus = i; draw(); }};
    list.appendChild(b);
  }});
}}
function draw() {{
  const n = nodes[focus];
  const rows = neighbors(focus);
  const shown = rows.slice(0, 12);
  document.getElementById("who").textContent = n.id;
  document.getElementById("detail").textContent = n.tip.replaceAll("\\n", " · ")
    + (rows.length > shown.length ? " · drawing " + shown.length + " of " + rows.length + " links" : "");
  const cx = 450, cy = 270, radius = shown.length <= 1 ? 0 : 210;
  const parts = [];
  shown.forEach((edge, k) => {{
    const angle = shown.length === 1 ? -Math.PI / 2 : (-Math.PI / 2) + k * 2 * Math.PI / shown.length;
    const x = cx + Math.cos(angle) * radius;
    const y = cy + Math.sin(angle) * radius;
    const other = nodes[edge.other];
    parts.push('<line x1="'+cx+'" y1="'+cy+'" x2="'+x+'" y2="'+y+'" stroke="#6d6458" stroke-width="'+(1+Math.min(6, Math.sqrt(edge.w)))+'" />');
    const mx = (cx + x) / 2, my = (cy + y) / 2;
    parts.push('<text x="'+mx+'" y="'+my+'" text-anchor="middle" font-size="12" fill="#1c1a17">'+edge.w+'</text>');
    parts.push('<g data-i="'+edge.other+'" style="cursor:pointer">');
    parts.push('<circle cx="'+x+'" cy="'+y+'" r="16" fill="'+other.color+'" />');
    const anchor = x >= cx ? "start" : "end";
    const tx = x + (x >= cx ? 20 : -20);
    parts.push('<text x="'+tx+'" y="'+(y+4)+'" text-anchor="'+anchor+'" font-size="13" fill="#1c1a17">'+esc(other.id)+'</text>');
    parts.push("</g>");
  }});
  parts.push('<circle cx="'+cx+'" cy="'+cy+'" r="22" fill="'+n.color+'" />');
  parts.push('<text x="'+cx+'" y="'+(cy-32)+'" text-anchor="middle" font-size="14" font-weight="700" fill="#1c1a17">'+esc(n.id)+'</text>');
  svg.innerHTML = parts.join("");
  svg.querySelectorAll("g[data-i]").forEach(g => {{
    g.onclick = () => {{ focus = Number(g.dataset.i); draw(); }};
  }});
  const body = document.querySelector("#pairs tbody");
  body.replaceChildren();
  rows.forEach(edge => {{
    const tr = document.createElement("tr");
    const other = nodes[edge.other];
    tr.innerHTML = "<td></td><td>"+edge.dir+"</td><td>"+edge.w+"</td>";
    const cell = tr.firstChild;
    const jump = document.createElement("button");
    jump.className = "acct";
    jump.textContent = other.id;
    jump.onclick = () => {{ focus = edge.other; draw(); }};
    cell.appendChild(jump);
    body.appendChild(tr);
  }});
  drawList();
}}
document.getElementById("q").addEventListener("input", drawList);
draw();
</script>
</body></html>
"""


def build_network(edges, features, min_weight, max_nodes):
    selected = select_graph(edges, min_weight, max_nodes)
    payload = graph_payload(selected, features) if len(selected) else {"nodes": [], "links": []}
    return render_network_html(payload), selected


def main():
    args = parse_args()
    edge_path = Path(args.edges)
    if not edge_path.exists():
        raise FileNotFoundError(f"{edge_path} is missing. Run src/features.py first.")
    edges = pd.read_csv(edge_path)
    features = load_features(args.features)
    html, selected = build_network(edges, features, args.min_weight, args.max_nodes)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    accounts = set(selected["source"]).union(selected["target"]) if len(selected) else set()
    print(f"Wrote {out} ({len(accounts)} accounts, {len(selected)} replies)")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", default="data/processed/interaction_edges.csv")
    parser.add_argument("--features", default="data/processed/account_features.csv")
    parser.add_argument("--out", default="reports/network.html")
    parser.add_argument("--min-weight", type=float, default=2)
    parser.add_argument("--max-nodes", type=int, default=80)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
