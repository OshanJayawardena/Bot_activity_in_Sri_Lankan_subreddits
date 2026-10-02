"""Write an interactive reply network to reports/network.html.

Open that file in a browser. Drag nodes, scroll to zoom, and search for an account.
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
header {{display:flex;gap:16px;align-items:center;padding:10px 14px;background:#fff;border-bottom:1px solid #ddd}}
input {{font:inherit;padding:6px 8px;min-width:240px}}
#stage {{position:relative;height:calc(100vh - 58px)}}
canvas {{width:100%;height:100%;display:block;cursor:grab}}
#tip {{position:absolute;pointer-events:none;background:#1c1a17;color:#fff;padding:8px 10px;border-radius:6px;white-space:pre;font-size:12px;display:none}}
.legend i {{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}}
</style></head>
<body>
<header>
  <strong>Reply network</strong>
  <input id="q" placeholder="Find an account" autocomplete="off">
  <span id="count"></span>
  <span class="legend"><i style="background:#d85a30"></i>first seen ≤7d</span>
  <span class="legend"><i style="background:#e0a106"></i>signal score ≥ 0.5</span>
  <span class="legend"><i style="background:#3d6f8f"></i>other accounts</span>
</header>
<div id="stage"><canvas id="c"></canvas><div id="tip"></div></div>
<script>
const data = {blob};
const canvas = document.getElementById("c");
const tip = document.getElementById("tip");
const ctx = canvas.getContext("2d");
const nodes = data.nodes.map((n, i) => ({{
  ...n, i,
  x: Math.cos(i) * 280,
  y: Math.sin(i * 2.3) * 280,
  vx: 0, vy: 0
}}));
const links = data.links;
document.getElementById("count").textContent = nodes.length + " accounts, " + links.length + " replies";
let scale = 1, ox = 0, oy = 0, drag = null, hover = null, selected = null;
function resize() {{
  canvas.width = canvas.clientWidth * devicePixelRatio;
  canvas.height = canvas.clientHeight * devicePixelRatio;
}}
resize();
addEventListener("resize", resize);
function world(ev) {{
  const r = canvas.getBoundingClientRect();
  return [(ev.clientX - r.left - ox) / scale, (ev.clientY - r.top - oy) / scale];
}}
function nodeAt(x, y) {{
  for (let i = nodes.length - 1; i >= 0; i--) {{
    const n = nodes[i];
    const dx = n.x - x, dy = n.y - y;
    if (dx * dx + dy * dy <= (n.r + 3) * (n.r + 3)) return n;
  }}
  return null;
}}
canvas.addEventListener("pointerdown", ev => {{
  const [x, y] = world(ev);
  const n = nodeAt(x, y);
  if (n) {{ drag = n; selected = n; }}
  else {{ drag = "pan"; selected = null; canvas._px = ev.clientX; canvas._py = ev.clientY; }}
  canvas.setPointerCapture(ev.pointerId);
}});
canvas.addEventListener("pointermove", ev => {{
  const [x, y] = world(ev);
  if (drag && drag !== "pan") {{ drag.x = x; drag.y = y; drag.vx = 0; drag.vy = 0; }}
  else if (drag === "pan") {{
    ox += ev.clientX - canvas._px; oy += ev.clientY - canvas._py;
    canvas._px = ev.clientX; canvas._py = ev.clientY;
  }}
  hover = nodeAt(x, y);
  const r = canvas.getBoundingClientRect();
  if (hover) {{
    tip.style.display = "block";
    tip.style.left = (ev.clientX - r.left + 12) + "px";
    tip.style.top = (ev.clientY - r.top + 12) + "px";
    tip.textContent = hover.tip;
  }} else tip.style.display = "none";
}});
canvas.addEventListener("pointerup", () => {{ drag = null; }});
canvas.addEventListener("wheel", ev => {{
  ev.preventDefault();
  const factor = ev.deltaY < 0 ? 1.08 : 0.92;
  const r = canvas.getBoundingClientRect();
  const mx = ev.clientX - r.left, my = ev.clientY - r.top;
  ox = mx - (mx - ox) * factor;
  oy = my - (my - oy) * factor;
  scale *= factor;
}}, {{passive: false}});
document.getElementById("q").addEventListener("input", ev => {{
  const q = ev.target.value.trim().toLowerCase();
  selected = q ? nodes.find(n => n.id.toLowerCase().includes(q)) || null : null;
  if (selected) {{
    const r = canvas.getBoundingClientRect();
    ox = r.width / 2 - selected.x * scale;
    oy = r.height / 2 - selected.y * scale;
  }}
}});
function tick() {{
  const n = nodes.length;
  for (let i = 0; i < n; i++) {{
    for (let j = i + 1; j < n; j++) {{
      let dx = nodes[j].x - nodes[i].x, dy = nodes[j].y - nodes[i].y;
      let d2 = dx * dx + dy * dy || 0.01;
      let f = 180 / d2;
      let d = Math.sqrt(d2);
      dx /= d; dy /= d;
      nodes[i].vx -= dx * f; nodes[i].vy -= dy * f;
      nodes[j].vx += dx * f; nodes[j].vy += dy * f;
    }}
    nodes[i].vx += -nodes[i].x * 0.002;
    nodes[i].vy += -nodes[i].y * 0.002;
  }}
  for (const e of links) {{
    const a = nodes[e.s], b = nodes[e.t];
    let dx = b.x - a.x, dy = b.y - a.y;
    a.vx += dx * 0.01; a.vy += dy * 0.01;
    b.vx -= dx * 0.01; b.vy -= dy * 0.01;
  }}
  if (drag && drag !== "pan") {{ drag.vx = 0; drag.vy = 0; }}
  for (const node of nodes) {{
    if (node === drag) continue;
    node.x += node.vx; node.y += node.vy;
    node.vx *= 0.72; node.vy *= 0.72;
  }}
}}
function draw() {{
  const w = canvas.width, h = canvas.height;
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  ctx.save();
  ctx.translate(ox, oy);
  ctx.scale(scale, scale);
  const focus = selected || hover;
  const near = new Set();
  if (focus) {{
    near.add(focus.i);
    for (const e of links) {{
      if (e.s === focus.i) near.add(e.t);
      if (e.t === focus.i) near.add(e.s);
    }}
  }}
  for (const e of links) {{
    const a = nodes[e.s], b = nodes[e.t];
    const dim = focus && !(near.has(e.s) && near.has(e.t));
    ctx.strokeStyle = dim ? "rgba(80,70,60,0.08)" : "rgba(60,50,40,0.45)";
    ctx.lineWidth = Math.min(5, 0.6 + Math.sqrt(e.w));
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  }}
  for (const node of nodes) {{
    const dim = focus && !near.has(node.i);
    ctx.globalAlpha = dim ? 0.15 : 1;
    ctx.fillStyle = node.color;
    ctx.beginPath(); ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2); ctx.fill();
    if (!dim && (scale > 1.15 || node === focus || node.r > 12)) {{
      ctx.fillStyle = "#1c1a17";
      ctx.font = "11px Arial";
      ctx.fillText(node.id, node.x + node.r + 2, node.y + 3);
    }}
  }}
  ctx.restore();
  requestAnimationFrame(() => {{ tick(); draw(); }});
}}
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
