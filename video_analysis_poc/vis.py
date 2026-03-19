# ── Visualize the Collision Knowledge Graph ────────────────────────
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load the exported GraphML (or just use G directly)
G_viz = G.copy()

# Define colour and shape mapping by node type
_NODE_COLORS = {
    "collision_pattern": "#E74C3C",  # red
    "vehicle_type": "#3498DB",  # blue
    "road_geometry": "#2ECC71",  # green
    "regulation": "#F39C12",  # amber
}
_EDGE_COLORS = {
    "INVOLVES": "#888888",
    "OCCURS_ON": "#2ECC71",
    "APPLIES_TO": "#F39C12",
    "OFTEN_FOLLOWED_BY": "#E74C3C",
}

# Build layout — shell layout groups nodes by type for readability
shells = [
    [n for n, d in G_viz.nodes(data=True) if d.get("node_type") == "collision_pattern"],
    [
        n
        for n, d in G_viz.nodes(data=True)
        if d.get("node_type") in ("vehicle_type", "road_geometry")
    ],
    [n for n, d in G_viz.nodes(data=True) if d.get("node_type") == "regulation"],
]
shells = [s for s in shells if s]  # drop empty shells
pos = nx.shell_layout(G_viz, nlist=shells, rotate=0.2)

# Node properties
node_colors = [
    _NODE_COLORS.get(G_viz.nodes[n].get("node_type", ""), "#CCCCCC")
    for n in G_viz.nodes()
]
node_sizes = [
    2200 if G_viz.nodes[n].get("node_type") == "collision_pattern" else 1400
    for n in G_viz.nodes()
]

# Short labels: strip prefix (e.g. "pattern:", "vtype:")
labels = {}
for n, d in G_viz.nodes(data=True):
    if d.get("node_type") == "collision_pattern":
        labels[n] = d.get("pattern_name", n.split(":")[-1])
    elif d.get("node_type") == "vehicle_type":
        labels[n] = d.get("label", n.split(":")[-1])
    elif d.get("node_type") == "regulation":
        labels[n] = (
            d.get("reg_id", n.split(":")[-1])
            .replace("uk_", "")
            .replace("_", " ")
            .title()
        )
    else:
        labels[n] = n.split(":")[-1].replace("_", " ").title()

# Edge colours
edge_colors = [
    _EDGE_COLORS.get(G_viz.edges[e].get("relation", ""), "#AAAAAA")
    for e in G_viz.edges()
]

# Draw
fig, ax = plt.subplots(1, 1, figsize=(18, 12), facecolor="white")

nx.draw_networkx_nodes(
    G_viz,
    pos,
    ax=ax,
    node_color=node_colors,
    node_size=node_sizes,
    alpha=0.9,
    edgecolors="white",
    linewidths=1.5,
)
nx.draw_networkx_labels(
    G_viz, pos, labels=labels, ax=ax, font_size=7, font_weight="bold"
)
nx.draw_networkx_edges(
    G_viz,
    pos,
    ax=ax,
    edge_color=edge_colors,
    arrows=True,
    arrowsize=14,
    width=1.5,
    alpha=0.7,
    connectionstyle="arc3,rad=0.08",
)

# Edge relation labels
edge_labels = {(u, v): d.get("relation", "") for u, v, d in G_viz.edges(data=True)}
nx.draw_networkx_edge_labels(
    G_viz,
    pos,
    edge_labels=edge_labels,
    ax=ax,
    font_size=5.5,
    font_color="#555555",
    rotate=False,
)

# Legend
legend_handles = [
    mpatches.Patch(color=c, label=t.replace("_", " ").title())
    for t, c in _NODE_COLORS.items()
]
legend_handles += [
    plt.Line2D([0], [0], color=c, linewidth=2, label=r.replace("_", " "))
    for r, c in _EDGE_COLORS.items()
]
ax.legend(
    handles=legend_handles,
    loc="upper left",
    fontsize=8,
    title="Node / Edge Types",
    title_fontsize=9,
    framealpha=0.9,
)

ax.set_title(
    "Traffic Collision Knowledge Graph\n"
    f"({G_viz.number_of_nodes()} nodes, {G_viz.number_of_edges()} edges)",
    fontsize=14,
    fontweight="bold",
    pad=16,
)
ax.axis("off")
plt.tight_layout()
plt.savefig(
    "./collision_kg_visualisation.png", dpi=150, bbox_inches="tight", facecolor="white"
)
plt.show()

print(f"Visualisation saved: ./collision_kg_visualisation.png")
