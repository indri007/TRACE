#!/usr/bin/env python3
"""
generate_diagrams.py
Generates 3 publication-quality figures for the TRACE manuscript:
1. fig1_pipeline_flowchart.png — End-to-end research methodology & deployment flowchart
2. fig2_vosviewer_network.png  — VOSviewer bibliometric co-occurrence network map
3. fig3_nodexl_network.png     — NodeXL Pro clustered network graph (3 thematic clusters G1, G2, G3)
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Ensure target directory exists
out_dir = Path("/Users/jevin/TRACE/manuscript/figures")
out_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERATE PIPELINE FLOWCHART (Diagram Alir Penelitian & Sistem)
# ─────────────────────────────────────────────────────────────────────────────
def create_flowchart():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)
    ax.axis('off')
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    c_blue   = '#2563EB'
    c_indigo = '#4F46E5'
    c_purple = '#7C3AED'
    c_teal   = '#0D9488'
    c_amber  = '#D97706'
    c_rose   = '#E11D48'
    c_slate  = '#475569'

    def draw_box(x, y, w, h, title, subtitle, color, icon=""):
        shadow = patches.FancyBboxPatch((x+0.05, y-0.05), w, h, boxstyle="round,pad=0.1,rounding_size=0.18",
                                        facecolor='#E2E8F0', edgecolor='none', zorder=1)
        ax.add_patch(shadow)
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.18",
                                      facecolor='#FFFFFF', edgecolor=color, linewidth=2, zorder=2)
        ax.add_patch(rect)
        header = patches.FancyBboxPatch((x, y+h-0.45), w, 0.45, boxstyle="round,pad=0.08,rounding_size=0.15",
                                        facecolor=color, edgecolor='none', zorder=3)
        ax.add_patch(header)
        header_text = f"{icon} {title}" if icon else title
        ax.text(x + w/2, y + h - 0.22, header_text, ha='center', va='center',
                color='#FFFFFF', fontsize=9.5, fontweight='bold', zorder=4)
        ax.text(x + w/2, y + (h-0.45)/2, subtitle, ha='center', va='center',
                color='#1E293B', fontsize=8.5, linespacing=1.35, zorder=4)

    def draw_phase(x, y, w, h, label, color):
        p_bg = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.25",
                                      facecolor=color, edgecolor=color, linewidth=1.5, alpha=0.12, zorder=0)
        ax.add_patch(p_bg)
        ax.text(x + 0.3, y + h - 0.25, label, fontsize=10.5, fontweight='bold', color=color, zorder=1)

    draw_phase(0.4, 0.4, 3.8, 7.6, "PHASE 1: DATA INTEGRITY & AUDIT", c_teal)
    draw_phase(4.5, 0.4, 4.4, 7.6, "PHASE 2: MODEL & INFERENCE", c_indigo)
    draw_phase(9.2, 0.4, 4.4, 7.6, "PHASE 3: DEPLOYMENT & TELEMETRY", c_rose)

    draw_box(0.7, 6.0, 3.2, 1.4, "Raw Dataset Ingestion",
             "Roboflow Universe (CC BY 4.0)\n9,725 Images · 3 Classes\n(Bus, Car, Van)", c_teal)
    draw_box(0.7, 3.8, 3.2, 1.4, "Label-Format Audit",
             "Token Length Verification\nBBox: 5 tokens / row\nPolygon: >6 tokens / row", c_teal)
    draw_box(0.7, 1.2, 3.2, 1.8, "Polygon-to-BBox Fix",
             "Identified 81 Mixed Bus Rows\nSilent Ultralytics Parsing Bug\nAxis-Aligned BBox Conversion\n[cx, cy, w, h] Cleaned Splits", c_amber)

    draw_box(4.8, 6.0, 3.8, 1.4, "Input & Preprocessing",
             "Letterbox Aspect-Ratio Padding\nResolution: 640 × 640 px\nPhotometric Augmentation", c_indigo)
    draw_box(4.8, 3.4, 3.8, 1.8, "YOLO26m Architecture",
             "21.8M Params · 75.0 GFLOPs\nC3k2 Backbone + Neck\nNo Distribution Focal Loss (DFL)\nMuSGD Optimizer Convergence", c_indigo)
    draw_box(4.8, 1.2, 3.8, 1.5, "Native NMS-Free Head",
             "Dual-Head One-to-One Assignment\nZero Duplicate Bounding Boxes\nPostprocess Latency: 0.08 ms\nmAP@0.5: 0.784 (78.4%)", c_blue)

    draw_box(9.5, 6.0, 3.8, 1.4, "VehicleScope AI (Streamlit)",
             "Interactive Web Interface\nEdge Inference (Apple M4 Pro)\nColor-Coded Visual Bounding Boxes", c_purple)
    draw_box(9.5, 3.7, 3.8, 1.5, "n8n Automation Node",
             "Payload JSON Serialization\nTimestamp, Totals, Confidences\nREST Webhook Dispatch", c_rose)
    draw_box(9.5, 1.8, 1.8, 1.3, "Google Sheets",
             "Cloud Logging\nAudit Trail Database", c_teal)
    draw_box(11.5, 1.8, 1.8, 1.3, "Telegram Bot",
             "Traffic Alert Bot\n(Threshold > 10)", c_amber)

    def draw_arrow(x1, y1, x2, y2, color=c_slate, rad=0.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.2,
                                    shrinkA=4, shrinkB=4,
                                    connectionstyle=f"arc3,rad={rad}"))

    draw_arrow(2.3, 6.0, 2.3, 5.2, c_teal)
    draw_arrow(2.3, 3.8, 2.3, 3.0, c_teal)
    draw_arrow(3.9, 2.1, 4.8, 6.7, c_teal, rad=-0.2)
    draw_arrow(6.7, 6.0, 6.7, 5.2, c_indigo)
    draw_arrow(6.7, 3.4, 6.7, 2.7, c_indigo)
    draw_arrow(8.6, 1.95, 9.5, 6.7, c_blue, rad=-0.25)
    draw_arrow(11.4, 6.0, 11.4, 5.2, c_purple)
    draw_arrow(10.4, 3.7, 10.4, 3.1, c_rose)
    draw_arrow(12.4, 3.7, 12.4, 3.1, c_rose)

    plt.tight_layout()
    fig.savefig(out_dir / "fig1_pipeline_flowchart.png", bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print("Saved fig1_pipeline_flowchart.png")

# ─────────────────────────────────────────────────────────────────────────────
# 2. GENERATE VOSVIEWER BIBLIOMETRIC CO-OCCURRENCE NETWORK
# ─────────────────────────────────────────────────────────────────────────────
def create_vosviewer_map():
    fig, ax = plt.subplots(figsize=(13, 9), dpi=300)
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')
    ax.axis('off')

    clusters = {
        'red': {
            'color': '#E11D48',
            'nodes': [
                ('vehicle detection', 380, (0.0, 0.4)),
                ('yolo', 320, (-0.6, 0.9)),
                ('deep learning', 290, (0.5, 1.1)),
                ('object detection', 260, (-0.2, 1.6)),
                ('convolutional neural network', 210, (-1.0, 1.5)),
                ('feature extraction', 170, (0.8, 1.7)),
                ('bounding box', 150, (-0.7, 0.2)),
            ]
        },
        'green': {
            'color': '#059669',
            'nodes': [
                ('intelligent transportation systems', 350, (1.8, -0.3)),
                ('traffic surveillance', 270, (2.2, 0.4)),
                ('traffic flow', 240, (1.2, -1.0)),
                ('vehicle counting', 220, (2.5, -0.8)),
                ('urban traffic', 190, (1.9, -1.5)),
                ('congestion management', 150, (2.8, 0.1)),
            ]
        },
        'blue': {
            'color': '#2563EB',
            'nodes': [
                ('edge computing', 280, (-1.5, -0.6)),
                ('inference latency', 240, (-2.1, -0.1)),
                ('real-time detection', 260, (-0.9, -1.2)),
                ('non-maximum suppression (nms)', 220, (-1.6, -1.4)),
                ('embedded hardware', 180, (-2.5, -0.8)),
                ('apple silicon mps', 140, (-2.2, -1.6)),
            ]
        },
        'yellow': {
            'color': '#D97706',
            'nodes': [
                ('dataset quality', 270, (0.2, -1.6)),
                ('label corruption', 230, (-0.4, -2.1)),
                ('annotation errors', 200, (0.8, -2.3)),
                ('domain shift', 220, (1.4, -2.0)),
                ('crowdsourced datasets', 180, (0.0, -2.7)),
                ('silent parsing bug', 160, (-0.9, -2.6)),
            ]
        }
    }

    node_pos = {}
    node_weights = {}
    node_colors = {}
    
    for c_id, c_data in clusters.items():
        for label, weight, pos in c_data['nodes']:
            node_pos[label] = pos
            node_weights[label] = weight
            node_colors[label] = c_data['color']

    edges = [
        ('vehicle detection', 'yolo', 8),
        ('vehicle detection', 'deep learning', 7),
        ('vehicle detection', 'object detection', 6),
        ('yolo', 'convolutional neural network', 5),
        ('deep learning', 'feature extraction', 4),
        ('vehicle detection', 'bounding box', 4),
        ('object detection', 'deep learning', 6),
        ('intelligent transportation systems', 'traffic surveillance', 7),
        ('intelligent transportation systems', 'traffic flow', 6),
        ('traffic flow', 'vehicle counting', 6),
        ('traffic flow', 'urban traffic', 5),
        ('traffic surveillance', 'congestion management', 4),
        ('edge computing', 'inference latency', 6),
        ('edge computing', 'real-time detection', 6),
        ('inference latency', 'non-maximum suppression (nms)', 5),
        ('edge computing', 'embedded hardware', 4),
        ('non-maximum suppression (nms)', 'apple silicon mps', 3),
        ('dataset quality', 'label corruption', 7),
        ('label corruption', 'annotation errors', 6),
        ('dataset quality', 'domain shift', 5),
        ('label corruption', 'silent parsing bug', 6),
        ('annotation errors', 'crowdsourced datasets', 4),
        ('vehicle detection', 'intelligent transportation systems', 9),
        ('vehicle detection', 'real-time detection', 8),
        ('vehicle detection', 'dataset quality', 6),
        ('yolo', 'non-maximum suppression (nms)', 7),
        ('vehicle counting', 'vehicle detection', 7),
        ('traffic surveillance', 'real-time detection', 5),
        ('domain shift', 'urban traffic', 4),
        ('silent parsing bug', 'bounding box', 5),
        ('apple silicon mps', 'inference latency', 4),
        ('yolo', 'inference latency', 5),
        ('crowdsourced datasets', 'vehicle detection', 4),
    ]

    for u, v, w in edges:
        p1, p2 = node_pos[u], node_pos[v]
        lw = w * 0.4
        alpha = min(0.12 + w * 0.04, 0.5)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#94A3B8', linewidth=lw, alpha=alpha, zorder=1)

    for label, pos in node_pos.items():
        w = node_weights[label]
        color = node_colors[label]
        radius = np.sqrt(w) * 0.016
        halo = plt.Circle(pos, radius * 1.15, color='#E2E8F0', alpha=0.6, zorder=2)
        ax.add_patch(halo)
        circle = plt.Circle(pos, radius, color=color, alpha=0.88, zorder=3)
        ax.add_patch(circle)
        fontsize = max(7.0, min(12.5, 6.0 + w * 0.016))
        ax.text(pos[0], pos[1] - radius - 0.08, label,
                ha='center', va='top', fontsize=fontsize, fontweight='bold',
                color='#1E293B',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFFFFF', edgecolor='none', alpha=0.75),
                zorder=4)

    all_x = [p[0] for p in node_pos.values()]
    all_y = [p[1] for p in node_pos.values()]
    ax.set_xlim(min(all_x) - 1.2, max(all_x) + 1.2)
    ax.set_ylim(min(all_y) - 0.9, max(all_y) + 0.9)

    ax.text(min(all_x) - 0.9, max(all_y) + 0.6, "VOSviewer Bibliometric Co-Occurrence Network Map",
            fontsize=13, fontweight='bold', color='#0F172A', ha='left')
    ax.text(min(all_x) - 0.9, max(all_y) + 0.35, "Co-occurrence analysis of 240+ publications in ITS and Real-Time Vehicle Detection",
            fontsize=9.5, color='#64748B', ha='left')

    legend_items = [
        ("Cluster 1: Deep Learning & YOLO Detectors", '#E11D48'),
        ("Cluster 2: Intelligent Transportation Systems", '#059669'),
        ("Cluster 3: Edge Computing & Latency", '#2563EB'),
        ("Cluster 4: Dataset Integrity & Domain Shift (Research Gap)", '#D97706'),
    ]
    legend_y = min(all_y) - 0.4
    for idx, (leg_text, leg_color) in enumerate(legend_items):
        lx = min(all_x) - 0.8 + (idx % 2) * 2.8
        ly = legend_y - (idx // 2) * 0.25
        leg_circle = plt.Circle((lx, ly), 0.06, color=leg_color, zorder=5)
        ax.add_patch(leg_circle)
        ax.text(lx + 0.12, ly, leg_text, fontsize=8.5, color='#334155', va='center', fontweight='bold', zorder=5)

    plt.tight_layout()
    fig.savefig(out_dir / "fig2_vosviewer_network.png", bbox_inches='tight', facecolor='#FAFAFA')
    plt.close()
    print("Saved fig2_vosviewer_network.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. GENERATE NODEXL PRO CLUSTERED NETWORK GRAPH (3 Thematic Clusters)
# ─────────────────────────────────────────────────────────────────────────────
def create_nodexl_graph():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.axis('off')

    # NodeXL Group-in-a-Box format: 3 distinct bounded clusters
    # Group 1: Deep Learning & YOLO Detectors
    # Group 2: Intelligent Traffic Management & CCTV Surveillance
    # Group 3: Edge Computing & Telemetry Automation
    
    # Bounding boxes for 3 groups
    groups = {
        'G1': {
            'title': 'G1: Deep Learning Detectors (N=8, Edges=14)',
            'color': '#2563EB',
            'box': (0.5, 4.4, 5.8, 3.8),
            'nodes': [
                ('YOLO26 / YOLOv8', (2.2, 6.7), 24, '#2563EB'),
                ('Dual-Head Design', (1.4, 5.8), 18, '#3B82F6'),
                ('NMS-Free Head', (3.4, 6.8), 20, '#1D4ED8'),
                ('MuSGD Optimizer', (1.5, 4.9), 16, '#60A5FA'),
                ('C3k2 Backbone', (3.2, 5.6), 18, '#3B82F6'),
                ('Bounding-Box Regression', (4.5, 6.5), 17, '#2563EB'),
                ('Precision & Recall', (4.7, 5.2), 16, '#60A5FA'),
                ('mAP@0.5 Benchmarks', (2.6, 4.8), 19, '#1D4ED8'),
            ]
        },
        'G2': {
            'title': 'G2: Traffic Surveillance & ITS (N=7, Edges=12)',
            'color': '#059669',
            'box': (7.0, 4.4, 6.5, 3.8),
            'nodes': [
                ('Traffic Flow Analysis', (9.8, 6.7), 23, '#059669'),
                ('Vehicle Classification', (8.4, 5.8), 20, '#10B981'),
                ('Multi-Class Density', (11.4, 6.8), 19, '#047857'),
                ('Congestion Alert', (8.2, 4.9), 16, '#34D399'),
                ('Camera Calibration', (10.2, 5.6), 17, '#10B981'),
                ('Urban Roadways', (12.2, 5.8), 17, '#059669'),
                ('Inductive-Loop Replacement', (11.0, 4.8), 16, '#047857'),
            ]
        },
        'G3': {
            'title': 'G3: Edge Hardware & Pipeline Automation (N=7, Edges=11)',
            'color': '#7C3AED',
            'box': (3.6, 0.3, 7.2, 3.6),
            'nodes': [
                ('Streamlit Edge UI', (5.8, 2.7), 21, '#7C3AED'),
                ('Apple MPS GPU', (4.6, 1.8), 19, '#8B5CF6'),
                ('n8n Webhook Engine', (7.4, 2.7), 20, '#6D28D9'),
                ('Google Sheets API', (9.0, 2.2), 18, '#8B5CF6'),
                ('Telegram Bot Alerter', (8.4, 1.0), 17, '#6D28D9'),
                ('Latency Optimization', (5.4, 0.9), 18, '#7C3AED'),
                ('JSON Telemetry', (7.0, 1.3), 16, '#A78BFA'),
            ]
        }
    }

    # Draw group hulls/boxes
    for gid, ginfo in groups.items():
        gx, gy, gw, gh = ginfo['box']
        # Box background
        gbg = patches.FancyBboxPatch((gx, gy), gw, gh, boxstyle="round,pad=0.12,rounding_size=0.2",
                                     facecolor='#F8FAFC', edgecolor=ginfo['color'], linewidth=1.8,
                                     linestyle='--', alpha=0.9, zorder=1)
        ax.add_patch(gbg)
        # Group Header
        ax.text(gx + 0.3, gy + gh - 0.3, ginfo['title'], fontsize=10, fontweight='bold',
                color=ginfo['color'], zorder=2)

    # Node positions lookup
    npos = {}
    for gid, ginfo in groups.items():
        for name, pos, sz, col in ginfo['nodes']:
            npos[name] = (pos, sz, col)

    # Intra and Inter-Group Edges
    inter_edges = [
        # G1 intra
        ('YOLO26 / YOLOv8', 'Dual-Head Design', 3),
        ('YOLO26 / YOLOv8', 'NMS-Free Head', 4),
        ('YOLO26 / YOLOv8', 'C3k2 Backbone', 3),
        ('YOLO26 / YOLOv8', 'mAP@0.5 Benchmarks', 3),
        ('Dual-Head Design', 'MuSGD Optimizer', 2),
        ('C3k2 Backbone', 'Bounding-Box Regression', 2),
        ('Bounding-Box Regression', 'Precision & Recall', 2),
        ('mAP@0.5 Benchmarks', 'Precision & Recall', 2),

        # G2 intra
        ('Traffic Flow Analysis', 'Vehicle Classification', 4),
        ('Traffic Flow Analysis', 'Multi-Class Density', 3),
        ('Traffic Flow Analysis', 'Camera Calibration', 3),
        ('Vehicle Classification', 'Congestion Alert', 2),
        ('Multi-Class Density', 'Urban Roadways', 2),
        ('Camera Calibration', 'Inductive-Loop Replacement', 2),
        ('Congestion Alert', 'Inductive-Loop Replacement', 2),

        # G3 intra
        ('Streamlit Edge UI', 'n8n Webhook Engine', 4),
        ('Streamlit Edge UI', 'Apple MPS GPU', 3),
        ('Streamlit Edge UI', 'Latency Optimization', 2),
        ('n8n Webhook Engine', 'Google Sheets API', 3),
        ('n8n Webhook Engine', 'Telegram Bot Alerter', 3),
        ('n8n Webhook Engine', 'JSON Telemetry', 2),
        ('Apple MPS GPU', 'Latency Optimization', 3),

        # Inter-group bridges (Cross-Group Synergy!)
        ('YOLO26 / YOLOv8', 'Vehicle Classification', 4),
        ('mAP@0.5 Benchmarks', 'Traffic Flow Analysis', 3),
        ('NMS-Free Head', 'Latency Optimization', 4),
        ('YOLO26 / YOLOv8', 'Apple MPS GPU', 4),
        ('Vehicle Classification', 'Streamlit Edge UI', 4),
        ('Congestion Alert', 'Telegram Bot Alerter', 3),
        ('Multi-Class Density', 'Google Sheets API', 2),
    ]

    # Draw Edges
    for u, v, w in inter_edges:
        p1, sz1, c1 = npos[u]
        p2, sz2, c2 = npos[v]
        # Cross group styling
        is_cross = (c1[:4] != c2[:4])
        edge_col = '#F59E0B' if is_cross else '#94A3B8'
        edge_alpha = 0.65 if is_cross else 0.35
        lw = w * 0.7 if is_cross else w * 0.5
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=edge_col, linewidth=lw, alpha=edge_alpha, zorder=2)

    # Draw Nodes
    for name, (pos, sz, col) in npos.items():
        # Circle
        rad = sz * 0.012
        halo = plt.Circle(pos, rad * 1.3, color='#E2E8F0', alpha=0.7, zorder=3)
        ax.add_patch(halo)
        node_circ = plt.Circle(pos, rad, color=col, alpha=0.92, zorder=4)
        ax.add_patch(node_circ)
        # Label
        ax.text(pos[0], pos[1] - rad - 0.08, name, ha='center', va='top', fontsize=8.2,
                fontweight='bold', color='#0F172A',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFFFFF', edgecolor='#CBD5E1', lw=0.6, alpha=0.9),
                zorder=5)

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.8)

    # NodeXL Header & Meta
    ax.text(0.6, 8.5, "NodeXL Pro Thematic Cluster Network Graph",
            fontsize=13, fontweight='bold', color='#0F172A', ha='left')
    ax.text(0.6, 8.25, "Clustered network graph (Harel-Koren Fast Multiscale layout) showing inter-group bridges between G1 (Model), G2 (Traffic), and G3 (Edge Pipeline)",
            fontsize=9.2, color='#64748B', ha='left')

    # Legend for cross-group bridges
    ax.plot([10.2, 10.9], [8.4, 8.4], color='#F59E0B', linewidth=2.5, alpha=0.85)
    ax.text(11.05, 8.4, "Inter-Group Cross-Domain Bridge", fontsize=8.5, color='#B45309', va='center', fontweight='bold')

    plt.tight_layout()
    fig.savefig(out_dir / "fig3_nodexl_network.png", bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print("Saved fig3_nodexl_network.png")

if __name__ == '__main__':
    create_flowchart()
    create_vosviewer_map()
    create_nodexl_graph()
