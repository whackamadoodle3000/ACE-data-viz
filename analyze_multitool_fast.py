#!/usr/bin/env python3
"""
Fast multi-tool accuracy analysis - optimized to avoid hanging.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def log(msg):
    """Print with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {msg}", flush=True)

log("Starting script...")
log("Importing matplotlib...")
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
log("Importing matplotlib.pyplot...")
import matplotlib.pyplot as plt
log("Importing numpy...")
import numpy as np
log("All imports complete!")

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "plots"

# Define the approaches to analyze
APPROACHES = {
    'Baseline All': BASE_DIR / 'score_baseline_all/DeepSeek-V3.2-Exp-FC/ace_metrics',
    'Playbook All': BASE_DIR / 'score_playbook_all/DeepSeek-V3.2-Exp-FC/ace_metrics',
    'ACE Claude': BASE_DIR / 'score_ace_claude_run/DeepSeek-V3.2-Exp-FC/ace_metrics',
    'SAI Dynamic': BASE_DIR / 'results_sai/score_ace_claude_run/DeepSeek-V3.2-Exp-FC/ace_metrics',
}


def load_json_file(json_path):
    """Load JSON file, handling git-lfs pointers."""
    try:
        log(f"  Opening file: {json_path.name}")
        with open(json_path, 'r') as f:
            first_line = f.readline()
            if first_line.startswith('version https://git-lfs.github.com'):
                log(f"  File is git-lfs pointer, skipping")
                return None
            f.seek(0)
            log(f"  Parsing JSON (this may take a moment for large files)...")
            data = json.load(f)
            log(f"  JSON parsed successfully")
            return data
    except Exception as e:
        log(f"  Error loading {json_path.name}: {e}")
        return None


def get_accuracy_by_tool_count(data):
    """Extract accuracy by tool count from JSON data."""
    if not data or 'records' not in data:
        log(f"  No records found in data")
        return {}
    
    log(f"  Processing {len(data['records'])} records...")
    stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for i, record in enumerate(data['records']):
        if i % 200 == 0 and i > 0:
            log(f"    Processed {i} records...")
        tc = record.get('tool_count', 0)
        if tc >= 2:  # Only multi-tool tests
            stats[tc]['total'] += 1
            if record.get('valid', False):
                stats[tc]['correct'] += 1
    
    log(f"  Calculating accuracies...")
    # Calculate accuracy
    result = {}
    for tc, s in stats.items():
        if s['total'] > 0:
            result[tc] = {
                'accuracy': s['correct'] / s['total'] * 100,
                'correct': s['correct'],
                'total': s['total']
            }
    log(f"  Found {len(result)} tool counts with data")
    return result


def main():
    log("="*70)
    log("Loading data from all approaches...")
    log("="*70)
    
    results = {}
    
    for name, metrics_dir in APPROACHES.items():
        log(f"\nProcessing: {name}")
        log(f"  Directory: {metrics_dir}")
        
        if not metrics_dir.exists():
            log(f"  SKIP - Directory not found")
            continue
        
        log(f"  Searching for JSON files...")
        json_files = list(metrics_dir.glob('*.json'))
        if not json_files:
            log(f"  SKIP - No JSON files found")
            continue
        
        log(f"  Found {len(json_files)} JSON file(s): {json_files[0].name}")
        data = load_json_file(json_files[0])
        if data:
            acc_data = get_accuracy_by_tool_count(data)
            if acc_data:
                results[name] = acc_data
                total_tests = sum(d['total'] for d in acc_data.values())
                log(f"  ✅ SUCCESS - Loaded {total_tests} multi-tool tests")
            else:
                log(f"  SKIP - No multi-tool data found")
        else:
            log(f"  SKIP - Invalid or git-lfs JSON file")
    
    if not results:
        log("\n❌ No data to plot!")
        return
    
    # Get all tool counts
    log("\n" + "="*70)
    log("Data loading complete! Preparing to generate plots...")
    log("="*70)
    all_tool_counts = sorted(set(tc for data in results.values() for tc in data.keys()))
    
    log(f"Tool counts found: {all_tool_counts}")
    log(f"Approaches loaded: {list(results.keys())}")
    
    # Prepare data for plotting
    approach_names = list(results.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    
    # === PLOT 1: Multi-bar comparison ===
    log("\nGenerating Plot 1: Multi-bar comparison by tool count...")
    log("  Creating figure...")
    fig, ax = plt.subplots(figsize=(14, 8))
    log("  Figure created")
    
    x = np.arange(len(all_tool_counts))
    width = 0.8 / len(approach_names)
    
    log("  Adding bars to plot...")
    for i, approach in enumerate(approach_names):
        log(f"    Adding bars for {approach}...")
        accuracies = []
        for tc in all_tool_counts:
            if tc in results[approach]:
                accuracies.append(results[approach][tc]['accuracy'])
            else:
                accuracies.append(0)
        
        offset = width * (i - len(approach_names)/2 + 0.5)
        bars = ax.bar(x + offset, accuracies, width, label=approach,
                     color=colors[i % len(colors)], alpha=0.85)
        
        # Add percentage labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}%', ha='center', va='bottom', fontsize=8)
    
    log("  Formatting plot...")
    ax.set_xlabel('Number of Tools Required', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Multi-Tool Function Calling Accuracy by Tool Count\n(Excluding Single-Tool Tests)',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([str(tc) for tc in all_tool_counts])
    ax.legend(loc='best', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)
    
    log("  Applying tight layout...")
    plt.tight_layout()
    output1 = OUTPUT_DIR / 'multitool_by_count.png'
    log(f"  Saving to {output1.name}...")
    plt.savefig(output1, dpi=300, bbox_inches='tight')
    log(f"  ✅ Plot 1 saved: {output1.name}")
    plt.close()
    log("  Figure closed")
    
    # === PLOT 2: Overall multi-tool accuracy ===
    log("\nGenerating Plot 2: Overall multi-tool accuracy comparison...")
    log("  Creating figure...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    log("  Calculating overall accuracies...")
    overall_acc = []
    overall_labels = []
    
    for approach in approach_names:
        total_correct = sum(results[approach][tc]['correct'] for tc in results[approach])
        total_tests = sum(results[approach][tc]['total'] for tc in results[approach])
        acc = (total_correct / total_tests * 100) if total_tests > 0 else 0
        overall_acc.append(acc)
        overall_labels.append(f"{approach}\n({total_correct}/{total_tests})")
    
    log("  Adding bars...")
    bars = ax.bar(range(len(approach_names)), overall_acc,
                 color=colors[:len(approach_names)], alpha=0.85, width=0.6)
    
    for bar, acc in zip(bars, overall_acc):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{acc:.1f}%', ha='center', va='bottom',
               fontsize=12, fontweight='bold')
    
    log("  Formatting plot...")
    ax.set_ylabel('Overall Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Overall Multi-Tool Accuracy (2+ Tools)\nAcross Different Approaches',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(range(len(approach_names)))
    ax.set_xticklabels(overall_labels, fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)
    
    log("  Applying tight layout...")
    plt.tight_layout()
    output2 = OUTPUT_DIR / 'multitool_overall.png'
    log(f"  Saving to {output2.name}...")
    plt.savefig(output2, dpi=300, bbox_inches='tight')
    log(f"  ✅ Plot 2 saved: {output2.name}")
    plt.close()
    log("  Figure closed")
    
    # === Print summary ===
    log("\n" + "="*70)
    log("Printing summary...")
    log("="*70)
    print("\n" + "="*70)
    print("SUMMARY: Multi-Tool Accuracy (2+ Tools)")
    print("="*70)
    for approach in approach_names:
        total_correct = sum(results[approach][tc]['correct'] for tc in results[approach])
        total_tests = sum(results[approach][tc]['total'] for tc in results[approach])
        acc = (total_correct / total_tests * 100) if total_tests > 0 else 0
        print(f"{approach:20s}: {acc:5.1f}% ({total_correct}/{total_tests})")
    
    print("\n" + "="*70)
    print("By Tool Count:")
    print("="*70)
    header = f"{'Tool Count':<12}"
    for approach in approach_names:
        header += f"{approach:<18}"
    print(header)
    print("-" * 70)
    
    for tc in all_tool_counts:
        row = f"{tc:<12}"
        for approach in approach_names:
            if tc in results[approach]:
                acc = results[approach][tc]['accuracy']
                cnt = f"({results[approach][tc]['correct']}/{results[approach][tc]['total']})"
                row += f"{acc:5.1f}% {cnt:<10}"
            else:
                row += f"{'N/A':<18}"
        print(row)
    
    log("\n" + "="*70)
    log("✅ Analysis complete!")
    log("="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

