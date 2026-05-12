### written by Claude


import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def extract_validation_ppl_from_log(log_file):
    """
    Extract validation perplexities and their step numbers from a JoeyNMT log file.
    
    Args:
        log_file: path to log file
    
    Returns:
        dict with step -> ppl mapping
    """
    results = {}
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Look for "Evaluation result" lines which contain ppl values
    for i, line in enumerate(lines):
        if "Evaluation result (greedy)" in line:
            # Extract PPL value from lines like:
            # "Evaluation result (greedy) loss:   2.10, ppl:   8.01, acc:   0.46, ..."
            match = re.search(r'ppl:\s+([\d.]+)', line)
            if match:
                ppl = float(match.group(1))
                
                # Look backwards to find the step number
                # It should be in a line like "Epoch   X, Step:  XXXX, ..."
                step = None
                for j in range(i, max(0, i-50), -1):
                    step_match = re.search(r'Step:\s+(\d+)', lines[j])
                    if step_match:
                        step = int(step_match.group(1))
                        break
                
                if step is not None:
                    results[step] = ppl
    
    # Sort by step number
    return dict(sorted(results.items()))


def create_comparison_table(baseline_ppl, prenorm_ppl, postnorm_ppl, output_file="results_table.txt"):
    """
    Create a table comparing validation perplexities across models.
    
    Args:
        baseline_ppl: dict of step -> ppl for baseline
        prenorm_ppl: dict of step -> ppl for prenorm
        postnorm_ppl: dict of step -> ppl for postnorm
        output_file: where to save the table
    """
    
    # Get all unique steps (at multiples of 500)
    all_steps = set()
    all_steps.update(baseline_ppl.keys())
    all_steps.update(prenorm_ppl.keys())
    all_steps.update(postnorm_ppl.keys())
    all_steps = sorted([s for s in all_steps if s % 500 == 0])
    
    # Create table
    rows = []
    for step in all_steps:
        baseline_val = baseline_ppl.get(step, None)
        prenorm_val = prenorm_ppl.get(step, None)
        postnorm_val = postnorm_ppl.get(step, None)
        
        rows.append({
            'Validation Step': step,
            'Baseline': baseline_val,
            'Prenorm': prenorm_val,
            'Postnorm': postnorm_val
        })
    
    df = pd.DataFrame(rows)
    
    # Print to console
    print("\n" + "="*80)
    print("VALIDATION PERPLEXITY COMPARISON TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("VALIDATION PERPLEXITY COMPARISON TABLE\n")
        f.write("="*80 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n" + "="*80 + "\n")
    
    print(f"Table saved to {output_file}")
    return df


def plot_comparison(baseline_ppl, prenorm_ppl, postnorm_ppl, output_file="ppl_comparison.png"):
    """
    Create a line chart comparing validation perplexities across models.
    
    Args:
        baseline_ppl: dict of step -> ppl for baseline
        prenorm_ppl: dict of step -> ppl for prenorm
        postnorm_ppl: dict of step -> ppl for postnorm
        output_file: where to save the figure
    """
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Extract steps and PPL values
    baseline_steps = sorted(baseline_ppl.keys())
    baseline_values = [baseline_ppl[s] for s in baseline_steps]
    
    prenorm_steps = sorted(prenorm_ppl.keys())
    prenorm_values = [prenorm_ppl[s] for s in prenorm_steps]
    
    postnorm_steps = sorted(postnorm_ppl.keys())
    postnorm_values = [postnorm_ppl[s] for s in postnorm_steps]
    
    # Plot lines
    ax.plot(baseline_steps, baseline_values, 'o-', linewidth=2.5, markersize=6, 
            label='Baseline', color='#1f77b4', alpha=0.8)
    ax.plot(prenorm_steps, prenorm_values, 's-', linewidth=2.5, markersize=6,
            label='Pre-norm', color='#ff7f0e', alpha=0.8)
    ax.plot(postnorm_steps, postnorm_values, '^-', linewidth=2.5, markersize=6,
            label='Post-norm', color='#2ca02c', alpha=0.8)
    
    # Labels and formatting
    ax.set_xlabel('Training Step', fontsize=13, fontweight='bold')
    ax.set_ylabel('Validation Perplexity', fontsize=13, fontweight='bold')
    ax.set_title('Validation Perplexity Comparison: Baseline vs Pre-norm vs Post-norm', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format axes
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.show()


def main():
    """Main function."""
    
    # Log file paths
    baseline_log = "./logs/comparison/baseline.log"
    prenorm_log = "./logs/comparison/prenorm.log"
    postnorm_log = "./logs/comparison/postnorm.log"
    
    print("="*80)
    print("EXTRACTING VALIDATION PERPLEXITIES FROM LOG FILES")
    print("="*80 + "\n")
    
    # Extract perplexities
    print(f"Reading baseline log: {baseline_log}")
    baseline_ppl = extract_validation_ppl_from_log(baseline_log)
    print(f"  Found {len(baseline_ppl)} validation results")
    
    print(f"Reading pre-norm log: {prenorm_log}")
    prenorm_ppl = extract_validation_ppl_from_log(prenorm_log)
    print(f"  Found {len(prenorm_ppl)} validation results")
    
    if postnorm_log and Path(postnorm_log).exists():
        print(f"Reading post-norm log: {postnorm_log}")
        postnorm_ppl = extract_validation_ppl_from_log(postnorm_log)
        print(f"  Found {len(postnorm_ppl)} validation results")
    else:
        print("Post-norm log not provided or not found. Using empty dict.")
        postnorm_ppl = {}
    
    # Create table
    print("\nCreating comparison table...")
    df = create_comparison_table(baseline_ppl, prenorm_ppl, postnorm_ppl,
                                 output_file="./results/results_table.txt")
    
    # Create plot (only if we have data for at least 2 models)
    if len(baseline_ppl) > 0 and len(prenorm_ppl) > 0:
        print("\nCreating comparison plot...")
        plot_comparison(baseline_ppl, prenorm_ppl, postnorm_ppl,
                       output_file="./results/ppl_comparison.png")
    else:
        print("Not enough data to create comparison plot.")
    
    # Print some statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    if baseline_ppl:
        baseline_min = min(baseline_ppl.values())
        baseline_min_step = [k for k, v in baseline_ppl.items() if v == baseline_min][0]
        print(f"Baseline:  Best PPL = {baseline_min:.2f} at step {baseline_min_step}")
    
    if prenorm_ppl:
        prenorm_min = min(prenorm_ppl.values())
        prenorm_min_step = [k for k, v in prenorm_ppl.items() if v == prenorm_min][0]
        print(f"Pre-norm:  Best PPL = {prenorm_min:.2f} at step {prenorm_min_step}")
    
    if postnorm_ppl:
        postnorm_min = min(postnorm_ppl.values())
        postnorm_min_step = [k for k, v in postnorm_ppl.items() if v == postnorm_min][0]
        print(f"Post-norm: Best PPL = {postnorm_min:.2f} at step {postnorm_min_step}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
