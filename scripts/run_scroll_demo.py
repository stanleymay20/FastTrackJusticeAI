#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run the complete scroll judgment demonstration suite.
This script generates example judgments and then analyzes them to demonstrate
the scroll-aware judgment generation system.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_command(command, description):
    """Run a command and print its output."""
    print(f"\n{'=' * 80}")
    print(f" {description} ".center(80, '='))
    print(f"{'=' * 80}\n")
    
    start_time = time.time()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        shell=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    elapsed_time = time.time() - start_time
    
    if process.returncode == 0:
        print(f"\n✅ {description} completed successfully in {elapsed_time:.2f} seconds.")
    else:
        print(f"\n❌ {description} failed with exit code {process.returncode}.")
    
    return process.returncode

def main():
    """Run the complete scroll judgment demonstration suite."""
    # Get the scripts directory
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create a timestamp for the run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a log file
    log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            f"logs/scroll_demo_{timestamp}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Redirect stdout and stderr to the log file
    sys.stdout = open(log_file, 'w')
    sys.stderr = sys.stdout
    
    print(f"Scroll Judgment Demonstration Suite")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file}")
    
    # Step 1: Generate example judgments
    generate_script = os.path.join(scripts_dir, "generate_example_judgments.py")
    generate_cmd = f"python {generate_script}"
    if run_command(generate_cmd, "Generating example judgments") != 0:
        print("Failed to generate example judgments. Aborting.")
        return 1
    
    # Step 2: Analyze the judgments
    analyze_script = os.path.join(scripts_dir, "scroll_judgment_comparator.py")
    analyze_cmd = f"python {analyze_script}"
    if run_command(analyze_cmd, "Analyzing judgments") != 0:
        print("Failed to analyze judgments. Aborting.")
        return 1
    
    # Step 3: Generate a summary report
    print("\n" + "=" * 80)
    print(" Scroll Judgment Demonstration Summary ".center(80, '='))
    print("=" * 80 + "\n")
    
    # Get the reports directory
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
    
    # Check if the markdown report exists
    markdown_report = os.path.join(reports_dir, "scroll_judgment_analysis.md")
    if os.path.exists(markdown_report):
        print(f"Analysis report generated: {markdown_report}")
        print("You can view this report to see the detailed analysis of the judgments.")
    
    # Check if the visualizations directory exists
    visualizations_dir = os.path.join(reports_dir, "visualizations")
    if os.path.exists(visualizations_dir):
        print(f"Visualizations generated in: {visualizations_dir}")
        print("These visualizations show the differences in judgments across scroll phases.")
    
    # Get the examples directory
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
    
    # Count the number of generated judgments
    judgment_count = len([f for f in os.listdir(examples_dir) if f.endswith(".txt")])
    print(f"Generated {judgment_count} example judgments in: {examples_dir}")
    
    print("\n" + "=" * 80)
    print(" Demonstration Complete ".center(80, '='))
    print("=" * 80 + "\n")
    
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 