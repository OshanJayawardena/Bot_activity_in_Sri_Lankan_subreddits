import subprocess
import sys

def pipeline_commands(executable=None, config="config.yaml"):
    executable = executable or sys.executable
    return [
        [executable, "src/collect_arctic.py", "--config", config],
        [executable, "src/features.py", "--config", config],
        [executable, "src/text_analysis.py", "--config", config],
        [executable, "src/report.py", "--config", config],
    ]

def run_pipeline(commands=None, runner=subprocess.run):
    if commands is None:
        commands = pipeline_commands()
    for cmd in commands:
        print("\n>>>", " ".join(cmd))
        runner(cmd, check=True)
    print("\nDone. Open reports/report.html")

if __name__ == "__main__":
    run_pipeline()
