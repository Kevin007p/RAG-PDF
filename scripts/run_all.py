import subprocess
import sys

# Paths to your two scripts
path = "scripts/"
script1 = "eval_retriever.py"  # Replace with the filename of the first script
script2 = "eval_generation.py"  # Replace with the filename of the second script

def run_script(script_path):
    try:
        print(f"Running script: {script_path}")
        subprocess.run([sys.executable, path + script_path], check=True)
        print(f"Finished running: {script_path}\n")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Run the first script
    run_script(script1)
    
    # Run the second script
    run_script(script2)
