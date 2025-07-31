import subprocess
import sys
import time
import webbrowser
from threading import Timer

def open_browser():
    """Open browser after a short delay"""
    time.sleep(3)
    webbrowser.open('http://localhost:8501')

def run_dashboard():
    """Run the Streamlit dashboard"""
    print(" Starting Prostate Cancer Analytics Dashboard...")
    print(" Loading interactive visualizations...")
    print(" Dashboard will open in your browser at: http://localhost:8501")
    print(" Press Ctrl+C to stop the dashboard")
    print("-" * 60)
    
    # Open browser after 3 seconds
    Timer(3.0, open_browser).start()
    
    try:
        # Run streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"], 
                      cwd=".", check=True)
    except KeyboardInterrupt:
        print("\n Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running dashboard: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    run_dashboard()