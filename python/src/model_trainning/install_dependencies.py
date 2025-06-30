import subprocess
import sys

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")

def main():
    print("🚀 Installing dependencies for Student Performance Prediction Model...")
    
    packages = [
        "torch",
        "pandas", 
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn"
    ]
    
    for package in packages:
        print(f"\n📦 Installing {package}...")
        install_package(package)
    
    print("\n✨ Installation completed!")
    print("\nYou can now run:")
    print("  python student_performance_prediction.py  # To train the model")
    print("  python demo_prediction.py                # To run demo predictions")

if __name__ == "__main__":
    main() 