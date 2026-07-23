"""Download NSL-KDD and UNSW-NB15 datasets from GitHub."""
import urllib.request
import os

os.makedirs("data/raw", exist_ok=True)

# NSL-KDD
print("Downloading NSL-KDD dataset...")
nsl_kdd_base = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/"
nsl_kdd_files = {
    "KDDTrain+.txt": nsl_kdd_base + "KDDTrain%2B.txt",
    "KDDTest+.txt": nsl_kdd_base + "KDDTest%2B.txt",
}

for fname, url in nsl_kdd_files.items():
    dest = os.path.join("data", "raw", fname)
    if os.path.exists(dest):
        print(f"  Already exists: {fname} ({os.path.getsize(dest)} bytes)")
    else:
        print(f"  Downloading {fname}...")
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved: {fname} ({os.path.getsize(dest)} bytes)")

# UNSW-NB15
print("\nDownloading UNSW-NB15 dataset...")
unsw_nb15_base = "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/"
unsw_nb15_files = {
    "UNSW_NB15_training-set.csv": unsw_nb15_base + "UNSW_NB15_training-set.csv",
    "UNSW_NB15_testing-set.csv": unsw_nb15_base + "UNSW_NB15_testing-set.csv",
}

for fname, url in unsw_nb15_files.items():
    dest = os.path.join("data", "raw", fname)
    if os.path.exists(dest):
        print(f"  Already exists: {fname} ({os.path.getsize(dest)} bytes)")
    else:
        print(f"  Downloading {fname}...")
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved: {fname} ({os.path.getsize(dest)} bytes)")

print("\nDatasets ready!")
