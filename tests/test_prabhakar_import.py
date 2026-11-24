import sys
import os

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from inference.github_importer import parse_github_url, inspect_repository, register_multiple_github_models

def main():
    url = "https://github.com/Prabhakar-Yadav/Feature-Extraction-from-Satellite-Imagery-BTP-II"
    parsed = parse_github_url(url)
    print("Parsed GitHub URL successfully:", parsed["web_url"])

    scratch_dir = r"C:\BTP II\scratch_prabhakar"
    info = inspect_repository(scratch_dir, parsed)
    print(f"Discovered {len(info['weight_files'])} weight files/assets in repository and releases.")
    print(f"Discovered {len(info['discovered_models'])} distinct model profiles:")

    assert len(info["discovered_models"]) == 2, f"Expected 2 production models, got {len(info['discovered_models'])}"
    print(f"Verified exactly {len(info['discovered_models'])} current semester production models discovered.")

    for idx, m in enumerate(info["discovered_models"]):
        print(f"  [{idx + 1}] {m['display_name']}")
        print(f"       Architecture : {m['architecture']}")
        print(f"       Target Sensor: {m['sensor']} ({m['native_gsd_m']}m GSD)")
        print(f"       Reported OA  : {m['overall_accuracy']}%")
        print(f"       Checkpoint   : {os.path.basename(m['checkpoint_path'])}")

    print("\nSimulating batch registration of all models...")
    regs = register_multiple_github_models(info)
    print(f"Successfully registered {len(regs)} models into platform registry!")
    for r in regs:
        print(f"  -> Model ID: {r['id']} | Display: {r['display_name']}")

if __name__ == "__main__":
    main()
