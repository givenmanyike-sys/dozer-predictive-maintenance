"""Generate clean, high-resolution diagnostic figures for executive reporting."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from src.data.load_data import load_workbook
from src.features.cleaning import prepare_telemetry

def main():
    out_dir = Path("outputs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    wb = load_workbook("data/raw/dozer_predictive_maintenance.xlsx")
    telemetry = prepare_telemetry(wb["telemetry"])
    events = wb["events"]

    # 1. Fleet Timeline Plot
    plt.figure(figsize=(11, 4.5), dpi=300)
    machines = list(telemetry["Machine ID"].unique())
    for idx, (machine, group) in enumerate(telemetry.groupby("Machine ID")):
        plt.plot(group["Timestamp"], [idx] * len(group), "|", label=f"{machine} Telemetry", markersize=6, alpha=0.6)
    
    colors = {"Unplanned Failure": "red", "False Alarm": "orange", "Scheduled Maintenance": "green"}
    for _, event in events.iterrows():
        m_idx = machines.index(event["Machine ID"])
        cat = event["Category"]
        plt.scatter(
            event["Event Timestamp"],
            m_idx,
            color=colors.get(cat, "blue"),
            s=140,
            zorder=5,
            edgecolor="black",
            label=f"{event['Machine ID']}: {cat}",
        )
    
    plt.yticks(range(len(machines)), machines, fontsize=11, fontweight="bold")
    plt.title("Fleet Operating Telemetry & Ground-Truth Event Timeline (6 Months)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Timestamp", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "01_fleet_telemetry_timeline.png")
    plt.close()

    # 2. DZ-127 Hydraulic Failure Degradation
    dz127 = telemetry[telemetry["Machine ID"] == "DZ-127"].sort_values("Timestamp")
    plt.figure(figsize=(11, 5), dpi=300)
    plt.plot(dz127["Timestamp"], dz127["Hyd Oil Temp Max"], color="crimson", label="Hydraulic Oil Temp Max (°C)", linewidth=1.5)
    plt.axvline(pd.Timestamp("2025-04-18 17:00:00"), color="black", linestyle=":", linewidth=2.2, label="Unplanned Failure (April 18)")
    plt.axhline(85, color="orange", linestyle="--", label="OEM Warning Threshold (85°C)", linewidth=1.2)
    plt.axhline(95, color="red", linestyle="--", label="OEM Critical Threshold (95°C)", linewidth=1.2)
    plt.title("DZ-127 Hydraulic System Thermal Runaway Leading to Failure", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Hydraulic Oil Temp Max (°C)", fontsize=11)
    plt.legend(loc="upper left", framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "02_dz127_hydraulic_failure_telemetry.png")
    plt.close()

    # 3. DZ-118 Engine Failure Degradation
    dz118 = telemetry[telemetry["Machine ID"] == "DZ-118"].sort_values("Timestamp")
    fig, ax1 = plt.subplots(figsize=(11, 5), dpi=300)
    ax2 = ax1.twinx()
    l1 = ax1.plot(dz118["Timestamp"], dz118["Oil Temp Max"], color="tab:red", label="Engine Oil Temp Max (°C)", linewidth=1.5)
    l2 = ax2.plot(dz118["Timestamp"], dz118["Blow-by Press Max"], color="tab:blue", linestyle="--", label="Blow-by Pressure Max (kPa)", linewidth=1.5)
    l3 = ax1.axvline(pd.Timestamp("2025-05-23 23:00:00"), color="black", linestyle=":", linewidth=2.2, label="Unplanned Failure (May 23)")
    ax1.set_title("DZ-118 Engine Thermal & Blow-by Degradation Pre-Failure", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Date", fontsize=11)
    ax1.set_ylabel("Engine Oil Temp (°C)", color="tab:red", fontsize=11)
    ax2.set_ylabel("Blow-by Pressure (kPa)", color="tab:blue", fontsize=11)
    lines = l1 + l2 + [l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "03_dz118_engine_failure_telemetry.png")
    plt.close()

    # 4. False Alarm vs True Failure Comparison
    dz142 = telemetry[telemetry["Machine ID"] == "DZ-142"].sort_values("Timestamp")
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)
    
    ax_a.plot(dz142["Timestamp"], dz142["Coolant Temp Max"], color="darkorange", linewidth=1.3)
    ax_a.axvline(pd.Timestamp("2025-03-22 05:00:00"), color="orange", linestyle=":", linewidth=2, label="False Alarm (1-hr Spike)")
    ax_a.set_title("DZ-142: False Alarm (Isolated Spike)", fontsize=12, fontweight="bold")
    ax_a.set_ylabel("Coolant Temp Max (°C)", fontsize=10)
    ax_a.set_xlabel("Date", fontsize=10)
    ax_a.legend(loc="upper left")
    ax_a.grid(True, alpha=0.3)

    ax_b.plot(dz118["Timestamp"], dz118["Oil Temp Max"], color="crimson", linewidth=1.3)
    ax_b.axvline(pd.Timestamp("2025-05-23 23:00:00"), color="red", linestyle=":", linewidth=2, label="True Failure (Sustained Rise)")
    ax_b.set_title("DZ-118: True Breakdown (Multi-Day Trend)", fontsize=12, fontweight="bold")
    ax_b.set_ylabel("Engine Oil Temp Max (°C)", fontsize=10)
    ax_b.set_xlabel("Date", fontsize=10)
    ax_b.legend(loc="upper left")
    ax_b.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "04_dz142_false_alarm_vs_true_failure.png")
    plt.close()

    print("All 4 report figures successfully generated in outputs/figures/!")

if __name__ == "__main__":
    main()
