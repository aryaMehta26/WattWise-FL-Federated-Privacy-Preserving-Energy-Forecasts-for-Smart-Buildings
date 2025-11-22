"""Quick script to view EDA results summary."""
import json
from pathlib import Path

print("\n" + "="*60)
print("EDA RESULTS SUMMARY")
print("="*60)

# Temporal Analysis
with open('results/eda/temporal_analysis.json') as f:
    temporal = json.load(f)

print("\n📊 TEMPORAL ANALYSIS")
print("-" * 60)
print(f"Peak Hour: {temporal['hourly_analysis']['peak_hour']}:00")
print(f"Low Hour: {temporal['hourly_analysis']['low_hour']}:00")
print(f"Peak Month: {temporal['seasonal_analysis']['peak_month']}")
print(f"Weekday Avg: {temporal['daily_analysis']['weekday_average']:.2f} kWh")
print(f"Weekend Avg: {temporal['daily_analysis']['weekend_average']:.2f} kWh")
print(f"Date Range: {temporal['summary']['date_range']['days']} days")
print(f"Unique Buildings: {temporal['summary']['unique_buildings']}")

# Building Analysis
with open('results/eda/building_analysis.json') as f:
    building = json.load(f)

print("\n🏢 BUILDING ANALYSIS")
print("-" * 60)
print(f"Total Buildings: {building['summary']['total_buildings']}")
print(f"Unique Sites: {building['summary']['unique_sites']}")
print(f"Unique Use Types: {building['summary']['unique_use_types']}")
print(f"Avg Consumption: {building['summary']['avg_consumption']:.2f} kWh")
print(f"Highest Use Type: {building['use_type_analysis']['highest_consumption_type']}")
print(f"Lowest Use Type: {building['use_type_analysis']['lowest_consumption_type']}")

# Meter Analysis
with open('results/eda/meter_analysis.json') as f:
    meter = json.load(f)

print("\n⚡ METER ANALYSIS")
print("-" * 60)
print(f"Meter Types Analyzed: {', '.join(meter['summary']['meter_types_analyzed'])}")
for m in meter['meter_comparison']:
    print(f"  {m['meter_type'].upper():15s} - {m['n_buildings']:4d} buildings, avg={m['mean']:.2f}")

# Generated Files
print("\n📁 GENERATED FILES")
print("-" * 60)
eda_files = list(Path('results/eda').glob('*'))
print("EDA Reports:")
for f in eda_files:
    size = f.stat().st_size / 1024
    print(f"  ✓ {f.name:30s} ({size:>8.2f} KB)")

fig_files = list(Path('results/figures').glob('*.png'))
print("\nVisualizations:")
for f in fig_files:
    size = f.stat().st_size / 1024
    print(f"  ✓ {f.name:30s} ({size:>8.2f} KB)")

print("\n" + "="*60)
print("✅ All EDA implementations completed successfully!")
print("="*60 + "\n")

