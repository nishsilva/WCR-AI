from pathlib import Path

# Change this if you want the project created somewhere else
root = Path("./")

dirs = [
    "apps/api",
    "apps/web/pages",
    "src/wcr_agent/agent",
    "src/wcr_agent/schemas",
    "src/wcr_agent/data_access",
    "src/wcr_agent/analysis",
    "src/wcr_agent/plotting",
    "src/wcr_agent/tools",
    "data/raw",
    "data/processed",
    "data/reference",
    "scripts",
]

files = [
    "apps/api/main.py",
    "apps/web/Home.py",
    "apps/web/pages/1_Chat.py",
    "apps/web/pages/2_Census_Explorer.py",
    "apps/web/pages/3_Ring_Detail.py",
    "src/wcr_agent/agent/orchestrator.py",
    "src/wcr_agent/agent/tool_registry.py",
    "src/wcr_agent/agent/prompts.py",
    "src/wcr_agent/schemas/census.py",
    "src/wcr_agent/schemas/filters.py",
    "src/wcr_agent/schemas/outputs.py",
    "src/wcr_agent/data_access/census.py",
    "src/wcr_agent/data_access/regions.py",
    "src/wcr_agent/data_access/io_utils.py",
    "src/wcr_agent/analysis/filter_census.py",
    "src/wcr_agent/analysis/summarize_census.py",
    "src/wcr_agent/analysis/displacement.py",
    "src/wcr_agent/analysis/yearly_counts.py",
    "src/wcr_agent/analysis/compare_groups.py",
    "src/wcr_agent/plotting/maps.py",
    "src/wcr_agent/plotting/distributions.py",
    "src/wcr_agent/plotting/comparisons.py",
    "src/wcr_agent/tools/filter_rings_tool.py",
    "src/wcr_agent/tools/summarize_rings_tool.py",
    "src/wcr_agent/tools/map_births_tool.py",
    "src/wcr_agent/tools/map_deaths_tool.py",
    "src/wcr_agent/tools/map_segments_tool.py",
    "src/wcr_agent/tools/compare_groups_tool.py",
    "src/wcr_agent/tools/export_results_tool.py",
    "data/raw/WCR_Master_Dataset_as_at_061319.csv",
    "data/processed/wcr_census.parquet",
    "data/reference/regions.geojson",
    "scripts/build_wcr_census.py",
    "src/wcr_agent/__init__.py",
]

# Create directories
for d in dirs:
    (root / d).mkdir(parents=True, exist_ok=True)

# Create files
for f in files:
    path = root / f
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

print(f"Project structure created under: {root.resolve()}")