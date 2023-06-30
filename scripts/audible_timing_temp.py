from pathlib import Path
import pandas as pd

fps = 29.97
audible_timing_path = Path(
    f"/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/6_Video/Day11_AM_Cam1_GH010630_audible_timings.txt")
audible_timing = pd.read_csv(audible_timing_path)
audible_timing = audible_timing.stack().reset_index().iloc[:, 1:]
audible_timing.columns = ['words', 'time']
audible_timing.to_csv(f"/users/rdarie/data/rdarie/Neural Recordings/raw/ISI-C-003/6_Video/Day11_AM_Cam1_GH010630_audible_timings_stacked.csv")