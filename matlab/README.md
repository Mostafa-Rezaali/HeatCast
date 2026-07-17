# MATLAB plotting utilities

This folder contains MATLAB scripts for plotting HeatCast W34 NetCDF exports.

Continuous W34 z-score movie:

```matlab
addpath('matlab')
make_w34_truth_hindcast_movie('matlab_exports/w34_heatcast_ens_stack.nc', [], ...
    'Mode','continuous')
```

The continuous movie shows observed W34 truth on the left and HeatCast W34
hindcast on the right. Both are 14-day mean z-score fields.

Exceedance-probability movie:

```matlab
addpath('matlab')
make_w34_exceedance_movie('matlab_exports/w34_heatcast_ens_stack.nc')
```

For a shorter interactive run with a pause after each plotted frame:

```matlab
addpath('matlab')
runShort = make_w34_exceedance_movie('matlab_exports/w34_heatcast_ens_stack.nc', [], ...
    'FrameStep',14, ...
    'FramePauseSeconds',0.05);
runShort()
```

Both paths read the NetCDF one time slice at a time, so MATLAB does not need to
keep the full 3-D arrays in memory.

Expected NetCDF variables:

- `ground_truth_3d(y, x, time)`: continuous observed W34 mean z-score
- `model_output_3d(y, x, time)`: continuous HeatCast W34 mean z-score
- `truth_exceedance(y, x, time)`: observed W34 exceedance label
- `prob_heatcast_ens_stack(y, x, time)`: HeatCast+ENS exceedance probability
- `target_date_yyyymmdd(time)`: target window center date labels
- `lat`, `lon`: plotting grids
