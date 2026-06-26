function make_w34_truth_hindcast_movie(ncFile, outMovie, varargin)
%MAKE_W34_TRUTH_HINDCAST_MOVIE Create a side-by-side W34 truth/hindcast MP4.
%
% Usage:
%   make_w34_truth_hindcast_movie('w34_heatcast_ens_stack.nc')
%
% Optional name-value arguments:
%   'FrameStep'    : plot every Nth time slice. Default: 1
%   'FrameRate'    : movie frame rate. Default: 8
%   'StartIndex'   : first time index. Default: 1
%   'EndIndex'     : last time index. Default: all times
%   'CLim'         : color limits for z-score fields. Default: [-2.5 2.5]
%   'TruthVar'     : NetCDF truth variable. Default: 'ground_truth_3d'
%   'HindcastVar'  : NetCDF hindcast variable. Default: 'model_output_3d'
%   'BaseVar'      : Climatology probability variable for BSS. Default: 'prob_climatology'
%   'EventThreshold' : Probability threshold for hit/FAR. Default: 0.5
%   'ReliabilityBins': Number of reliability bins. Default: 10
%   'UseBasemap'   : Use MATLAB geographic axes basemap if available. Default: true
%   'Basemap'      : MATLAB basemap name. Default: 'grayland'
%   'MapPointStride': Geographic basemap point stride. Default: 3
%   'MapMarkerSize': Geographic basemap marker size. Default: 5
%   'ProbabilityDisplayThreshold': Hide lower probabilities on basemap. Default: 0.05
%   'VideoProfile' : VideoWriter profile. Default: auto MP4, fallback AVI
%
% The script reads one time slice at a time. Output arrays are displayed as
% latitude x longitude maps with time as the third dimension.

if nargin < 1 || isempty(ncFile)
    ncFile = 'w34_heatcast_ens_stack.nc';
end
if nargin < 2 || isempty(outMovie)
    outMovie = fullfile('matlab_plots', 'outputs', 'w34_truth_hindcast_movie.mp4');
end

p = inputParser;
addParameter(p, 'FrameStep', 1, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'FrameRate', 8, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'StartIndex', 1, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'EndIndex', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x >= 1));
addParameter(p, 'CLim', [-2.5 2.5], @(x) isnumeric(x) && numel(x) == 2);
addParameter(p, 'TruthVar', 'ground_truth_3d', @(x) ischar(x) || isstring(x));
addParameter(p, 'HindcastVar', 'model_output_3d', @(x) ischar(x) || isstring(x));
addParameter(p, 'BaseVar', 'prob_climatology', @(x) ischar(x) || isstring(x));
addParameter(p, 'EventThreshold', 0.5, @(x) isnumeric(x) && isscalar(x) && x >= 0 && x <= 1);
addParameter(p, 'ReliabilityBins', 10, @(x) isnumeric(x) && isscalar(x) && x >= 2);
addParameter(p, 'UseBasemap', true, @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
addParameter(p, 'Basemap', 'grayland', @(x) ischar(x) || isstring(x));
addParameter(p, 'MapPointStride', 3, @(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p, 'MapMarkerSize', 5, @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'ProbabilityDisplayThreshold', 0.05, @(x) isnumeric(x) && isscalar(x) && x >= 0 && x <= 1);
addParameter(p, 'VideoProfile', 'auto', @(x) ischar(x) || isstring(x));
parse(p, varargin{:});
opt = p.Results;

ncFile = char(ncFile);
ncFile = resolveNetcdfPath(ncFile);
outMovie = char(outMovie);
truthVar = char(opt.TruthVar);
hindcastVar = char(opt.HindcastVar);
baseVar = char(opt.BaseVar);
exceedanceMode = isExceedanceMovie(truthVar, hindcastVar);
if exceedanceMode && isequal(opt.CLim, [-2.5 2.5])
    opt.CLim = [0 1];
end
hasBaseField = exceedanceMode && hasVariable(ncFile, baseVar);

time = ncread(ncFile, 'time');
targetDate = ncread(ncFile, 'target_date_yyyymmdd');
initDate = readOptionalVector(ncFile, 'init_date_yyyymmdd');
windowLeads = readOptionalVector(ncFile, 'window_lead');
if isempty(windowLeads)
    windowLeads = 15:28;
end
nt = numel(time);
if isempty(opt.EndIndex)
    endIndex = nt;
else
    endIndex = min(nt, round(opt.EndIndex));
end
startIndex = max(1, round(opt.StartIndex));
frameStep = max(1, round(opt.FrameStep));

lat = ncread(ncFile, 'lat');
lon = ncread(ncFile, 'lon');

truthInfo = ncinfo(ncFile, truthVar);
truthSize = truthInfo.Size;
if numel(truthSize) ~= 3
    error('%s must be a 3-D variable.', truthVar);
end

% Read first slice by discovering the actual NetCDF time dimension. MATLAB's
% ncread dimension order can differ from the y,x,time wording used in Python.
[truth0, truthTimeDim] = readMatchedTimeSlice(ncFile, truthVar, startIndex, nt, lat, lon, []);
[hind0, hindTimeDim] = readMatchedTimeSlice(ncFile, hindcastVar, startIndex, nt, lat, lon, truthTimeDim);
baseTimeDim = [];
if hasBaseField
    [base0, baseTimeDim] = readMatchedTimeSlice(ncFile, baseVar, startIndex, nt, lat, lon, truthTimeDim); %#ok<NASGU>
end
[truth0, latPlot, lonPlot] = orientFieldToGrid(lat, lon, truth0);
hind0 = orientFieldToGrid(lat, lon, hind0);

lonLim = [min(lonPlot(:), [], 'omitnan'), max(lonPlot(:), [], 'omitnan')];
latLim = [min(latPlot(:), [], 'omitnan'), max(latPlot(:), [], 'omitnan')];

outDir = fileparts(outMovie);
if ~isempty(outDir) && ~exist(outDir, 'dir')
    mkdir(outDir);
end

[writer, outMovie] = makeVideoWriter(outMovie, char(opt.VideoProfile));
writer.FrameRate = opt.FrameRate;
if isprop(writer, 'Quality')
    writer.Quality = 95;
end
open(writer);
cleanupObj = onCleanup(@() close(writer));

fig = figure('Color', 'w', 'Position', [100 100 1600 720]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
if exceedanceMode
    colormap(fig, probabilityWhiteRed(256));
else
    colormap(fig, blueWhiteRed(256));
end

useBasemap = logical(opt.UseBasemap);
sampleMask = [];
try
    if useBasemap
        [axTruth, axHind, hTruth, hHind, sampleMask] = initializeBasemapPanels( ...
            tl, latPlot, lonPlot, truth0, hind0, opt, exceedanceMode, hindcastVar, latLim, lonLim);
    else
        error('Basemap disabled by UseBasemap=false.');
    end
catch err
    if useBasemap
        warning('Basemap plotting unavailable (%s). Falling back to native lat/lon surface axes.', err.message);
        try
            delete(findall(fig, 'Type', 'geoaxes'));
        catch
        end
    end
    useBasemap = false;
    [axTruth, axHind, hTruth, hHind] = initializeSurfacePanels( ...
        tl, latPlot, lonPlot, truth0, hind0, opt, exceedanceMode, hindcastVar, latLim, lonLim);
end

for t = startIndex:frameStep:endIndex
    truth = readMatchedTimeSlice(ncFile, truthVar, t, nt, lat, lon, truthTimeDim);
    hindcast = readMatchedTimeSlice(ncFile, hindcastVar, t, nt, lat, lon, hindTimeDim);
    if hasBaseField
        base = readMatchedTimeSlice(ncFile, baseVar, t, nt, lat, lon, baseTimeDim);
        base = orientFieldToGrid(lat, lon, base);
    else
        base = [];
    end
    truth = orientFieldToGrid(lat, lon, truth);
    hindcast = orientFieldToGrid(lat, lon, hindcast);

    if useBasemap
        truthMask = sampleMask.base & displayMaskForBasemap(truth, true, exceedanceMode, opt.ProbabilityDisplayThreshold);
        hindMask = sampleMask.base & displayMaskForBasemap(hindcast, false, exceedanceMode, opt.ProbabilityDisplayThreshold);
        hTruth = updateGeoScatter(hTruth, axTruth, latPlot, lonPlot, truth, truthMask, opt.MapMarkerSize);
        hHind = updateGeoScatter(hHind, axHind, latPlot, lonPlot, hindcast, hindMask, opt.MapMarkerSize);
        freezeGeoLimits(axTruth, latLim, lonLim);
        freezeGeoLimits(axHind, latLim, lonLim);
    else
        set(hTruth, 'CData', truth);
        set(hHind, 'CData', hindcast);
        setSurfaceAlpha(hTruth, truth);
        setSurfaceAlpha(hHind, hindcast);
    end

    [windowText, initText, centerText] = windowDateText(targetDate(t), initDate, t, windowLeads);
    if exceedanceMode
        metricsText = exceedanceMetricsText(truth, hindcast, base, opt.EventThreshold, round(opt.ReliabilityBins));
        title(axTruth, sprintf('Observed W34 exceedance label | %s', centerText));
        title(axHind, sprintf('%s probability | %s', hindcastDisplayName(hindcastVar), centerText));
        sgtitle(tl, sprintf(['W34 exceedance probability | %d-day mean over leads +%d..+%d (%s) | ', ...
            'init %s | frame %d/%d\n%s'], ...
            numel(windowLeads), min(windowLeads), max(windowLeads), windowText, initText, t, nt, metricsText), ...
            'FontWeight', 'bold');
    else
        metrics = fieldMetrics(truth, hindcast);
        title(axTruth, sprintf('Observed W34 truth | %s', centerText));
        title(axHind, sprintf('HeatCast W34 hindcast | %s', centerText));
        sgtitle(tl, sprintf(['W34 continuous z-score fields | %d-day mean over leads +%d..+%d (%s) | ', ...
            'init %s | frame %d/%d\nMAE=%.3f  RMSE=%.3f  bias=%.3f  r=%.3f  R2=%.3f  N=%s land cells'], ...
            numel(windowLeads), min(windowLeads), max(windowLeads), windowText, initText, t, nt, ...
            metrics.mae, metrics.rmse, metrics.bias, metrics.r, metrics.r2, formatInteger(metrics.n)), ...
            'FontWeight', 'bold');
    end

    drawnow;
    writeVideo(writer, getframe(fig));
end

fprintf('Wrote movie: %s\n', outMovie);

end

function values = readOptionalVector(ncFile, varName)
if hasVariable(ncFile, varName)
    values = ncread(ncFile, varName);
else
    values = [];
end
end

function tf = hasVariable(ncFile, varName)
info = ncinfo(ncFile);
names = {info.Variables.Name};
tf = any(strcmp(names, varName));
end

function enableMapGrid(ax)
grid(ax, 'on');
ax.GridColor = [0.25 0.25 0.25];
ax.GridAlpha = 0.35;
ax.Layer = 'top';
xticks(ax, -130:10:-60);
yticks(ax, 25:5:50);
end

function [axTruth, axHind, hTruth, hHind, sampleMask] = initializeBasemapPanels( ...
    tl, latPlot, lonPlot, truth0, hind0, opt, exceedanceMode, hindcastVar, latLim, lonLim)
stride = max(1, round(opt.MapPointStride));
baseMask = false(size(latPlot));
baseMask(1:stride:end, 1:stride:end) = true;
baseMask = baseMask & isfinite(latPlot) & isfinite(lonPlot);
sampleMask = struct();
sampleMask.base = baseMask;
if ~any(baseMask(:))
    error('No finite map points available for geographic basemap plotting.');
end
truthMask = baseMask & displayMaskForBasemap(truth0, true, exceedanceMode, opt.ProbabilityDisplayThreshold);
hindMask = baseMask & displayMaskForBasemap(hind0, false, exceedanceMode, opt.ProbabilityDisplayThreshold);

axTruth = geoaxes(tl);
axTruth.Layout.Tile = 1;
applyBasemap(axTruth, char(opt.Basemap));
freezeGeoLimits(axTruth, latLim, lonLim);
hTruth = makeGeoScatter(axTruth, latPlot, lonPlot, truth0, truthMask, opt.MapMarkerSize);
setScatterAlpha(hTruth);
freezeGeoLimits(axTruth, latLim, lonLim);
caxis(axTruth, opt.CLim);
colorbar(axTruth);
title(axTruth, initialTruthTitle(exceedanceMode));

axHind = geoaxes(tl);
axHind.Layout.Tile = 2;
applyBasemap(axHind, char(opt.Basemap));
freezeGeoLimits(axHind, latLim, lonLim);
hHind = makeGeoScatter(axHind, latPlot, lonPlot, hind0, hindMask, opt.MapMarkerSize);
setScatterAlpha(hHind);
freezeGeoLimits(axHind, latLim, lonLim);
caxis(axHind, opt.CLim);
colorbar(axHind);
title(axHind, initialHindcastTitle(exceedanceMode, hindcastVar));
end

function freezeGeoLimits(ax, latLim, lonLim)
geolimits(ax, latLim, lonLim);
try
    ax.LatitudeLimitsMode = 'manual';
    ax.LongitudeLimitsMode = 'manual';
catch
end
end

function mask = displayMaskForBasemap(field, isTruth, exceedanceMode, probabilityDisplayThreshold)
mask = isfinite(field);
if exceedanceMode && isTruth
    mask = mask & field > 0.5;
elseif exceedanceMode
    mask = mask & field >= probabilityDisplayThreshold;
end
end

function h = makeGeoScatter(ax, latGrid, lonGrid, field, mask, markerSize)
[latVals, lonVals, values] = maskedMapVectors(latGrid, lonGrid, field, mask);
h = geoscatter(ax, latVals, lonVals, markerSize, values, 'filled');
setScatterAlpha(h);
end

function h = updateGeoScatter(h, ax, latGrid, lonGrid, field, mask, markerSize)
[latVals, lonVals, values] = maskedMapVectors(latGrid, lonGrid, field, mask);
try
    set(h, 'LatitudeData', latVals, 'LongitudeData', lonVals, 'CData', values, 'SizeData', markerSize);
catch
    try
        delete(h);
    catch
    end
    h = geoscatter(ax, latVals, lonVals, markerSize, values, 'filled');
    setScatterAlpha(h);
end
end

function [latVals, lonVals, values] = maskedMapVectors(latGrid, lonGrid, field, mask)
if any(mask(:))
    latVals = latGrid(mask);
    lonVals = lonGrid(mask);
    values = field(mask);
else
    latVals = NaN;
    lonVals = NaN;
    values = NaN;
end
end

function setScatterAlpha(h)
try
    h.MarkerFaceAlpha = 0.72;
    h.MarkerEdgeAlpha = 0.0;
catch
end
end

function applyBasemap(ax, requestedBasemap)
try
    geobasemap(ax, requestedBasemap);
catch
    geobasemap(ax, 'darkwater');
end
end

function [axTruth, axHind, hTruth, hHind] = initializeSurfacePanels( ...
    tl, latPlot, lonPlot, truth0, hind0, opt, exceedanceMode, hindcastVar, latLim, lonLim)
lonLim = [min(lonPlot(:), [], 'omitnan'), max(lonPlot(:), [], 'omitnan')];
latLim = [min(latPlot(:), [], 'omitnan'), max(latPlot(:), [], 'omitnan')];

axTruth = nexttile(tl, 1);
hTruth = surface(axTruth, lonPlot, latPlot, zeros(size(truth0)), truth0, 'EdgeColor', 'none');
view(axTruth, 2);
set(axTruth, 'YDir', 'normal');
axis(axTruth, 'equal', 'tight');
xlim(axTruth, lonLim);
ylim(axTruth, latLim);
enableMapGrid(axTruth);
caxis(axTruth, opt.CLim);
colorbar(axTruth);
title(axTruth, initialTruthTitle(exceedanceMode));
xlabel(axTruth, 'Longitude');
ylabel(axTruth, 'Latitude');
setSurfaceAlpha(hTruth, truth0);

axHind = nexttile(tl, 2);
hHind = surface(axHind, lonPlot, latPlot, zeros(size(hind0)), hind0, 'EdgeColor', 'none');
view(axHind, 2);
set(axHind, 'YDir', 'normal');
axis(axHind, 'equal', 'tight');
xlim(axHind, lonLim);
ylim(axHind, latLim);
enableMapGrid(axHind);
caxis(axHind, opt.CLim);
colorbar(axHind);
title(axHind, initialHindcastTitle(exceedanceMode, hindcastVar));
xlabel(axHind, 'Longitude');
ylabel(axHind, 'Latitude');
setSurfaceAlpha(hHind, hind0);
end

function tf = isExceedanceMovie(truthVar, hindcastVar)
tf = strcmpi(truthVar, 'truth_exceedance') || startsWith(lower(hindcastVar), 'prob_');
end

function s = initialTruthTitle(exceedanceMode)
if exceedanceMode
    s = 'Observed W34 exceedance label';
else
    s = 'Observed W34 truth';
end
end

function s = initialHindcastTitle(exceedanceMode, hindcastVar)
if exceedanceMode
    s = sprintf('%s probability', hindcastDisplayName(hindcastVar));
else
    s = 'HeatCast W34 hindcast';
end
end

function s = hindcastDisplayName(hindcastVar)
switch lower(hindcastVar)
    case 'prob_heatcast_ens_stack'
        s = 'HeatCast+ENS stack';
    case 'prob_heatcast_c'
        s = 'HeatCast-C';
    case 'prob_ens_calibrated'
        s = 'ENS calibrated';
    case 'prob_ens_raw_fraction'
        s = 'ENS raw fraction';
    otherwise
        s = strrep(hindcastVar, '_', '\_');
end
end

function text = exceedanceMetricsText(truth, prob, base, eventThreshold, reliabilityBins)
metrics = exceedanceMetrics(truth, prob, base, eventThreshold, reliabilityBins);
if isfinite(metrics.bss)
    bssText = sprintf('BSS=%.3f', metrics.bss);
else
    bssText = 'BSS=NA';
end
text = sprintf(['Brier=%.4f  %s  ECE=%.3f  rel_slope=%.3f  hit=%.3f  FAR=%.3f  ', ...
    'obs=%.3f  mean_p=%.3f  thr=%.2f  N=%s land cells'], ...
    metrics.brier, bssText, metrics.ece, metrics.slope, metrics.hitRate, metrics.falseAlarmRate, ...
    metrics.obsRate, metrics.meanProb, eventThreshold, formatInteger(metrics.n));
end

function metrics = exceedanceMetrics(truth, prob, base, eventThreshold, reliabilityBins)
valid = isfinite(truth) & isfinite(prob);
y = double(truth(valid) > 0.5);
p = min(max(double(prob(valid)), 0), 1);
metrics.n = numel(y);
if metrics.n == 0
    metrics = emptyExceedanceMetrics(metrics.n);
    return;
end

metrics.brier = mean((p - y) .^ 2);
metrics.obsRate = mean(y);
metrics.meanProb = mean(p);

if ~isempty(base)
    b = min(max(double(base(valid)), 0), 1);
    baseValid = isfinite(b);
    if any(baseValid)
        baseBrier = mean((b(baseValid) - y(baseValid)) .^ 2);
        modelBrierForBss = mean((p(baseValid) - y(baseValid)) .^ 2);
        metrics.bss = 1 - modelBrierForBss / (baseBrier + eps);
    else
        metrics.bss = NaN;
    end
else
    metrics.bss = NaN;
end

predEvent = p >= eventThreshold;
hits = sum(predEvent & (y == 1));
misses = sum(~predEvent & (y == 1));
falseAlarms = sum(predEvent & (y == 0));
correctNegatives = sum(~predEvent & (y == 0));
metrics.hitRate = hits / max(hits + misses, 1);
metrics.falseAlarmRate = falseAlarms / max(falseAlarms + correctNegatives, 1);

[metrics.ece, metrics.slope] = reliabilityMetrics(y, p, reliabilityBins);
end

function metrics = emptyExceedanceMetrics(n)
metrics.n = n;
metrics.brier = NaN;
metrics.bss = NaN;
metrics.ece = NaN;
metrics.slope = NaN;
metrics.hitRate = NaN;
metrics.falseAlarmRate = NaN;
metrics.obsRate = NaN;
metrics.meanProb = NaN;
end

function [ece, slope] = reliabilityMetrics(y, p, reliabilityBins)
edges = linspace(0, 1, reliabilityBins + 1);
ece = 0;
binPred = [];
binObs = [];
binW = [];
for i = 1:reliabilityBins
    if i == reliabilityBins
        inBin = p >= edges(i) & p <= edges(i + 1);
    else
        inBin = p >= edges(i) & p < edges(i + 1);
    end
    n = sum(inBin);
    if n == 0
        continue;
    end
    mp = mean(p(inBin));
    mo = mean(y(inBin));
    w = n / numel(y);
    ece = ece + w * abs(mp - mo);
    binPred(end + 1, 1) = mp; %#ok<AGROW>
    binObs(end + 1, 1) = mo; %#ok<AGROW>
    binW(end + 1, 1) = n; %#ok<AGROW>
end
if numel(binPred) >= 2 && std(binPred) > 1e-8
    x = [ones(size(binPred)), binPred];
    sw = sqrt(binW);
    beta = (x .* sw) \ (binObs .* sw);
    slope = beta(2);
else
    slope = NaN;
end
end

function metrics = fieldMetrics(truth, pred)
valid = isfinite(truth) & isfinite(pred);
t = double(truth(valid));
p = double(pred(valid));
metrics.n = numel(t);
if metrics.n == 0
    metrics.mae = NaN;
    metrics.rmse = NaN;
    metrics.bias = NaN;
    metrics.r = NaN;
    metrics.r2 = NaN;
    return;
end
err = p - t;
metrics.mae = mean(abs(err));
metrics.rmse = sqrt(mean(err .^ 2));
metrics.bias = mean(err);
sse = sum((t - p) .^ 2);
sst = sum((t - mean(t)) .^ 2);
metrics.r2 = 1 - sse / (sst + eps);
if std(t) > 1e-8 && std(p) > 1e-8
    c = corrcoef(t, p);
    metrics.r = c(1, 2);
else
    metrics.r = NaN;
end
end

function [windowText, initText, centerText] = windowDateText(targetYmd, initDate, frameIndex, windowLeads)
centerDt = ymdToDatetime(targetYmd);
centerText = datetimeLabel(centerDt);
if ~isempty(initDate) && numel(initDate) >= frameIndex && isfinite(double(initDate(frameIndex))) && initDate(frameIndex) > 10000000
    initDt = ymdToDatetime(initDate(frameIndex));
    initText = datetimeLabel(initDt);
    startDt = initDt + days(min(windowLeads));
    endDt = initDt + days(max(windowLeads));
else
    initText = 'unavailable';
    n = numel(windowLeads);
    startDt = centerDt - days(floor((n - 1) / 2));
    endDt = startDt + days(n - 1);
end
windowText = sprintf('%s to %s', datetimeLabel(startDt), datetimeLabel(endDt));
end

function dt = ymdToDatetime(value)
txt = sprintf('%08d', round(double(value)));
dt = datetime(str2double(txt(1:4)), str2double(txt(5:6)), str2double(txt(7:8)));
end

function label = datetimeLabel(dt)
dt.Format = 'yyyy-MM-dd';
label = char(dt);
end

function s = formatInteger(value)
if ~isfinite(value)
    s = '0';
else
    s = sprintf('%.0f', value);
end
end

function [writer, outMovie] = makeVideoWriter(outMovie, requestedProfile)
% Prefer MP4, but fall back to Motion JPEG AVI when MATLAB lacks MPEG-4.
if strcmpi(requestedProfile, 'auto')
    profiles = {'MPEG-4', 'Motion JPEG AVI'};
else
    profiles = {requestedProfile};
end

lastError = [];
for i = 1:numel(profiles)
    profile = profiles{i};
    candidate = outMovieWithProfileExtension(outMovie, profile);
    try
        writer = VideoWriter(candidate, profile);
        requestedMovie = outMovie;
        outMovie = candidate;
        if ~strcmp(candidate, requestedMovie)
            fprintf('Using video profile %s: %s\n', profile, candidate);
        else
            fprintf('Using video profile %s: %s\n', profile, outMovie);
        end
        return;
    catch err
        lastError = err;
        if ~strcmpi(requestedProfile, 'auto')
            rethrow(err);
        end
    end
end

error('No supported VideoWriter profile found. Last error: %s', lastError.message);
end

function outMovie = outMovieWithProfileExtension(outMovie, profile)
[folder, name, ext] = fileparts(outMovie);
if strcmpi(profile, 'Motion JPEG AVI')
    wantedExt = '.avi';
elseif strcmpi(profile, 'MPEG-4')
    wantedExt = '.mp4';
else
    wantedExt = ext;
end
if isempty(ext) || ~strcmpi(ext, wantedExt)
    outMovie = fullfile(folder, [name wantedExt]);
end
end

function ncFile = resolveNetcdfPath(ncFile)
% Resolve common local/HPC export locations and fail with useful candidates.
if isfile(ncFile)
    return;
end

[~, name, ext] = fileparts(ncFile);
if isempty(ext)
    ext = '.nc';
end

candidates = {
    ncFile
    fullfile('matlab_plots', [name ext])
    fullfile('matlab_exports', [name ext])
    fullfile('.', [name ext])
    fullfile('matlab_plots', 'w34_heatcast_ens_stack.nc')
    fullfile('matlab_exports', 'w34_heatcast_ens_stack.nc')
    fullfile('.', 'w34_heatcast_ens_stack.nc')
    };

for i = 1:numel(candidates)
    if isfile(candidates{i})
        ncFile = candidates{i};
        fprintf('Using NetCDF file: %s\n', ncFile);
        return;
    end
end

matches = [dir('*.nc'); dir(fullfile('matlab_plots', '*.nc')); dir(fullfile('matlab_exports', '*.nc'))];
fprintf('Could not open requested NetCDF: %s\n', ncFile);
if ~isempty(matches)
    fprintf('Available NetCDF candidates:\n');
    for i = 1:numel(matches)
        fprintf('  %s\n', fullfile(matches(i).folder, matches(i).name));
    end
end
error('NetCDF file not found. Pass one of the available paths printed above.');
end

function [field, timeDim] = readMatchedTimeSlice(ncFile, varName, t, nt, lat, lon, preferredTimeDim)
% Read one time slice and return a 2-D field matching lat/lon or their transpose.
info = ncinfo(ncFile, varName);
sz = info.Size;
if numel(sz) ~= 3
    error('%s must be a 3-D variable.', varName);
end

dimNames = {info.Dimensions.Name};
timeByName = find(strcmp(dimNames, 'time'), 1);
timeBySize = find(sz == nt);
candidates = unique([preferredTimeDim, timeByName, timeBySize, 1:numel(sz)], 'stable');
candidates = candidates(candidates >= 1 & candidates <= numel(sz));

for i = 1:numel(candidates)
    dim = candidates(i);
    if sz(dim) < t
        continue;
    end
    start = ones(1, numel(sz));
    count = sz;
    start(dim) = t;
    count(dim) = 1;
    raw = squeeze(ncread(ncFile, varName, start, count));
    if isvector(raw)
        continue;
    end
    if isGridCompatible(raw, lat, lon)
        field = raw;
        timeDim = dim;
        return;
    end
end

tried = sprintf('%d ', candidates);
error('%s: could not read a 2-D time slice compatible with lat/lon. var size=%s, lat size=%s, lon size=%s, tried time dims=[%s].', ...
    varName, mat2str(sz), mat2str(size(lat)), mat2str(size(lon)), strtrim(tried));
end

function ok = isGridCompatible(field, lat, lon)
ok = (isequal(size(field), size(lat)) && isequal(size(field), size(lon))) || ...
     (isequal(size(field), size(lat')) && isequal(size(field), size(lon')));
end

function [fieldOut, latOut, lonOut] = orientFieldToGrid(lat, lon, field)
% Return field in the same orientation as the native lat/lon grid.
if isequal(size(field), size(lat)) && isequal(size(field), size(lon))
    fieldOut = field;
elseif isequal(size(field'), size(lat)) && isequal(size(field'), size(lon))
    fieldOut = field';
else
    error('field size %s does not match lat/lon size %s or its transpose.', mat2str(size(field)), mat2str(size(lat)));
end
latOut = lat;
lonOut = lon;
end

function setSurfaceAlpha(h, field)
% Hide ocean/unavailable NaNs while keeping valid grid cells opaque.
set(h, 'AlphaData', double(isfinite(field)), 'FaceAlpha', 'flat', 'AlphaDataMapping', 'none');
end

function [latOut, lonOut] = orientGridToField(lat, lon, field)
% Match lat/lon orientation to a displayed 2-D field.
if isequal(size(lat), size(field)) && isequal(size(lon), size(field))
    latOut = lat;
    lonOut = lon;
elseif isequal(size(lat'), size(field)) && isequal(size(lon'), size(field))
    latOut = lat';
    lonOut = lon';
else
    error('lat/lon size %s does not match field size %s.', mat2str(size(lat)), mat2str(size(field)));
end
end

function s = dateLabel(value)
value = double(value);
if isfinite(value) && value > 10000000
    txt = sprintf('%08d', round(value));
    s = sprintf('%s-%s-%s', txt(1:4), txt(5:6), txt(7:8));
else
    s = sprintf('time index %g', value);
end
end

function cmap = blueWhiteRed(n)
if nargin < 1
    n = 256;
end
n = max(2, round(n));
half = floor(n / 2);
blue = [0.103, 0.318, 0.639];
white = [1, 1, 1];
red = [0.698, 0.094, 0.168];
lower = [linspace(blue(1), white(1), half)', linspace(blue(2), white(2), half)', linspace(blue(3), white(3), half)'];
upper = [linspace(white(1), red(1), n - half)', linspace(white(2), red(2), n - half)', linspace(white(3), red(3), n - half)'];
cmap = [lower; upper];
end

function cmap = probabilityWhiteRed(n)
if nargin < 1
    n = 256;
end
n = max(2, round(n));
white = [1, 1, 1];
lightRed = [0.984, 0.705, 0.682];
red = [0.698, 0.094, 0.168];
breakPoint = max(2, round(0.55 * n));
lower = [linspace(white(1), lightRed(1), breakPoint)', ...
         linspace(white(2), lightRed(2), breakPoint)', ...
         linspace(white(3), lightRed(3), breakPoint)'];
upperN = n - breakPoint;
if upperN > 0
    upper = [linspace(lightRed(1), red(1), upperN)', ...
             linspace(lightRed(2), red(2), upperN)', ...
             linspace(lightRed(3), red(3), upperN)'];
    cmap = [lower; upper];
else
    cmap = lower;
end
end
