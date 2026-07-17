function runMovie = make_w34_exceedance_movie(ncFile, outMovie, varargin)
%MAKE_W34_EXCEEDANCE_MOVIE Convenience runner for W34 exceedance probability movies.
%
% Normal use:
%   make_w34_exceedance_movie('matlab_exports/w34_heatcast_ens_stack.nc')
%
% Function-handle use:
%   runShort = make_w34_exceedance_movie('matlab_exports/w34_heatcast_ens_stack.nc', [], ...
%       'FrameStep', 14, 'FramePauseSeconds', 0.05);
%   runShort()
%
% Defaults are tuned for the HeatCast+ENS W34 exceedance movie:
%   truth_exceedance vs prob_heatcast_ens_stack
%   ProbabilityDisplayThreshold = 0.15
%   CLim = [0 0.5]
%   FrameStep = 7

if nargin < 1 || isempty(ncFile)
    ncFile = fullfile('matlab_exports', 'w34_heatcast_ens_stack.nc');
end
if nargin < 2
    outMovie = [];
end

args = [{'TruthVar', 'truth_exceedance', ...
         'HindcastVar', 'prob_heatcast_ens_stack', ...
         'ProbabilityDisplayThreshold', 0.15, ...
         'CLim', [0 0.5], ...
         'FrameStep', 7}, varargin];

runMovie = @() make_w34_truth_hindcast_movie(ncFile, outMovie, args{:});

if nargout == 0
    runMovie();
    clear runMovie;
end
end
