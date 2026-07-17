function make_w34_continuous_movie(ncFile, outMovie, varargin)
%MAKE_W34_CONTINUOUS_MOVIE Plot continuous W34 z-score truth and hindcast fields.
%
% Usage:
%   addpath('matlab')
%   make_w34_continuous_movie('matlab_exports/w34_heatcast_ens_stack.nc')
%
% This plots:
%   left  = ground_truth_3d, observed 14-day W34 mean z-score
%   right = model_output_3d, HeatCast 14-day W34 mean z-score hindcast
%
% The exceedance-probability movie is separate:
%   make_w34_truth_hindcast_movie(..., 'TruthVar','truth_exceedance',
%       'HindcastVar','prob_heatcast_ens_stack')

if nargin < 1 || isempty(ncFile)
    ncFile = fullfile('matlab_exports', 'w34_heatcast_ens_stack.nc');
end
if nargin < 2 || isempty(outMovie)
    outMovie = fullfile('matlab', 'outputs', 'w34_continuous_truth_hindcast_movie.mp4');
end

make_w34_truth_hindcast_movie(ncFile, outMovie, ...
    'TruthVar', 'ground_truth_3d', ...
    'HindcastVar', 'model_output_3d', ...
    'CLim', [-2.5 2.5], ...
    varargin{:});

end
