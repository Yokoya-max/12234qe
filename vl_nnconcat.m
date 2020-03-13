function y = vl_nnconcat(inputs, dim, dzdy, varargin)

opts.inputSizes = [] ;
opts = vl_argparse(opts, varargin, 'nonrecursive') ;

if nargin < 2, dim = 3; end;
if nargin < 3, dzdy = []; end;

if isempty(dzdy)
  y = cat(dim, inputs{:});
else
  if isempty(opts.inputSizes)
    opts.inputSizes = cellfun(@(inp) [size(inp,1),size(inp,2),size(inp,3),size(inp,4)], inputs, 'UniformOutput', false) ;
  end
  start = 1 ;
  y = cell(1, numel(opts.inputSizes)) ;
  s.type = '()' ;
  s.subs = {':', ':', ':', ':'} ;
  for i = 1:numel(opts.inputSizes)
    stop = start + opts.inputSizes{i}(dim) ;
    s.subs{dim} = start:stop-1 ;
    y{i} = subsref(dzdy,s) ;
    start = stop ;
  end
end