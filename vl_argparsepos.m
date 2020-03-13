function [opts, pos, unknown] = vl_argparsepos(opts, args, varargin)

  idx = find(strcmpi(varargin, 'flags'), 1) ;
  if ~isempty(idx)
    assert(idx + 1 <= numel(varargin) && iscellstr(varargin{idx + 1}), ...
      'Expected a cell array of strings after ''flags'' option.');
    
    flags = varargin{idx + 1} ;
    varargin(idx : idx + 1) = [] ;
    
    usedFlag = false(size(flags)) ;
    for i = 1:numel(flags)
      pos = find(strcmpi(flags{i}, args), 1) ;
      if ~isempty(pos)
        usedFlag(i) = true ;
        args(pos) = [] ;
      end
    end
  else 
    flags = {} ;
    usedFlag = [] ;
  end

  idx = (numel(args) - 1 : -2 : 1) ;
  
  if isempty(idx)
    firstPair = numel(args) + 1 ;
  else
    pos = find(~cellfun(@ischar, args(idx)), 1) ;

    if isempty(pos) 
      firstPair = idx(end) ;
    else 
      firstPair = idx(pos) + 2 ;
    end
  end

  namedArgs = args(firstPair:end) ;
  pos = args(1:firstPair-1) ;

  if nargout >= 3
    [opts, unknown] = vl_argparse(opts, namedArgs, varargin{:}) ;
    unknown = [unknown, flags(usedFlag)] ;
  else
    opts = vl_argparse(opts, namedArgs, varargin{:}) ;
    assert(isempty(flags), 'Cannot specify flags with less than 3 return values.') ;
  end
  
end