function [opts, args] = vl_argparse(opts, args, varargin)

if ~isstruct(opts) && ~isobject(opts), error('OPTS must be a structure') ; end
if ~iscell(args), args = {args} ; end

recursive = true ;
merge = false ;

if numel(varargin) > 2
  error('There can be at most two options.') ;
end

for i = 1:numel(varargin)
  switch lower(varargin{i})
    case 'nonrecursive'
      recursive = false ;
    case 'merge'
      merge = true ;
    otherwise
      error('Unknown option specified.') ;
  end
end

opts_ = opts ;
optNames = fieldnames(opts)' ;

ai = 1 ;
keep = false(size(args)) ;
while ai <= numel(args)

  if recursive && isstruct(args{ai})
    params = fieldnames(args{ai})' ;
    values = struct2cell(args{ai})' ;
    if nargout <= 1
      opts = vl_argparse(opts, vertcat(params,values), varargin{:}) ;
    else
      [opts, rest] = vl_argparse(opts, reshape(vertcat(params,values), 1, []), varargin{:}) ;
      args{ai} = cell2struct(rest(2:2:end), rest(1:2:end), 2) ;
      keep(ai) = true ;
    end
    ai = ai + 1 ;
    continue ;
  end

  if ~isstr(args{ai})
    error('Expected either a param-value pair or a structure.') ;
  end

  param = args{ai} ;
  value = args{ai+1} ;
  
  if any(param == '.')
    parts = strsplit(param, '.') ;
    subs = struct('type', repmat({'.'}, 1, numel(parts) - 1), 'subs', parts(2:end)) ;
    param = parts{1} ;
    value = subsasgn(struct(), subs, value) ;
  end

  p = find(strcmpi(param, optNames)) ;
  if numel(p) ~= 1
    if merge 
      field = param ;
    elseif nargout <= 1
      error('Unknown parameter ''%s''', param) ;
    else
      keep([ai,ai+1]) = true ;
      ai = ai + 2 ;
      continue ;
    end
  else
    field = optNames{p} ;
  end

  if ~recursive
    opts.(field) = value ;
  else
    if isfield(opts_, field) && isstruct(opts_.(field)) && numel(fieldnames(opts_.(field))) > 0
      if ~isstruct(value)
        error('Cannot assign a non-struct value to the struct parameter ''%s''.', ...
          field) ;
      end
      if nargout > 1
        [opts.(field), args{ai+1}] = vl_argparse(opts.(field), value, varargin{:}) ;
      else
        opts.(field) = vl_argparse(opts.(field), value, varargin{:}) ;
      end
    else
      if ~isstruct(value)
        opts.(field) = value ;
      else
        opts.(field) = vl_argparse(opts.(field), value, 'merge') ;
      end
    end
  end

  ai = ai + 2 ;
end

args = args(keep) ;