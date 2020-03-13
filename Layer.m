classdef Layer < matlab.mixin.Copyable


  properties (Access = public)
    inputs = {}  % list of inputs, either constants or other Layers
  end
  
  properties (SetAccess = public, GetAccess = public)
    func = []  % main function being called
    name = ''  % optional name (for debugging mostly; a layer is a unique handle object that can be passed around)
    numOutputs = []  % to manually specify the number of outputs returned in fwd mode
    numInputDer = []  % to manually specify the number of input derivatives returned in bwd mode
    accumDer = true  % to manually specify that the input derivatives are *not* accumulated. used to implement ReLU short-circuiting.
    meta = []  % optional meta properties
    source = []  % call stack (source files and line numbers) where this Layer was created
    diagnostics = []  % whether to plot the mean, min and max of the Layer's output var. empty for automatic (network outputs only).
    optimize = []  % whether to optimize this Layer (e.g. merge vl_nnwsum), empty for function-dependent default
    debugStop = false % call `keyboard` during forward pass
    precious = true
  end
  
  properties (SetAccess = {?Net}, GetAccess = public)
    outputVar = 0  % index of the output var in a Net, used during its construction
    id = []  % unique ID assigned on creation; does not have to persist, just be different from all other Layers in memory
  end
  
  properties (Access = protected)
    copied = []  % reference of deep copied object, used internally for deepCopy()
    enableCycleChecks = true  % to avoid redundant cycle checks when implicitly calling set.inputs()
  end
  
  methods (Static)
    varargout = create(func, args, varargin)
    generator = fromFunction(func, varargin)
    netOutputs = fromCompiledNet(net)
    netOutputs = fromDagNN(dag, customFn)
    workspaceNames(modifier)
  end
  
  methods
    function obj = Layer(func, varargin)  % wrap a function call
      obj.saveStack() ;  % record source file and line number, for debugging
      
      obj.id = Layer.uniqueId() ;  % set unique ID, needed for some operations
      
      if nargin == 0 && (isa(obj, 'Input') || isa(obj, 'Param') || isa(obj, 'Selector'))
        return  % called during Input, Param or Selector construction, nothing to do
      end
      
      % if not a function handle, assume DagNN or SimpleNN, and convert
      if ~isa(func, 'function_handle')
        assert(isstruct(func) || isa(func, 'dagnn.DagNN'), ...
          'Input must be a function handle, a SimpleNN or a DagNN.') ;
        
        obj = Layer.fromDagNN(func) ;
        if isscalar(obj)
          obj = obj{1} ;
        else  % wrap multiple outputs in a weighted sum (this is a constructor, only 1 object out)
          for i = 1:numel(obj)  % keep original objects, don't let them be optimized into a single weighted sum
            obj{i}.optimize = false ;
          end
           obj = Layer(@vl_nnwsum, obj{:}, 'weights', ones(1, numel(obj))) ;
        end
        return
      end
      
      obj.enableCycleChecks = false ;
      obj.func = func ;
      obj.inputs = varargin(:)' ;
      obj.enableCycleChecks = true ;
    end
    
    function set.inputs(obj, newInputs)
      if obj.enableCycleChecks
        % must check for cycles, to ensure DAG structure.
        visited = Layer.initializeRecursion() ;
        for i = 1:numel(newInputs)
          if isa(newInputs{i}, 'Layer')
            newInputs{i}.cycleCheckRecursive(obj, visited) ;
          end
        end
      end
      
      obj.inputs = newInputs;
    end
    function y = vl_nnpool(obj, varargin)
      y = Layer(@vl_nnpool, obj, varargin{:}) ;
    end
    function y = vl_nnrelu(obj, varargin)
      y = Layer(@vl_nnrelu, obj, varargin{:}) ;
    end
    function y = vl_nnsigmoid(obj, varargin)
      y = Layer(@vl_nnsigmoid, obj, varargin{:}) ;
    end
    function y = vl_nnbilinearsampler(obj, varargin)
      y = Layer(@vl_nnbilinearsampler, obj, varargin{:}) ;
    end
    function y = vl_nnaffinegrid(obj, varargin)
      y = Layer(@vl_nnaffinegrid, obj, varargin{:}) ;
    end
    function y = vl_nncrop(obj, varargin)
      y = Layer(@vl_nncrop, obj, varargin{:}) ;
    end
    function y = vl_nnnoffset(obj, varargin)
      y = Layer(@vl_nnnoffset, obj, varargin{:}) ;
    end
    function y = vl_nnnormalize(obj, varargin)
      y = Layer(@vl_nnnormalize, obj, varargin{:}) ;
    end
    function y = vl_nnnormalizelp(obj, varargin)
      y = Layer(@vl_nnnormalizelp, obj, varargin{:}) ;
    end
    function y = vl_nnspnorm(obj, varargin)
      y = Layer(@vl_nnspnorm, obj, varargin{:}) ;
    end
    function y = vl_nnsoftmax(obj, varargin)
      y = Layer(@vl_nnsoftmax, obj, varargin{:}) ;
    end
    function y = vl_nnpdist(obj, varargin)
      y = Layer(@vl_nnpdist, obj, varargin{:}) ;
    end
    function y = vl_nnsoftmaxloss(obj, varargin)
      y = Layer(@vl_nnsoftmaxloss, obj, varargin{:}) ;
      y.numInputDer = 1 ;  % only the first derivative is defined
    end
    function [hn, cn] = vl_nnlstm(varargin)
      [hn, cn] = Layer.create(@vl_nnlstm, varargin) ;
    end
    function y = vl_nnmaxout(obj, varargin)
      y = Layer(@vl_nnmaxout, obj, varargin{:}) ;
    end
    
    function y = reshape(obj, varargin)
      y = Layer(@reshape, obj, varargin{:}) ;
      y.numInputDer = 1 ;  % only the first derivative is defined
      y.precious = false ; 
    end
    function y = repmat(obj, varargin)
      y = Layer(@repmat, obj, varargin{:}) ;
      y.numInputDer = 1 ;  % only the first derivative is defined
      y.precious = false ; 
    end
    function y = repelem(obj,varargin)
      y = Layer(@repelem,obj,varargin{:});
      y.numInputDer = 1 ;  % only the first derivative is defined
      y.precious = false ; 
    end
    function y = permute(obj, varargin)
      y = Layer(@permute, obj, varargin{:}) ;
      y.precious = false ; 
    end
    function y = ipermute(obj, varargin)
      y = Layer(@ipermute, obj, varargin{:}) ;
      y.precious = false ; 
    end
    function y = shiftdim(obj, varargin)
      y = Layer(@shiftdim, obj, varargin{:}) ;
      y.precious = false ; 
    end
    function y = squeeze(obj, varargin)
      y = Layer(@squeeze, obj, varargin{:}) ;
      y.precious = false ; 
    end
    function y = flip(obj, varargin)
      y = Layer(@flip, obj, varargin{:}) ;
      y.precious = false ; 
    end
    function y = flipud(obj)
      y = Layer(@flip, obj, 1) ;
      y.precious = false ; 
    end
    function y = fliplr(obj)
      y = Layer(@flip, obj, 2) ;
      y.precious = false ; 
    end
    function y = rot90(obj, varargin)
      y = Layer(@rot90, obj, varargin{:}) ;
      y.precious = false ; 
    end
    function y = circshift(obj, varargin)
      y = Layer(@circshift, obj, varargin{:}) ;
      y.precious = false ; 
    end
    function y = size(obj, varargin)
      y = Layer(@size, obj, varargin{:}) ;
      y.numInputDer = 0 ;  % non-differentiable
    end
    function y = sum(obj, varargin)
      y = Layer(@sum, obj, varargin{:}) ;
      y.precious = false ; 
    end
    function y = mean(obj, varargin)
      y = Layer(@mean, obj, varargin{:}) ;
      y.precious = false ; 
    end
    function y = max(obj, varargin)
      y = Layer(@max, obj, varargin{:}) ;
    end
    function y = min(obj, varargin)
      y = Layer(@min, obj, varargin{:}) ;
    end
    function y = abs(obj, varargin)
      y = Layer(@abs, obj, varargin{:}) ;
    end
    function y = sqrt(obj, varargin)
      y = Layer(@sqrt, obj, varargin{:}) ;
    end
    function y = exp(obj, varargin)
      y = Layer(@exp, obj, varargin{:}) ;
    end
    function y = log(obj, varargin)
      y = Layer(@log, obj, varargin{:}) ;
    end
    function y = sin(obj, varargin)
      y = Layer(@sin, obj, varargin{:}) ;
    end
    function y = cos(obj, varargin)
      y = Layer(@cos, obj, varargin{:}) ;
    end
    function y = tan(obj, varargin)
      y = Layer(@tan, obj, varargin{:}) ;
    end
    function y = asin(obj, varargin)
      y = Layer(@asin, obj, varargin{:}) ;
    end
    function y = acos(obj, varargin)
      y = Layer(@acos, obj, varargin{:}) ;
    end
    function y = atan(obj, varargin)
      y = Layer(@atan, obj, varargin{:}) ;
    end
    function y = atan2(obj, varargin)
      y = Layer(@atan2, obj, varargin{:}) ;
    end
    function y = inv(obj, varargin)
      y = Layer(@inv, obj, varargin{:}) ;
    end
    function y = cat(obj, varargin)
      y = Layer(@cat, obj, varargin{:}) ;
      y.precious = false ; 
    end
    function y = accumarray(obj, varargin)
      y = Layer(@accumarray, obj, varargin{:}) ;
    end
    function y = gpuArray(obj)
      y = Layer(@gpuArray_wrapper, obj, Input('gpuMode')) ;
      y.numInputDer = 1 ;
    end
    function y = gather(obj)
      y = Layer(@gather, obj) ;
      y.precious = false ;
    end
    function y = single(obj)
      y = Layer(@single, obj) ;
      y.precious = false ;
    end
    function y = double(obj)
      y = Layer(@double, obj) ;
      y.precious = false ;
    end
    function y = complex(obj, varargin)
      y = Layer(@complex, obj, varargin{:}) ;
      y.precious = false ;
    end
    
    % overloaded matrix creation operators (no derivative).
    function y = rand(obj, varargin)
      y = Layer(@rand, obj, varargin{:}) ;
      y.numInputDer = 0 ;  % non-differentiable
    end
    function y = randi(obj, varargin)
      y = Layer(@randi, obj, varargin{:}) ;
      y.numInputDer = 0 ;  % non-differentiable
    end
    function y = randn(obj, varargin)
      y = Layer(@randn, obj, varargin{:}) ;
      y.numInputDer = 0 ;  % non-differentiable
    end
    function y = randperm(obj, varargin)
      y = Layer(@randperm, obj, varargin{:}) ;
      y.numInputDer = 0 ;  % non-differentiable
    end
    function y = zeros(obj, varargin)
      y = Layer(@zeros, obj, varargin{:}) ;
      y.numInputDer = 0 ;  % non-differentiable
    end
    function y = ones(obj, varargin)
      y = Layer(@ones, obj, varargin{:}) ;
      y.numInputDer = 0 ;  % non-differentiable
    end
    function y = inf(obj, varargin)
      y = Layer(@inf, obj, varargin{:}) ;
      y.numInputDer = 0 ;  % non-differentiable
    end
    function y = nan(obj, varargin)
      y = Layer(@nan, obj, varargin{:}) ;
      y.numInputDer = 0 ;  % non-differentiable
    end
    function y = eye(obj, varargin)
      y = Layer(@eye, obj, varargin{:}) ;
      y.numInputDer = 0 ;  % non-differentiable
    end
    
    
    function y = ne(a, b)
      y = Layer(@ne, a, b) ;
      y.numInputDer = 0 ;  % non-differentiable
    end
    function y = lt(a, b)
      y = Layer(@lt, a, b) ;
      y.numInputDer = 0 ;  % non-differentiable
    end
    function y = gt(a, b)
      y = Layer(@gt, a, b) ;
      y.numInputDer = 0 ;  % non-differentiable
    end
    function y = le(a, b)
      y = Layer(@le, a, b) ;
      y.numInputDer = 0 ;  % non-differentiable
    end
    function y = ge(a, b)
      y = Layer(@ge, a, b) ;
      y.numInputDer = 0 ;  % non-differentiable
    end
    
    function y = and(a, b)
      y = Layer(@and, a, b) ;
      y.numInputDer = 0 ;  % non-differentiable
    end
    function y = or(a, b)
      y = Layer(@or, a, b) ;
      y.numInputDer = 0 ;  % non-differentiable
    end
    function y = not(a)
      y = Layer(@not, a) ;
      y.numInputDer = 0 ;  % non-differentiable
    end
    function y = xor(a, b)
      y = Layer(@xor, a, b) ;
      y.numInputDer = 0 ;  % non-differentiable
    end
    function y = any(obj, varargin)
      y = Layer(@any, obj, varargin{:}) ;
      y.numInputDer = 0 ;  % non-differentiable
    end
    function y = all(obj, varargin)
      y = Layer(@all, obj, varargin{:}) ;
      y.numInputDer = 0 ;  % non-differentiable
    end
    function y = nnz(obj, varargin)
      y = Layer(@nnz, obj, varargin{:}) ;
      y.numInputDer = 0 ;  % non-differentiable
    end
    
    function y = vl_nnwsum(obj, varargin)
      y = Layer(@vl_nnwsum, obj, varargin{:}) ;
      y.precious = false ; 
    end
    
    function c = plus(a, b)
      c = vl_nnwsum(a, b, 'weights', [1, 1]) ;
    end
    function c = minus(a, b)
      c = vl_nnwsum(a, b, 'weights', [1, -1]) ;
    end
    function c = uminus(a)
      c = vl_nnwsum(a, 'weights', -1) ;
    end
    function c = uplus(a)
      c = a ;
    end
    
    function c = times(a, b)
      % optimization: for simple scalar constants, use a vl_nnwsum layer
      if isnumeric(a) && isscalar(a)
        c = vl_nnwsum(b, 'weights', a) ;
      elseif isnumeric(b) && isscalar(b)
        c = vl_nnwsum(a, 'weights', b) ;
      else  % general case
        c = Layer(@bsxfun, @times, a, b) ;
      end
    end
    function c = rdivide(a, b)
      if isnumeric(b) && isscalar(b)  % optimization for scalar constants
        c = vl_nnwsum(a, 'weights', 1 / b) ;
      else
        c = Layer(@bsxfun, @rdivide, a, b) ;
      end
    end
    function c = ldivide(a, b)
      if isnumeric(a) && isscalar(a)  % optimization for scalar constants
        c = vl_nnwsum(b, 'weights', 1 / a) ;
      else
        % @ldivide is just @rdivide with swapped inputs
        c = Layer(@bsxfun, @rdivide, b, a) ;
      end
    end
    function c = power(a, b)
      c = Layer(@bsxfun, @power, a, b) ;
    end
    
    function y = transpose(a)
      y = Layer(@transpose, a) ;
      y.precious = false ; 
    end
    function y = ctranspose(a)
      y = Layer(@ctranspose, a) ;
      y.precious = false ; 
    end
    
    function c = mtimes(a, b)
      % optimization: for simple scalar constants, use a vl_nnwsum layer
      if isnumeric(a) && isscalar(a)
        c = vl_nnwsum(b, 'weights', a) ;
      elseif isnumeric(b) && isscalar(b)
        c = vl_nnwsum(a, 'weights', b) ;
      else  % general case
        c = Layer(@mtimes, a, b) ;
      end
    end
    function c = mrdivide(a, b)
      if isnumeric(b) && isscalar(b)  % optimization for scalar constants
        c = vl_nnwsum(a, 'weights', 1 / b) ;
      else
        c = Layer(@mrdivide, a, b) ;
      end
    end
    function c = mldivide(a, b)
      if isnumeric(a) && isscalar(a)  % optimization for scalar constants
        c = vl_nnwsum(b, 'weights', 1 / a) ;
      else
        % @mldivide is just @mrdivide with swapped inputs
        c = Layer(@mrdivide, b, a) ;
      end
    end
    
    function y = vertcat(obj, varargin)
      y = Layer(@cat, 1, obj, varargin{:}) ;
      y.precious = false ; 
    end
    function y = horzcat(obj, varargin)
      y = Layer(@cat, 2, obj, varargin{:}) ;
      y.precious = false ; 
    end
    
    function y = colon(obj, varargin)
      y = Layer(@colon, obj, varargin{:}) ;
      y.numInputDer = 0 ;  % non-differentiable
    end

    function y = sort(obj, varargin)
      y = Layer(@sort, obj, varargin{:}) ;
    end
    
    % overloaded indexing
    function varargout = subsref(a, s)
      if strcmp(s(1).type, '()')
        varargout{1} = Layer(@slice_wrapper, a, s.subs{:}) ;
      else
        [varargout{1:nargout}] = builtin('subsref', a, s) ;
      end
    end
    function idx = end(obj, dim, ndim)
      error('Not supported, use SIZE(X,DIM) or a constant size instead.') ;
    end
    
    function y = bsxfun(varargin)
      error(['BSXFUN is already called implicitly for all binary operators. ' ...
        'Use the corresponding math operator instead of BSXFUN.']) ;
    end
  end
  
  methods(Access = protected)
    function other = copyElement(obj)
      other = copyElement@matlab.mixin.Copyable(obj) ;
      other.id = Layer.uniqueId() ;
    end
  end
  
  methods (Access = {?Net, ?Layer})
    rootLayer = optimizeGraph(rootLayer)
  end
  
  methods (Access = {?Net, ?Layer})
    function mergeRedundantInputs(obj)
     objs = obj.find() ;  % list all layers in forward order
      lookup = struct() ;  % lookup table of input name to respective object
      for k = 1:numel(objs)
        objs{k}.enableCycleChecks = false ;  % faster
        in = objs{k}.inputs ;
        for i = 1:numel(in)
          if isa(in{i}, 'Input') && ~isempty(in{i}.name)
            if ~isfield(lookup, in{i}.name)  % add this Input to lookup table
              lookup.(in{i}.name) = in{i} ;
            else  % an Input with that name exists, reuse it
              original = lookup.(in{i}.name) ;
              original.gpu = original.gpu || in{i}.gpu ;  % merge GPU state
              objs{k}.inputs{i} = original ;
            end
          end
        end
        objs{k}.enableCycleChecks = true ;
      end
    end
    
    function cycleCheckRecursive(obj, root, visited)
      if eq(obj, root, 'sameInstance')
        error('MatConvNet:CycleCheckFailed', 'Input assignment creates a cycle in the network.') ;
      end
      
      % recurse on inputs
      idx = obj.getNextRecursion(visited) ;
      for i = idx
        obj.inputs{i}.cycleCheckRecursive(root, visited) ;
      end
      visited(obj.id) = true ;
    end
    
    function idx = getNextRecursion(obj, visited)
 
      valid = false(1, numel(obj.inputs)) ;
      for i = 1:numel(obj.inputs)
        if isa(obj.inputs{i}, 'Layer')
          valid(i) = ~visited.isKey(obj.inputs{i}.id) ;
        end
      end
      idx = find(valid) ;
    end
    
    function saveStack(obj)
      stack = dbstack('-completenames') ;
      
      % path 2 folders up from current file's directory (<autonn>/matlab)
      p = [fileparts(fileparts(stack(1).file)), filesep] ;
      
      % find a non-matching directory (outside AutoNN), or package
      % (starts with +)
      for i = 2:numel(stack)
        file = stack(i).file ;
        if ~strncmp(file, p, numel(p)) || strncmp(file, [p, '+'], numel(p) + 1)
          obj.source = stack(i:end) ;
          return
        end
      end
      obj.source = struct('file',{}, 'name',{}, 'line',{}) ;
    end
  end
  
  methods (Static)
    function setDiagnostics(obj, value)
      if iscell(obj)  % applies recursively to nested cell arrays
        for i = 1:numel(obj)
          Layer.setDiagnostics(obj{i}, value) ;
        end
      else
        obj.diagnostics = value ;
      end
    end
  end
  
  methods (Static, Access = protected)
    function id = uniqueId()
      persistent nextId
      if isempty(nextId)
        nextId = uint32(1) ;
      end
      id = nextId ;
      nextId = nextId + 1 ;
    end
    
    function visited = initializeRecursion()
      % See getNextRecursion
      visited = containers.Map('KeyType','uint32', 'ValueType','any') ;
    end
  end
end