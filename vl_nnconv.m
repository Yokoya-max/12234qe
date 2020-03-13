function layer = vl_nnconv(varargin)
  opts = struct('size', [], 'weightScale', 'xavier', 'hasBias', true, ...
    'learningRate', [1 1], 'weightDecay', [1 0], 'pad', 0, 'transpose', false) ;
  [opts, posArgs, convOpts] = vl_argparsepos(opts, varargin, 'flags', ...
    {'cuDNN', 'noCuDNN', 'verbose', 'noDerData', 'noDerFilters', 'noDerBiases'}) ;
  
  if ~isempty(opts.size)
    % a size was specified, create Params
    assert(numel(posArgs) == 1, ...
      'Must specify only one input Layer when using the ''size'' option.') ;
    
    if opts.hasBias
      % create bias as the 3rd input
      if opts.transpose  % vl_nnconvt, use 3rd dimension of filters
        biasSize = opts.size(3) ;
      else  % vl_nnconv, use 4th dimension of filters
        biasSize = opts.size(4) ;
      end
      posArgs{3} = Param('value', zeros(biasSize, 1, 'single'), ...
                     'learningRate', opts.learningRate(max(1,end)), ...
                     'weightDecay', opts.weightDecay(max(1,end))) ;
    else
      posArgs{3} = [] ;
    end

    if isequal(opts.weightScale, 'xavier')
      scale = sqrt(2 / prod(opts.size(1:3))) ;
    else
      scale = opts.weightScale ;
    end

    posArgs{2} = Param('value', randn(opts.size, 'single') * scale, ...
                    'learningRate', opts.learningRate(1), ...
                    'weightDecay', opts.weightDecay(1)) ;
  else
    assert(numel(posArgs) == 3, ...
      'Must specify all 3 inputs, or the ''size'' option.') ;
  end
  
  if isequal(opts.pad, 'same')
    assert(isa(posArgs{2}, 'Param'), ...
      'Can only use ''same'' padding when the convolution filter is of class Param.') ;
    sz = size(posArgs{2}.value) ;
    
    pad = (sz(1:2) - 1) / 2 ;  % will be fractional for even filter sizes
    opts.pad = [floor(pad(1)), ceil(pad(1)), floor(pad(2)), ceil(pad(2))] ;
  end
  
  if any(opts.pad ~= 0)
    convOpts(end+1:end+2) = {'pad', opts.pad} ;
  end
  
  if opts.transpose
    func = @vl_nnconvt ;
  else
    func = @vl_nnconv ;
  end
  layer = Layer(func, posArgs{:}, convOpts{:}) ;
end