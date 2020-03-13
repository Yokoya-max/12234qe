function tnet = HSINNtidy(net)

tnet = struct('layers', {{}}, 'meta', struct()) ;

% copy meta information in net.meta subfield
if isfield(net, 'meta')
  tnet.meta = net.meta ;
end

if isfield(net, 'classes')
  tnet.meta.classes = net.classes ;
end

if isfield(net, 'normalization')
  tnet.meta.normalization = net.normalization ;
end

if  isfield(tnet, 'meta') && isfield(tnet.meta, 'normalization') && ...
   ~isfield(tnet.meta.normalization, 'cropSize') && ...
    isfield(tnet.meta.normalization, 'border') && ...
    isfield(tnet.meta.normalization, 'imageSize')
  insz = tnet.meta.normalization.imageSize(1:2);
  bigimSz = insz + tnet.meta.normalization.border;
  tnet.meta.normalization.cropSize = insz ./ bigimSz;
end

for l = 1:numel(net.layers)
  defaults = {'name', sprintf('layer%d', l), 'precious', false};
  layer = net.layers{l} ;
  if strcmp(layer.type, 'custom')
    tnet.layers{l} = layer ;
    continue;
  end

  switch layer.type
    case {'conv', 'convt', 'bnorm'}
      if ~isfield(layer, 'weights')
        layer.weights = {...
          layer.filters, ...
          layer.biases} ;
        layer = rmfield(layer, 'filters') ;
        layer = rmfield(layer, 'biases') ;
      end
  end
  if ~isfield(layer, 'weights')
    layer.weights = {} ;
  end

  if strcmp(layer.type, 'bnorm')
    if numel(layer.weights) < 3
      layer.weights{3} = ....
        zeros(numel(layer.weights{1}),2,'single') ;
    end
  end

  switch layer.type
    case 'conv'
      defaults = [ defaults {...
        'pad', 0, ...
        'stride', 1, ...
        'dilate', 1, ...
        'opts', {}}] ;

    case 'pool'
      defaults = [ defaults {...
        'pad', 0, ...
        'stride', 1, ...
        'opts', {}}] ;

    case 'convt'
      defaults = [ defaults {...
        'crop', 0, ...
        'upsample', 1, ...
        'numGroups', 1, ...
        'opts', {}}] ;

    case {'pool'}
      defaults = [ defaults {...
        'method', 'max', ...
        'pad', 0, ...
        'stride', 1, ...
        'opts', {}}] ;

    case 'relu'
      defaults = [ defaults {...
        'leak', 0}] ;

    case 'dropout'
      defaults = [ defaults {...
        'rate', 0.5}] ;

    case {'normalize', 'lrn'}
      defaults = [ defaults {...
        'param', [5 1 0.0001/5 0.75]}] ;

    case {'pdist'}
      defaults = [ defaults {...
        'noRoot', false, ...
        'aggregate', false, ...
        'p', 2, ...
        'epsilon', 1e-3, ...
        'instanceWeights', []} ];

    case {'bnorm'}
      defaults = [ defaults {...
        'epsilon', 1e-5 } ] ;
  end

  for i = 1:2:numel(defaults)
    if ~isfield(layer, defaults{i})
      layer.(defaults{i}) = defaults{i+1} ;
    end
  end

  % save back
  tnet.layers{l} = layer ;
end