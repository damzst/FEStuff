function out1 = myinterp2(arg1,arg2,arg3,arg4,arg5)

% Get output size
[nrows,ncols] = size(arg3);

% Compute interpolation parameters
s = 1 + (arg4-arg1(1))/(arg1(end)-arg1(1))*(ncols-1);
t = 1 + (arg5-arg2(1))/(arg2(end)-arg2(1))*(nrows-1);

% Check for out of range values of s and t and set to 1
sout = find((s<1)|(s>ncols));
if ~isempty(sout), s(sout) = 1; end
tout = find((t<1)|(t>nrows));
if ~isempty(tout), t(tout) = 1; end

% Matrix element indexing
ndx = floor(t)+floor(s-1)*nrows;

% Compute interpolation parameters:
% s(:) = (s - floor(s));
% t(:) = (t - floor(t));

% Compute intepolation parameters, check for boundary value.
if isempty(s), d = s; else d = find(s==ncols); end
s(:) = (s - floor(s));
if ~isempty(d), s(d) = s(d)+1; ndx(d) = ndx(d)-nrows; end

% Compute intepolation parameters, check for boundary value.
if isempty(t), d = t; else d = find(t==nrows); end
t(:) = (t - floor(t));
if ~isempty(d), t(d) = t(d)+1; ndx(d) = ndx(d)-1; end

% Now interpolate.
onemt = 1-t;

out1 = (arg3(ndx).*(onemt)+arg3(ndx+1).*t).*(1-s)+ ...
       (arg3(ndx+nrows).*(onemt)+arg3(ndx+(nrows+1)).*t).*s;
