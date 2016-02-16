function r = unified(LM,THETA)
l = LM(1); m = LM(2);
r = (l+m)*sin(THETA) ./ (l+cos(THETA));
end