function Theta = unified_theta(LM,r)
l = LM(1); m = LM(2);
Theta = acos(((l+m)*sqrt(r.^2*(1-l^2)+(l+m)^2)-l*r.^2)./...
          (r.^2 + (l+m)^2) );
end