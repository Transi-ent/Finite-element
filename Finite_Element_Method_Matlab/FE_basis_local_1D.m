function result=FE_basis_local_1D(x,vertices,basis_type,asis_index,derivative_order)

xnp1=vertices(2);
xn=
h=
if basis_type==1
    if basis_index==1
        if derivative_order==0
             result=(xnp1-x)/h;                                  %xnp1----xn_plus_1
        elseif derivative_order==1
            result=-1/h;
        elseif derivative_order>=2
            result=0;
        else
            warning='wrong input of derivative order!'
        end
    elseif basis_index==2
    else
        warning='wrong input of basis index!'
    
    
elseif basis_type==2

end