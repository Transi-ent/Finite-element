function A=assemble_matrix_1D_linear()

A=zeros(matrix_size(1),matrix_size(2));
%A=zeros(number_of_nodes,number_of_nodes);

for n=1:number_of_elements
    for alpha=1:Nlb                                             % number of local basis function
        for beta=1:Nlb                                           % number of local basis function
            
            int_value=Gauss_quatrature_1D_linear_trial_test(?);
            A(Tb(beta,n),Tb(alpha,n))=A(Tb(beta,n),Tb(alpha,n))+int_value;
        end
    end
end