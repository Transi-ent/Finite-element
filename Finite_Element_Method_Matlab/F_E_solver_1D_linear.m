function result=F_E_solver_1D_linear()

[P,T]=generate_mesh();

A=assemble_matrix_1D_linear();

b=assemble_vector_1D_linear();

[A,b]=treat_Dirichlet_boundary_1D();

result=A\b;