function [P,T]=generate_mesh(left,right,h)

N=(right-left)/h;

P=zeros(1,N+1);
T=zeros(2,N);

for i=1:N+1
    
    P(i)=left+(i-1)*h;
    
end

for i=1:?
    
    T(:,i)=[? ?]';
    
end