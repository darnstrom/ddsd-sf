using MatrixEquations 
function stationary_kalman(A,C,R,Q) 
    P,_ = ared(A,C',R,Q);
    K = P*C'*inv(C*P*C'+R) 
    return K 
end
function correct(x,y,C,K)
    r = y-C*x;
    x += K*r
    return x,r
end

function predict(x,u,A,B)
    return A*x+B*u 
end
