## Init
using Pkg
Pkg.instantiate()
using Printf
include("asif.jl")
include("examples.jl")
include("observer.jl")
using Plots, LinearAlgebra
using ControlSystems,ControlSystemIdentification
using DelimitedFiles
## Setup problem and attack policy 
m,l,gr = 2.0,1.0,10.0

f,g,_ = invpend_model()
#x0 = [-0.1;0.3];
x0 = zeros(2); 
Ts, T = 0.01, 100; 
N = Int(T/Ts)
X,U,Xh= zeros(2,N), zeros(1,N), zeros(2,N);
Y,Yh= zeros(1,N), zeros(1,N);

A,B  = [0 1.0; gr/l 0], [0; 1/(l*m)]
Q = 1e-5*diagm([1.0,1])

C = [1.0 0];
R = 1*diagm([1e-6]);

K = stationary_kalman(I(2)+Ts*A,C,R,Q)

a,b = 0.25,0.5
Ph = [1/(a^2) 0.5/(a*b);0.5/(a*b) 1/(b^2)]
h(x) =1-x'*Ph*x
∇h(x) =  -2*x'*Ph

λs,λvs = eigen(Ph)
1 ./(λs)
sf_angle = 180/pi*atan(λvs[1,2],λvs[1,1])

function desired_control(x,t;m=2,l=1,g=10,ydes=0)
    return m*l^2*(-(g/l)*sin(x[1])-dot([1.5; 1.5],x))+20*ydes;
end

## Simulate closed loop system 
x = x0;
xhat = x0; 
ydes = 0;
for i = 1:N 
    y = C*x + sqrt.(R)*randn(size(C,1)) # Get measurement
    Y[:,i],Yh[:,i] = y,C*xhat

    # False data injection 
    yadv = y; # TODO add attack

    if(mod(i,100)==1)
        ydes = randn();
    end
    udes = desired_control(xhat,Ts*i;ydes);
    u = asif_invpend(udes,h,∇h,f,g,xhat) # Compute "safe" control

    # Log information
    X[:,i],U[1,i], Xh[:,i] = x,u,xhat

    # Time update 
    x += Ts*(f(x)+g(x)*u)
    xhat += Ts*(f(xhat)+g(xhat)*u)+K*(y-C*xhat) # Step system
end

Xs = [X[:,i] for i in 1:N]

## Visualize training trajectory
plot(X[1,:],X[2,:])
#plot(Xh[1,:],Xh[2,:])
## Setup data set
est_ids = 1:Int(0.25*N)
val_ids = Int(0.25*N)+2:N
Dest = iddata(Yh[:,est_ids],[U;Y][:,est_ids],Ts)
Dval = iddata(Yh[:,val_ids],[U;Y][:,val_ids],Ts)
Dtot = iddata(Yh,[U;Y],Ts)
id_model = subspaceid(Dest, 2;stable=true, zeroD=true)
simplot(id_model,Dval)
simresult = lsim(id_model.sys,Dtot)
## Computing safe set
using JuMP
using SCS

zs = [simresult.x[:,i] for i in 1:size(simresult.x,2)]

sf_model = Model(SCS.Optimizer)
# We need to use a tighter tolerance for this example, otherwise the bounding
# ellipse won't actually be bounding...
set_attribute(sf_model, "eps_rel", 1e-6)
set_silent(sf_model)
N,n = length(zs),length(zs[1]);
@variable(sf_model, P[1:n, 1:n], PSD)
@variable(sf_model, v[1:n])

for z in zs
    @constraint(sf_model, [1;P*z-v] in MOI.SecondOrderCone(n+1))
end

@variable(sf_model, log_det_P)
@constraint(sf_model, [log_det_P; 1; vec(P)] in MOI.LogDetConeSquare(n))
@objective(sf_model, Max, log_det_P)

optimize!(sf_model)
solution_summary(sf_model)


PP = value.(P)
vv = value.(v)


scatter([z[1] for z in zs], [z[2] for z in zs])

latent_ellipse =  [tuple(PP \ [cos(θ) + vv[1], sin(θ) + vv[2]]...) for θ in 0:0.05:(2pi+0.05)];

Plots.plot!(
            latent_ellipse,
            c = :crimson,
            label = nothing,
           )


## Save offlie data
isdir("result") || mkdir("result");
open("result/iddata.dat"; write=true) do f
    ds = 4 
    write(f, "t y yh u \n")
    data = [collect(0:Ts:T-Ts) Y[:] Yh[:] U[:]]
    data = data[1:ds:end,:]
    writedlm(f, data)
end
open("result/latenttraj.dat"; write=true) do f
    ds = 4
    write(f, "t x y \n")
    data = [collect(0:Ts:T-Ts) [z[1] for z in zs] [z[2] for z in zs]]
    data = data[1:ds:end,:]
    writedlm(f, data)
end
open("result/latensafeset.dat"; write=true) do f
    ds = 1;
    write(f, "x y \n")
    data = [first.(latent_ellipse) last.(latent_ellipse)];
    data = data[1:ds:end,:]
    writedlm(f, [first.(latent_ellipse) last.(latent_ellipse)])
end
## Simulate closed loop system  - Attack
x0 = [-0.2;0.5];
x,xhat = x0,x0;
z = zeros(2)
δ=1e-3
Ts, T = 0.01, 3; 
N = Int(T/Ts)
Xa,Ua,Xha= zeros(2,N), zeros(1,N), zeros(2,N);
Ya,Yha= zeros(1,N), zeros(1,N);
Za,Rsa = zeros(2,N),zeros(1,N)

for i = 1:N 
    y = C*x + sqrt.(R)*randn(size(C,1)) # Get measurement
    Y[:,i],Yh[:,i] = y,C*xhat
    # 
    Δz = id_model.B[:,2]'*(2*(vv-PP*z))
    if(i*Ts > 1)
        yadv = id_model.C*z+δ*normalize([Δz])
    else
        yadv = y; 
    end

    # Linear-feedbfack, stabilizing
    udes = m*l^2*(-(gr/l)*sin(xhat[1])-dot([1.5; 1.5],xhat))
    #udes = 0;

    # Compute "safe" control
    u = asif_invpend(udes,h,∇h,f,g,xhat) 

    # Log information
    Xa[:,i],Ua[1,i], Xha[:,i],Za[:,i], Rsa[:,i] = x,u,xhat,z,yadv-C*xhat

    # Time update 
    z = id_model.A*z + id_model.B*[u;yadv]+id_model.K*(C*xhat-id_model.C*z)
    x += Ts*(f(x)+g(x)*u)
    xhat += Ts*(f(xhat)+g(xhat)*u)+K*(yadv-C*xhat) # Step system
end

## Visualize trajectory
plot(Xa[1,:],Xa[2,:])
plot!(Xha[1,:],Xha[2,:])
Ch = cholesky(Ph).U
sf_ellipse =  [tuple((Ch\ [cos(θ), sin(θ)])...) for θ in 0:0.05:(2pi+0.05)];
Plots.plot!(sf_ellipse,c = :green)
## Save data
isdir("result") || mkdir("result");
open("result/attack.dat"; write=true) do f
    ds = 1 
    write(f, "t x1 x2 xh1 xh2 r z1 z2  \n")
    data = [collect(0:Ts:T-Ts) Xa[1,:] Xa[2,:] Xha[1,:] Xha[2,:] Rsa[:] Za[1,:] Za[2,:]]
    data = data[1:ds:end,:]
    writedlm(f, data)
end
