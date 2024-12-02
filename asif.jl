using JuMP
using DAQP 
function asif_invpend(udes,h,∇h,f,g,x;ca=0.2)
    ca = 1e-2
    model = Model(DAQP.Optimizer)
    @variable(model, -3<=u<=3)
    @variable(model, ρ)
    @objective(model, Min, 0.5*((u-udes).^2)+1e3*ρ^2)
    Lfh,Lgh = ∇h(x)*f(x), ∇h(x)*g(x) 
    @constraint(model, c1, Lfh+Lgh*u + ρ ≥ -ca*h(x))
    optimize!(model);
    return value(u);
end
