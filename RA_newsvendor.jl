using JuMP, GLPK
using Distributions, Random

c = 1 #buying cost
q_short = 2 #shortage cost
q_dest = 0 #destruction cost
prices = (c,q_short,q_dest)

S = 1000 #number of scenarios
μ = 50 
σ = 5
Random.seed!(42) #for reproducibility

law = Normal(μ,σ)
# Alternative laws
#law = Cauchy(μ,σ)
#law = Exponential(μ)
d = rand(law,S)  # Sampled data

const OPTIMIZER = GLPK.Optimizer

function generate_model(d,prices)
    (c,q_short,q_dest) = prices 
    m = Model(OPTIMIZER)
    @variables(m,begin
        x >=0 
        y_short[1:S] >= 0 #
        y_dest[1:S] >= 0
        cost[1:S]
        end) 

    @constraints(m,begin
        [s in 1:S],  y_short[s] >= d[s] - x
        [s in 1:S],  y_dest[s] >= x- d[s]
        [s in 1:S],  cost[s] == (c*x+ q_short*y_short[s] 
        + q_dest*y_dest[s]) 
    end)
    return(m)
end 

function mean_cost(d,prices,x)
    (c,q_short,q_dest) = prices 
    cost = c*x
    for s in 1:S
        cost += 1/S*(q_short * max(d[s] - x,0) + q_dest * max(x- d[s],0))
    end
    return cost
end

##### Risk neutral 

function solve_risk_neutral(d,prices)
    println("Risk Neutral model")
    m = generate_model(d,prices)
    @objective(m, Min, 1/S*sum(m[:cost][s] for s=1:S))
    optimize!(m) 
    println("optimal value: ", objective_value(m))
    println("optimal first control: ", value(m[:x]))
    println(" ")
end 

##### TVAR 
β = 0.9 
function solve_TVAR(d,prices;β=β)
    println("TVAR model")
    m = generate_model(d,prices)
    @variable(m,t)
    @variable(m,z[1:S]>=0) #epigraphic variables
    @constraint(m,z.>=(m[:cost].-t))
    @objective(m, Min, t + 1/(1-β)*1/S*sum(z[s] for s=1:S))
    optimize!(m) 
    println("optimal risk-adjusted value: ", objective_value(m))
     x= value(m[:x])
    println("optimal mean-cost: ", mean_cost(d,prices,x) )
    println("optimal first control: ", x)
    println(" ")
end 


solve_risk_neutral(d,prices)
solve_TVAR(d,prices,β=β)