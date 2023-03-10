using DifferentialEquations, ParameterizedFunctions, Plots
using Lux, DataDrivenDiffEq, ModelingToolkit, OrdinaryDiffEq, LinearAlgebra #, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationFlux, OptimizationOptimJL , DiffEqSensitivity
using Statistics, ComponentArrays, Random
using DiffEqFlux, OptimizationOptimJL

# Set up ODE data
begin
    LV = @ode_def begin
        dð = Î±*ð - Î²*ð*ð
        dð = Î²*ð*ð*Î»l  - Î´*ð - Î³*ð*ðº
        dðº = Î³*ð*ðº*Î»w - Î·*ðº

    end Î± Î² Î»l Î»w Î³ Î´ Î·

    u0 = [20, 10, 6]
    tspan = (0, 10)
    parameters = (1.7, .2, 0.11, 0.12, .1, .2, .1)
    datasize = 30
    tsteps = range(tspan[1], tspan[2], length = datasize)

    prob = ODEProblem(LV, u0, tspan, parameters)
    og_data = solve(prob, Tsit5(), saveat = tsteps)
    sol = Array(solve(prob, Tsit5(), saveat = tsteps))
    plot(og_data)
end


## Defining a NN for NDE
begin
    rng = Random.default_rng()
    Random.seed!(1234)

    dudtââ = Lux.Chain(
        x -> x.^3,
        Lux.Dense(3, 20, tanh),
        Lux.Dense(20,20,tanh),
        Lux.Dense(20, 3)
        )

    pââ, strââ = Lux.setup(rng, dudtââ)
    prob_neuralode = NeuralODE(dudtââ, tspan, Tsit5(), saveat = tsteps)

    function predict_neuralode(p)
        Array(prob_neuralode(u0, p, strââ)[1])
    end
end


# Define a loss function
begin
    function loss_neuralode(network_params)
        pred = predict_neuralode(network_params)
        loss = sum(abs2, sol .- pred)
        return loss, pred
    end

    loss_neuralode(p,hyper_p) = loss_neuralode(p)
end


# Training 1
begin
    function plot_solutions(sol, prediction)
        plt = plot(tsteps, sol', label = ["org ð" "org ð±" "org ðº "])
        scatter!(plt, tsteps, prediction',  label = ["pred ð" "pred ð±" "pred ðº"])
        return plt
    end

    callback = function (p, l, pred; doplot = false)
        println(l)
        # plot current prediction against data
        if doplot
          plt =  plot_solutions(sol, pred)
            return plt
        end
        return false
    end

    pinit = Lux.ComponentArray(pââ)

    callback(pinit, loss_neuralode(pinit)...; doplot=true)
end


# Training 2
begin
    adtype = Optimization.AutoZygote()

    optf = OptimizationFunction(loss_neuralode, adtype)

    optprob = Optimization.OptimizationProblem(optf, pinit)

    result_neuralode = Optimization.solve(optprob,
                                       ADAM(0.05),
                                       callback = callback,
                                       maxiters = 300)
    
                                       callback(result_neuralode.u, loss_neuralode(result_neuralode.u)...; doplot=true)
end

# Training 3
begin
    optprob2 = remake(optprob,u0 = result_neuralode.u)

    result_neuralode2 = Optimization.solve(optprob2,
                                        Optim.BFGS(initial_stepnorm=0.01),
                                        callback=callback,
                                        allow_f_increases = false)

    callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot=true)
end