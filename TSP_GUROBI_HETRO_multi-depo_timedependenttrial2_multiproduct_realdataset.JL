using Random, JuMP, Gurobi, LinearAlgebra, Plots

# ---------------------------
# Data Setup for Multi-Period Multi-Depot VRPTW
# ---------------------------
Random.seed!(1234567)
n_customers = 10  # Number of customers
n_periods = 1     # Number of periods
time_limit_per_period = 240.0  # Total time limit for each route per period
n_products = 2   # Number of products

# Generate random customer data
x_coords = rand(0.0:0.1:100.0, n_customers)
y_coords = rand(0.0:0.1:100.0, n_customers)
service_times_customers = rand(1:5, n_customers)
#time_windows_customers = [(rand(0:1:10), rand(30:60)) for _ in 1:n_customers]
time_windows_customers = [(0,240) for _ in 1:n_customers]
#time_windows_customers = [(rand(0), rand(60)) for _ in 1:n_customers]
#time_windows_customers = [0, 60 for _ in 1:n_customers]
# For customers (adjusting for depots already in the data)
demands_customers = rand(1:5, n_customers, n_products)

# Function to generate availability matrix (each customer available is available in all periods)
# function generate_availability!(availability)
#     availability .= 1  # Set all elements in the matrix to 1
# end
function generate_availability!(availability)
    n_customers, n_periods = size(availability)
    for i in 1:n_customers
        availability[i, :] .= 0
        j = rand(1:n_periods)
        availability[i, j] = 1
    end
end

# Initialize and generate availability matrix
availability = zeros(Int, n_customers, n_periods)
generate_availability!(availability)

# Define multiple depots
n_depots = 2
depot_coords = [(0.0, 0.0),(100,100)]
#depot_coords = [(0.0, 0.0), (100.0, 100.0)]
x_coords = vcat([c[1] for c in depot_coords], x_coords)
y_coords = vcat([c[2] for c in depot_coords], y_coords)
service_times = vcat([0.0, 0.0], service_times_customers)
# Combine with depot demands (assumed zero):
demands = vcat(zeros(2, n_products), demands_customers)
time_windows = vcat([(0.0, 1e5), (0.0, 1e5)], time_windows_customers)
n_nodes = length(x_coords)

# Vehicle parameters
fixed_cost = [120.0, 100.0]         # Fixed cost per vehicle per period
variable_cost = [1.5, 1.2]          # Variable cost per unit distance per vehicle
#vehicle_capacities = [20, 24]
# Each row: vehicle; columns: capacity for product 1 and product 2
vehicle_capacities = [25 26; 23 24]
num_vehicles  = size(vehicle_capacities, 1)
vehicle_depots = [2, 1]

println("Availability of customers in each period:")
for i in 1:n_customers
    println("Customer $i: ", availability[i, :])
end

println("Customer Data:")
for i in 1:n_nodes
    println("Node $i: (x=$(x_coords[i]), y=$(y_coords[i])), Service Time=$(service_times[i]), Demand=$(demands[i]), Time Window=$(time_windows[i])")
end

# ---------------------------
# Time and Distance Matrix Functions
# ---------------------------
# Compute the base Euclidean time matrix (travel time equals Euclidean distance)
function calculate_time_matrix(x_coords, y_coords)
    n = length(x_coords)
    time_matrix = zeros(n, n)
    for i in 1:n, j in 1:n
        time_matrix[i, j] = sqrt((x_coords[i] - x_coords[j])^2 + (y_coords[i] - y_coords[j])^2)
    end
    return time_matrix
end

# Compute distance matrix (same as time in this case)
function calculate_distance_matrix(x_coords, y_coords)
    n = length(x_coords)
    dist = zeros(n, n)
    for i in 1:n, j in 1:n
        dist[i, j] = sqrt((x_coords[i] - x_coords[j])^2 + (y_coords[i] - y_coords[j])^2)
    end
    return dist
end

# For visualization or reference
base_time_matrix = calculate_time_matrix(x_coords, y_coords)
println("Base Time Matrix:")
println(base_time_matrix)

#dist_matrix = calculate_distance_matrix(x_coords, y_coords)
#time_matrix = calculate_time_matrix(x_coords, y_coords)

dist_matrix =[0. 1910.  362.  299.  486.  451.  241.  522.  610  632.  732.  228.;
            1859.    0. 1508. 1983. 1375. 1409. 1788. 1694. 1477. 1504. 1530. 1691;
            360. 1556.    0.  477.  176.  146.  187.  421.  360. 396.  483.  186.;
            267. 1907.  396.    0.  483.  533.  390.  438.  594. 616.  716.  248.;
            518. 1424  161.  608.    0.  161.  359.  373.  200. 236.  322.  316.;
            466. 1458.  127.  596.  161.    0.  299.  540.  361. 397.  483.  305.;
            238. 1697.  171.  434.  327.  257.    0.  442.  516. 537.  638.  192.;
            543. 1755.  439.  472.  406.  577.  459.    0.  274. 274.  397.  342.;
            616. 1511.  369.  634.  199.  360.  520.  245.    0. 36.  126.  414.;
            635. 1527.  414.  653.  243.  404.  539.  264.   44.  0.  125.  433.;
            689. 1547.  463.  707.  314.  475.  593.  318.  115. 92.    0.  487.;
            228. 1705.  194.  331.  281.  331.  197.  318.  405. 427.  528.    0.]


# Print the distance matrix for reference

n_nodes = length(x_coords)
n_vehicles = size(vehicle_capacities, 1)
n_depots = maximum(vehicle_depots)
depots = 1:n_depots
customers = setdiff(1:n_nodes, depots)
dist = dist_matrix/60 # Convert distance matrix to minutes calculate_distance_matrix(x_coords, y_coords)
M = 1e5

# Create a unique time matrix for each vehicle and period.
# timecalc is a 4D array: (start node, end node, vehicle, period)
timecalc = zeros(n_nodes, n_nodes, n_vehicles, n_periods)
base_tm = calculate_time_matrix(x_coords, y_coords)
base_tm = dist_matrix/60 # calculate_time_matrix(x_coords, y_coords)
for v in 1:n_vehicles, p in 1:n_periods
    # Adjust the base time matrix by a random factor for each vehicle-period (e.g., between 0.95 and 1.05)
    if p == 2
        factor = 1.2   # example: increase by 10% in period 2
    elseif p == 3
        factor = 1.0   # example: decrease by 10% in period 3
    else
        factor = 1.0   # default factor for all other periods
    end
        #factor = 1.0 + 0.1 * (rand() - 0.5)
    timecalc[:, :, v, p] = base_tm .* factor
end

# ---------------------------
# Solver for Multi-Period Multi-Depot VRPTW
# ---------------------------
function solve_multi_period_vrptw(x_coords, y_coords, service_times, time_windows, demands, vehicle_capacities, vehicle_depots, availability)
    
    # n_nodes = length(x_coords)
    # n_vehicles = length(vehicle_capacities)
    # n_depots = maximum(vehicle_depots)
    # depots = 1:n_depots
    # customers = setdiff(1:n_nodes, depots)
    # dist = calculate_distance_matrix(x_coords, y_coords)
    # M = 1e5

    # # Create a unique time matrix for each vehicle and period.
    # # timecalc is a 4D array: (start node, end node, vehicle, period)
    # timecalc = zeros(n_nodes, n_nodes, n_vehicles, n_periods)
    # base_tm = calculate_time_matrix(x_coords, y_coords)
    # for v in 1:n_vehicles, p in 1:n_periods
    #     # Adjust the base time matrix by a random factor for each vehicle-period (e.g., between 0.95 and 1.05)
    #     factor = 1.0 + 0.1 * (rand() - 0.5)
    #     timecalc[:, :, v, p] = base_tm .* factor
    # end


    model = Model(Gurobi.Optimizer)
    M=1e5 # Large constant

    # Decision variables:
    @variable(model, x[1:n_nodes, 1:n_nodes, 1:n_vehicles, 1:n_periods], Bin)
    @variable(model, T[1:n_nodes, 1:n_vehicles, 1:n_periods] >= 0)
    # Additional variable to indicate if vehicle v is used in period p
    @variable(model, y[1:n_vehicles, 1:n_periods], Bin)
    # Additional variable to indicate and track the capacity of each vehicle for each product type 
    @variable(model, q[1:n_nodes, 1:n_vehicles, 1:n_periods, 1:n_products] >= 0)
    # Add binary variables for each arc (i,j,v,p) indicating the time interval
    @variable(model, z1[1:n_nodes, 1:n_nodes, 1:n_vehicles, 1:n_periods], Bin)
    @variable(model, z2[1:n_nodes, 1:n_nodes, 1:n_vehicles, 1:n_periods], Bin)
    @variable(model, z3[1:n_nodes, 1:n_nodes, 1:n_vehicles, 1:n_periods], Bin)



    # (1) No self-loops
    for v in 1:n_vehicles, p in 1:n_periods, i in 1:n_nodes
        @constraint(model, x[i, i, v, p] == 0)
    end

    # (2) Each customer must be visited exactly once across all periods
    for i in customers
        @constraint(model, sum(x[j, i, v, p] for j in 1:n_nodes for v in 1:n_vehicles for p in 1:n_periods if j != i) == 1)
    end

    # (3) Respect customer availability
    # Note: customers indexing is offset by n_depots (depots come first)
    for p in 1:n_periods, i in customers
        for j in 1:n_nodes, v in 1:n_vehicles
            @constraint(model, x[j, i, v, p] <= availability[i - n_depots, p])
        end
    end

    # (4) Depot departure and return constraints
    for v in 1:n_vehicles, p in 1:n_periods
        depot = vehicle_depots[v]
        @constraint(model, sum(x[depot, j, v, p] for j in 1:n_nodes if j != depot) == y[v, p])
        @constraint(model, sum(x[i, depot, v, p] for i in 1:n_nodes if i != depot) == y[v, p])
    end

    # (5) Prevent vehicles from visiting other depots
    for v in 1:n_vehicles, p in 1:n_periods
        for d in depots
            if d != vehicle_depots[v]
                for j in 1:n_nodes
                    @constraint(model, x[d, j, v, p] == 0)
                    @constraint(model, x[j, d, v, p] == 0)
                end
            end
        end
    end

    # (6) Flow conservation for each customer
    for v in 1:n_vehicles, p in 1:n_periods, i in customers
        @constraint(model, sum(x[i, j, v, p] for j in 1:n_nodes if i != j) ==
                              sum(x[j, i, v, p] for j in 1:n_nodes if i != j))
    end

    # (7) Time window constraints
    for v in 1:n_vehicles, p in 1:n_periods, i in 1:n_nodes, j in customers
        @constraint(model, T[i, v, p] >= time_windows[i][1] - M * (1 - x[j, i, v, p]))
        @constraint(model, T[i, v, p] <= time_windows[i][2] + M * (1 - x[j, i, v, p]))
    end

    # (8) Time propagation constraints using the unique time matrix:
    for v in 1:n_vehicles, p in 1:n_periods, i in 1:n_nodes, j in customers
        if i != j
            @constraint(model, T[j, v, p] >= T[i, v, p] + service_times[i] + timecalc[i, j, v, p] - M * (1 - x[i, j, v, p]))
        end
    end
    # (8.1) Time propagation constraints with dynamic scaling
    for v in 1:n_vehicles, p in 1:n_periods, i in 1:n_nodes, j in customers
        if i != j
            base_time = timecalc[i,j,v,p]
            @constraint(model, T[j,v,p] >= T[i,v,p] + service_times[i] + 
                1.1*base_time*z1[i,j,v,p] + 
                1.2*base_time*z2[i,j,v,p] + 
                1.0*base_time*z3[i,j,v,p] - 
                M*(1 - x[i,j,v,p]))
        end
    end

    # (9) Capacity constraints
    for v in 1:n_vehicles, p in 1:n_periods
        @constraint(model, sum(demands[j] * sum(x[i, j, v, p] for i in 1:n_nodes if i != j) for j in customers) <= vehicle_capacities[v])
    end

    # (10) Route duration constraint (using depot departure time)
    for v in 1:n_vehicles, p in 1:n_periods
        depot = vehicle_depots[v]
        @constraint(model, T[depot, v, p] <= time_limit_per_period + M*(1 - y[v, p]))
    end
    #(11) Multi-product capacity constraints
    for v in 1:n_vehicles, p in 1:n_periods, i in 1:n_nodes, j in 1:n_nodes, k in 1:n_products
        if i != j
            @constraint(model,
                q[j,v,p,k] >= q[i,v,p,k] - demands[j,k] - M * (1 - x[i,j,v,p])
            )
        end
    end
    # Initialize depot at load according to vehicle capacity
    for v in 1:n_vehicles, p in 1:n_periods, k in 1:n_products
        depot = vehicle_depots[v]
        # Here we assume the vehicle departs fully loaded (or you can model a pickup at the depot)
        @constraint(model, q[depot,v,p,k] == vehicle_capacities[v,k])
    end
    # Ensure that every load does not exceed the compartment capacity
    for v in 1:n_vehicles, p in 1:n_periods, i in 1:n_nodes, k in 1:n_products
        @constraint(model, q[i,v,p,k] <= vehicle_capacities[v,k])
    end
    # (12) Ensure that  only one interval is active per used arc
    for i in 1:n_nodes, j in 1:n_nodes, v in 1:n_vehicles, p in 1:n_periods
        @constraint(model, z1[i,j,v,p] + z2[i,j,v,p] + z3[i,j,v,p] == x[i,j,v,p])
    end

    # Define departure time (T[i,v,p] + service_times[i])
    M = 1e5  # Large constant
    for i in 1:n_nodes, j in 1:n_nodes, v in 1:n_vehicles, p in 1:n_periods
        # If z1 is active, departure time <= 60
        @constraint(model, T[i,v,p] + service_times[i] <= 60 + M*(1 - z1[i,j,v,p]))
        # If z2 is active, 60 < departure time <= 120
        @constraint(model, T[i,v,p] + service_times[i] >= 60.01 - M*(1 - z2[i,j,v,p]))
        @constraint(model, T[i,v,p] + service_times[i] <= 120 + M*(1 - z2[i,j,v,p]))
        # If z3 is active, departure time > 120
        @constraint(model, T[i,v,p] + service_times[i] >= 120.01 - M*(1 - z3[i,j,v,p]))
    end
    # Objective: Minimize total cost (fixed cost + variable cost based on the unique time matrix)
    @objective(model, Min,
        sum(fixed_cost[v] * y[v, p] for v in 1:n_vehicles, p in 1:n_periods) +
        sum(variable_cost[v] * timecalc[i, j, v, p] * x[i, j, v, p] for i in 1:n_nodes, j in 1:n_nodes, v in 1:n_vehicles, p in 1:n_periods)
    )

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        println("Optimal solution found with total cost: ", objective_value(model))
        # Extract routes for each vehicle and period.
        routes = Dict{Int, Dict{Int, Vector{Int}}}()
        for p in 1:n_periods
            routes[p] = Dict{Int, Vector{Int}}()
            for v in 1:n_vehicles
                depot = vehicle_depots[v]
                current = depot
                route = [current]
                while true
                    next_found = false
                    for j in 1:n_nodes
                        if current != j && value(x[current, j, v, p]) > 0.5
                            push!(route, j)
                            current = j
                            next_found = true
                            break
                        end
                    end
                    if !next_found || current == depot
                        break
                    end
                end
                routes[p][v] = route
            end
        end

        # Print routes for each period.
        for p in 1:n_periods
            println("\nPeriod $p:")
            for v in 1:n_vehicles
                println("  Vehicle $v route: ", join(routes[p][v], " , "))
            end
        end
        return routes
    else
        println("No optimal solution found.")
        return nothing
    end
end
t = @elapsed begin
    # Your code block
    #sleep(2)

# Solve the multi-period VRPTW
    routes = solve_multi_period_vrptw(x_coords, y_coords, service_times, time_windows, demands, vehicle_capacities, vehicle_depots, availability)

# ---------------------------
# Plotting Function for Multi-Period Routes
# ---------------------------
function plot_multi_period_vrp(x_coords, y_coords, routes, depot_coords, n_periods; file_name="multi_period_gurobi_vrptw_multiproduct_solution.png")
    colors = [:red, :blue, :green, :orange, :purple, :cyan, :magenta, :black, :pink, :brown]
    line_styles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
    
    # Create the initial plot with the legend placed outside.
    plt = scatter(x_coords, y_coords, marker=:circle, label="Customers", legend=:outertopright)
    
    # Mark depots.
    for (i, depot) in enumerate(depot_coords)
         scatter!(plt, [depot[1]], [depot[2]], marker=:star5, label="Depot $i", markersize=10, color=colors[i])
    end

    # Plot each vehicle's route for every period.
    for p in 1:n_periods
        for (v, route) in routes[p]
            if length(route) > 1
                # Close the route by returning to the depot.
                cyclic_route = vcat(route, route[1])
                x_route = [x_coords[node] for node in cyclic_route]
                y_route = [y_coords[node] for node in cyclic_route]
                plot!(plt, x_route, y_route, label="P$p, Veh $v", lw=1.5, color=colors[v],
                      linestyle=line_styles[mod1(p, length(line_styles))], title="Multi-Period Gurobi VRPTW multiproduct Solution")
            end
        end
    end

    savefig(plt, file_name)
    println("Multi-Period VRPTW multiproduct solution plot saved as $file_name")
end

if routes !== nothing
    plot_multi_period_vrp(x_coords, y_coords, routes, depot_coords, n_periods)
end
end
println("Elapsed time: $t seconds")