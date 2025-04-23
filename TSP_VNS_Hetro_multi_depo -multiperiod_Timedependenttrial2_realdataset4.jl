using Random, LinearAlgebra, Plots


# ---------------------------
# Parameters and Data Generation
# ---------------------------
Random.seed!(1234567)
n_customers = 10  # Number of customers (excluding depots)
n_periods = 1 # Multi-period: 4 periods
time_limit_per_period = 240.0  # Total time limit for each route per period
n_products = 2  # Number of products

# Generate random customer data
x_coords = rand(0.0:0.1:100.0, n_customers)
y_coords = rand(0.0:0.1:100.0, n_customers)
service_times_customers = rand(1:5, n_customers)
#time_windows_customers = [(rand(0:10:100), rand(300:500)) for _ in 1:n_customers]
time_windows_customers = [(0,240) for _ in 1:n_customers]
# Assume multiple products with different demands for each customer
demands_customers = rand(1:5, n_customers, n_products)

# Function to ensure each customer is available in exactly one period.
# function generate_availability!(availability)
#     n_customers, n_periods = size(availability)
#     for i in 1:n_customers
#         availability[i, :] .= 0
#         j = rand(1:n_periods)
#         availability[i, j] = 1
#     end
# end
# function generate_availability!(availability)
#     n_customers, n_periods = size(availability)
#     @assert n_periods >= 2 "There must be at least two periods to have adjacent slots."
#     for i in 1:n_customers
#         availability[i, :] .= 0
#         # Choose a starting slot such that there's room for an adjacent slot
#         j = rand(1:n_periods-1)
#         availability[i, j] = 1
#         availability[i, j+1] = 1
#     end
# end
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
# Initialize and generate the availability matrix.
availability = zeros(Int, n_customers, n_periods)
generate_availability!(availability)

println("Availability of customers in each period:")    
for i in 1:n_customers
    println("Customer $i: ", availability[i, :])
end

# Define multiple depots
n_depots = 2
depot_coords = [(0.0, 0.0),(100.0, 100.0)]  # Example depot coordinates
#depot_coords = [(0.0, 0.0), (100.0, 100.0)]
x_coords = vcat([c[1] for c in depot_coords], x_coords)
y_coords = vcat([c[2] for c in depot_coords], y_coords)
service_times = vcat([0.0, 0.0], service_times_customers)
#demands = vcat([0.0, 0.0], demands_customers)
demands = vcat(zeros(2, n_products), demands_customers)
time_windows = vcat([(0.0, 1e5), (0.0, 1e5)], time_windows_customers)
n_nodes = length(x_coords)


# New: Vehicle cost parameters (adjust as needed)
# Vehicle parameters
fixed_cost = [100.0, 120.0]         # Fixed cost per vehicle per period
variable_cost = [1.2, 1.5]          # Variable cost per unit distance per vehicle
#vehicle_capacities = [20, 24]
# Each row: vehicle; columns: capacity for product 1 and product 2
vehicle_capacities = [23 24; 25 26]
num_vehicles  = size(vehicle_capacities, 1)
vehicle_depots = [1, 2]


# NEW: Compute Time Matrix (assume travel time equals Euclidean distance)
function calculate_time_matrix(x_coords, y_coords)
    n = length(x_coords)
    time_matrix = zeros(n, n)
    for i in 1:n, j in 1:n
        time_matrix[i, j] = sqrt((x_coords[i]-x_coords[j])^2 + (y_coords[i]-y_coords[j])^2)
    end
    return time_matrix
end
#time_matrix = calculate_time_matrix(x_coords, y_coords)
time_matrix =[0. 1910.  362.  299.  486.  451.  241.  522.  610  632.  732.  228.;
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
#time_matrix = calculate_time_matrix(x_coords, y_coords)

println("Time Matrix:")

println(time_matrix)


# Create a unique time matrix for each vehicle and period.
# timecalc is a 4D array: (start node, end node, vehicle, period)
# Assume: 
#   n_nodes is the number of nodes,
#   num_vehicles is the number of vehicles (e.g., length(vehicle_capacities)),
#   n_periods is the number of periods.
#base_tm = calculate_time_matrix(x_coords, y_coords)
base_tm = time_matrix /60 # Convert to minutes (if needed)
timecalc = zeros(n_nodes, n_nodes, num_vehicles, n_periods)
for v in 1:num_vehicles
    for p in 1:n_periods
        # Adjust the base time matrix by a random factor for each vehicle–period (e.g., between 0.95 and 1.05)
            # Assign specific factors for certain periods
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
end

println("Base Time Matrix:")
println(base_tm)
println("Unique Time Matrices created for each vehicle and period.")


# Print random customer data


println("Customer Data:")
for i in 1:n_nodes
    println("Node $i: (x=$(x_coords[i]), y=$(y_coords[i])), Service Time=$(service_times[i]), Demand=$(demands[i]), Time Window=$(time_windows[i])")
end

# ---------------------------
# Utility Functions
# ---------------------------
# Compute Euclidean distance matrix.
function calculate_distance_matrix(x_coords, y_coords)
    n = length(x_coords)
    dist = zeros(n, n)
    for i in 1:n, j in 1:n
        dist[i, j] = sqrt((x_coords[i]-x_coords[j])^2 + (y_coords[i]-y_coords[j])^2)
    end
    return dist
end

dist = calculate_distance_matrix(x_coords, y_coords)

# NEW: Compute Time Matrix (assume travel time equals Euclidean distance)
# function calculate_time_matrix(x_coords, y_coords)
#     n = length(x_coords)
#     time_mat = zeros(n, n)
#     for i in 1:n, j in 1:n
#         time_mat[i, j] = sqrt((x_coords[i]-x_coords[j])^2 + (y_coords[i]-y_coords[j])^2)
#     end
#     return time_mat
# end

#time_matrix = calculate_time_matrix(x_coords, y_coords)
time_matrix = time_matrix /60 # Convert to minutes (if needed)

# ---------------------------
# Helper Functions Adjusted for Multi-Product
# ---------------------------
# Calculate the cumulative demand on a route as a vector (ignoring depots)
function calculate_route_demand(route::Vector{Int}, demands, depots)
    total = zeros(size(demands, 2))
    for node in route
        if !(node in depots)
            total .+= demands[node, :]  # demands for each product
        end
    end
    return total
end

# Helper: compute total travel time for a route using the time matrix.
function route_time(route::Vector{Int}, time_matrix)
    total_time = 0.0
    for k in 1:(length(route)-1)
        total_time += time_matrix[route[k], route[k+1]]
    end
    return total_time
end

# Check feasibility for a route in a given period.
function is_route_feasible_period(route::Vector{Int}, time_matrix, service_times, time_windows, demands, capacity, depots, time_limit)
    current_time = 0.0
    total_demand = zeros(size(demands, 2))  # vector for each product
    #total_demand = 0
    for k in 1:(length(route)-1)
        i = route[k]
        j = route[k+1]
        # Calculate dynamic scaling factor based on current_time within the period
        scaling = if current_time <= 60.0
            1.10  # 10% increase (0-60 minutes)
        elseif current_time <= 120.0
            1.20  # 20% increase (60-120 minutes)
        else
            1.0  # reduce to original form 0% beyond 120 minutes (adjust if needed)
        end
        travel_time = time_matrix[i, j]*scaling
        arrival = current_time + travel_time
        tw_start, tw_end = time_windows[j]
        service_start = max(arrival, tw_start)
        if service_start > tw_end || (service_start + service_times[j] > time_limit)
            return false
        end
        current_time = service_start + service_times[j]
        if !(j in depots)
            total_demand += demands[j,:]
            # Check each product compartment; route is infeasible if any product exceeds capacity.
            if any(total_demand .> capacity)
                return false
            end
        end
    end
    return true
end

# Compute the total distance of a route (used in the cost calculation).
function route_distance(route::Vector{Int}, dist)
    total_distance = 0.0
    for k in 1:(length(route)-1)
        total_distance += dist[route[k], route[k+1]]
    end
    return total_distance
end

# Compute total cost of a multi-period solution.
# For each route, we add a fixed cost (if the vehicle is used) and the variable cost
# (multiplied by the route's travel distance).
# We assume the route index corresponds to the vehicle index (and use vehicle 1's cost if extra routes exist).
# For each route we will use a unique time matrix for that period and vehicle 
function solution_cost(solution::Dict{Int, Vector{Vector{Int}}}, timecalc, fixed_cost, variable_cost,num_vehicles)
    total = 0.0
    for (p, routes) in solution
        for (r_idx, route) in enumerate(routes)
            # If route only consists of depot start and depot end, skip cost.
            if length(route) == 2
                continue
            end
            # Get the appropriate time matrix slice for period p and vehicle r_idx.
            tm = get_vehicle_time_matrix(timecalc, p, r_idx, num_vehicles)
            vehicle_idx = r_idx <= num_vehicles ? r_idx : 1
            #route_dist = route_distance(route, dist)
            route_timevalue = route_time(route, tm)
            total += fixed_cost[vehicle_idx] + variable_cost[vehicle_idx] * route_timevalue
        end
    end
    return total
end


# # Helper: calculate total demand of a route (ignoring depots)
# function calculate_route_demand(route::Vector{Int}, demands, depots)
#     total = 0
#     for node in route
#         if !(node in depots)
#             total += demands[node]
#         end
#     end
#     return total
# end

# Helper: select the appropriate time matrix from timecalc given period and route (vehicle) index.
function get_vehicle_time_matrix(timecalc, period, route_index, num_vehicles)
    vehicle = route_index <= num_vehicles ? route_index : 1
    return timecalc[:, :, vehicle, period]
end
# ---------------------------
# Initial Solution Construction for Multi-Period
# ---------------------------
function initial_solution_multiperiod(timecalc, service_times, time_windows, demands, vehicle_capacities, vehicle_depots, depots, n_periods, availability, time_limit,num_vehicles)
    # Customers are nodes after the depots.
    customers = collect((maximum(depots)+1):n_nodes)
    
    # Assign each customer to one period in which they are available (randomly)
    period_assignment = Dict{Int, Int}()
    for cust in customers
        possible_periods = [p for p in 1:n_periods if availability[cust - maximum(depots), p] == 1]
        if isempty(possible_periods)
            error("Customer $cust is not available in any period!")
        end
        period_assignment[cust] = rand(possible_periods)
    end
    
    # Group customers by period.
    period_customers = Dict{Int, Vector{Int}}()
    for p in 1:n_periods
        period_customers[p] = [cust for cust in customers if period_assignment[cust] == p]
    end
    
    # Build a solution: for each period, create one route per vehicle.
    solution = Dict{Int, Vector{Vector{Int}}}()
    for p in 1:n_periods
        solution[p] = Vector{Vector{Int}}()
        unassigned = copy(period_customers[p])
        # For each vehicle, build a route using a greedy heuristic.
        for v in 1:num_vehicles
            depot = vehicle_depots[v]
            # Retrieve multi-product capacity vector for this vehicle:
            capacity = vehicle_capacities[v, :]
            route = [depot]  # route starts at the assigned depot
            current_time = 0.0
            current_load = zeros(n_products)
            changed = true
            while changed && !isempty(unassigned)
                changed = false
                best_cust = nothing
                best_increase = Inf
                best_new_time = 0.0
                last = route[end]
                # Use the time matrix for this vehicle and period
                tm = get_vehicle_time_matrix(timecalc, p, v, num_vehicles)
                for cust in unassigned
                    travel_time = tm[last, cust]
                    arrival = current_time + travel_time
                    new_load = current_load + demands[cust,:]
                    tw_start, tw_end = time_windows[cust]
                    service_start = max(arrival, tw_start)
                    if service_start > tw_end || (service_start + service_times[cust] > time_limit) || any(new_load .> capacity)
                        continue
                    end
                    # if new_load > capacity
                    #     continue
                    # end
                    cost_increase = tm[last, cust] + tm[cust, depot] - tm[last, depot]
                    if cost_increase < best_increase
                        best_increase = cost_increase
                        best_cust = cust
                        best_new_time = service_start + service_times[cust]
                    end
                end
                if best_cust !== nothing
                    push!(route, best_cust)
                    deleteat!(unassigned, findfirst(==(best_cust), unassigned))
                    current_time = best_new_time
                    current_load .+= demands[best_cust,:]
                    changed = true
                end
            end
            push!(route, depot)  # return to depot
            push!(solution[p], route)
        end
        # For any remaining customers in period p, create a singleton route (using depot of vehicle 1)
        for cust in unassigned
            #push!(solution[p], cust)
            depot = vehicle_depots[1]
            push!(solution[p], [depot, cust, depot])
        end
        # if !isempty(unassigned)
        #     error("Infeasible initial solution: Customers left unassigned in period $p")
        # end
    end
    
    return solution, period_assignment
end





# ---------------------------
# Neighborhood Operators for Multi-Period VNS
# ---------------------------
# 1. Intra-route 2-opt: Reverse a segment within a route.
function intra_route_2opt_multiperiod(solution::Dict{Int, Vector{Vector{Int}}}, timecalc, service_times, time_windows, demands, vehicle_capacities, depots, time_limit,num_vehicles)
    best_solution = deepcopy(solution)
    best_cost = solution_cost(solution, timecalc, fixed_cost, variable_cost,num_vehicles)
    improved = false
    for p in keys(solution)
        for r_idx in 1:length(solution[p])
            route = solution[p][r_idx]
            # Assume the vehicle capacity is given by the vehicle corresponding to index r_idx (if available)
            capacity = vehicle_capacities[min(r_idx, size(vehicle_capacities,1)), :]
            n_route = length(route)
            if n_route <= 3
                continue
            end
            # Get the appropriate time matrix slice for this route.
            tm = get_vehicle_time_matrix(timecalc, p, r_idx, num_vehicles)
            for i in 2:(n_route-2)
                for j in (i+1):(n_route-1)
                    new_route = copy(route)
                    new_route[i:j] = reverse(new_route[i:j])
                    if is_route_feasible_period(new_route, tm, service_times, time_windows, demands, capacity, depots, time_limit)
                        new_solution = deepcopy(solution)
                        new_solution[p][r_idx] = new_route
                        new_cost = solution_cost(new_solution, timecalc, fixed_cost, variable_cost,num_vehicles)
                        if new_cost < best_cost
                            best_solution = new_solution
                            best_cost = new_cost
                            improved = true
                        end
                    end
                end
            end
        end
    end
    return best_solution, improved
end

# 2. Inter-route relocate: Remove a customer from one route and insert into another (possibly across periods).
function inter_route_relocate_multiperiod(solution::Dict{Int, Vector{Vector{Int}}}, timecalc, service_times, time_windows, demands, vehicle_capacities, depots, time_limit, availability, n_periods,num_vehicles)
    best_solution = deepcopy(solution)
    best_cost = solution_cost(solution, timecalc, fixed_cost, variable_cost,num_vehicles)
    improved = false
    
    for p1 in keys(solution)
        for r1_idx in 1:length(solution[p1])
            route1 = solution[p1][r1_idx]
            capacity1 = vehicle_capacities[min(r1_idx, size(vehicle_capacities,1)), :]
            tm1 = get_vehicle_time_matrix(timecalc, p1, r1_idx, num_vehicles)
            
            for pos in 2:(length(route1)-1)
                cust = route1[pos]
                new_route1 = copy(route1)
                deleteat!(new_route1, pos)
                
                # Ensure the route is not empty after removal (keep at least depot)
                if length(new_route1) <= 2
                    continue
                end
                
                # Ensure the new route remains feasible
                if !is_route_feasible_period(new_route1, tm1, service_times, time_windows, demands, capacity1, depots, time_limit)
                    continue
                end
                
                for p2 in 1:n_periods
                    # Check if customer is available in the new period
                    if p2 != p1 && availability[cust - maximum(depots), p2] == 0
                        continue
                    end

                    for r2_idx in 1:length(solution[p2])
                        route2 = solution[p2][r2_idx]
                        capacity2 = vehicle_capacities[min(r2_idx, size(vehicle_capacities,1)), :]
                        tm2 = get_vehicle_time_matrix(timecalc, p2, r2_idx, num_vehicles)
                        
                        for ins in 2:length(route2)
                            new_route2 = copy(route2)
                            insert!(new_route2, ins, cust)
                            
                            # Ensure new route does not exceed vehicle capacity
                            if calculate_route_demand(new_route2, demands, depots) > capacity2
                                continue
                            end
                            
                            # Ensure feasibility of the new route
                            if is_route_feasible_period(new_route2, tm2, service_times, time_windows, demands, capacity2, depots, time_limit)
                                new_solution = deepcopy(solution)
                                new_solution[p1][r1_idx] = new_route1
                                new_solution[p2][r2_idx] = new_route2
                                new_cost = solution_cost(new_solution, timecalc, fixed_cost, variable_cost,num_vehicles)
                                
                                if new_cost < best_cost
                                    best_solution = new_solution
                                    best_cost = new_cost
                                    improved = true
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return best_solution, improved
end


# 3. Intra-route relocate: Reinsert a customer in a different position within the same route.
function intra_route_relocate_multiperiod(solution::Dict{Int, Vector{Vector{Int}}}, timecalc, service_times, time_windows, demands, vehicle_capacities, depots, time_limit,num_vehicles)
    best_solution = deepcopy(solution)
    best_cost = solution_cost(solution, timecalc, fixed_cost, variable_cost,num_vehicles)
    improved = false
    for p in keys(solution)
        for r_idx in 1:length(solution[p])
            route = solution[p][r_idx]
            capacity = vehicle_capacities[min(r_idx, size(vehicle_capacities,1)),:]
            if length(route) <= 3
                continue
            end
            tm = get_vehicle_time_matrix(timecalc, p, r_idx, num_vehicles)
            for pos in 2:(length(route)-1)
                cust = route[pos]
                new_route = copy(route)
                deleteat!(new_route, pos)
                for ins in 2:length(new_route)
                    candidate_route = copy(new_route)
                    insert!(candidate_route, ins, cust)
                    if is_route_feasible_period(candidate_route, tm, service_times, time_windows, demands, capacity, depots, time_limit)
                        new_solution = deepcopy(solution)
                        new_solution[p][r_idx] = candidate_route
                        new_cost = solution_cost(new_solution, timecalc, fixed_cost, variable_cost,num_vehicles)
                        if new_cost < best_cost
                            best_solution = new_solution
                            best_cost = new_cost
                            improved = true
                        end
                    end
                end
            end
        end
    end
    return best_solution, improved
end

# 4. Inter-route swap: Exchange customers between two routes (even across different periods).
function inter_route_swap_multiperiod(solution::Dict{Int, Vector{Vector{Int}}}, timecalc, service_times, time_windows, demands, vehicle_capacities, depots, time_limit, availability,num_vehicles)
    best_solution = deepcopy(solution)
    best_cost = solution_cost(solution, timecalc, fixed_cost, variable_cost,num_vehicles)
    improved = false
    
    periods = collect(keys(solution))
    
    for idx1 in 1:length(periods)-1
        for idx2 in (idx1+1):length(periods)
            p1 = periods[idx1]
            p2 = periods[idx2]
            
            for r1_idx in 1:length(solution[p1])
                for r2_idx in 1:length(solution[p2])
                    route1 = solution[p1][r1_idx]
                    route2 = solution[p2][r2_idx]
                    
                    capacity1 = vehicle_capacities[min(r1_idx, size(vehicle_capacities,1)),:]
                    capacity2 = vehicle_capacities[min(r2_idx, size(vehicle_capacities,2)),:]
                    tm1 = get_vehicle_time_matrix(timecalc, p1, r1_idx, num_vehicles)
                    tm2 = get_vehicle_time_matrix(timecalc, p2, r2_idx, num_vehicles)

                    
                    for pos1 in 2:(length(route1)-1)
                        for pos2 in 2:(length(route2)-1)
                            cust1 = route1[pos1]
                            cust2 = route2[pos2]
                            
                            # Ensure availability in the swapped period
                            if availability[cust1 - maximum(depots), p2] == 0 || availability[cust2 - maximum(depots), p1] == 0
                                continue
                            end
                            
                            new_route1 = copy(route1)
                            new_route2 = copy(route2)
                            
                            new_route1[pos1], new_route2[pos2] = new_route2[pos2], new_route1[pos1]
                            
                            # Ensure neither route is emptied
                            if length(new_route1) <= 2 || length(new_route2) <= 2
                                continue
                            end
                            
                            # Ensure both routes remain feasible
                            if is_route_feasible_period(new_route1, tm1, service_times, time_windows, demands, capacity1, depots, time_limit) &&
                               is_route_feasible_period(new_route2, tm2, service_times, time_windows, demands, capacity2, depots, time_limit)
                               
                                new_solution = deepcopy(solution)
                                new_solution[p1][r1_idx] = new_route1
                                new_solution[p2][r2_idx] = new_route2
                                new_cost = solution_cost(new_solution, timecalc, fixed_cost, variable_cost,num_vehicles)
                                
                                if new_cost < best_cost
                                    best_solution = new_solution
                                    best_cost = new_cost
                                    improved = true
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return best_solution, improved
end


# 5. Shake: Randomly remove a customer from one route and reinsert it (possibly in a different period).
function shake_multiperiod(solution::Dict{Int, Vector{Vector{Int}}}, timecalc, service_times, time_windows, demands, vehicle_capacities, depots, time_limit, availability, n_periods,num_vehicles)
    new_solution = deepcopy(solution)
    valid_periods = [p for p in keys(new_solution) if any(length(r) > 2 for r in new_solution[p])]
    if isempty(valid_periods)
        return new_solution
    end
    p1 = rand(valid_periods)
    routes_p1 = new_solution[p1]
    valid_routes = [r_idx for r_idx in 1:length(routes_p1) if length(routes_p1[r_idx]) > 2]
    if isempty(valid_routes)
        return new_solution
    end
    r1_idx = rand(valid_routes)
    route1 = new_solution[p1][r1_idx]
    pos = rand(2:(length(route1)-1))
    cust = route1[pos]
    deleteat!(route1, pos)
    new_solution[p1][r1_idx] = route1
    # Choose a period for reinsertion (can be same or different)
    p2 = rand(1:n_periods)
    if p2 != p1 && availability[cust - maximum(depots), p2] == 0
        p2 = p1  # fallback if not available in p2
    end
    routes_p2 = new_solution[p2]
    r2_idx = rand(1:length(routes_p2))
    route2 = new_solution[p2][r2_idx]
    ins = rand(2:length(route2))
    insert!(route2, ins, cust)
    capacity = vehicle_capacities[min(r2_idx, size(vehicle_capacities,1)),:]
    tm2 = get_vehicle_time_matrix(timecalc, p2, r2_idx, num_vehicles)

    if is_route_feasible_period(route2, tm2, service_times, time_windows, demands, capacity, depots, time_limit)
        new_solution[p2][r2_idx] = route2
    end
    return new_solution
end

# ---------------------------
# VNS Procedure for Multi-Period VRPTW
# ---------------------------

function vns_multiperiod_vrptw(timecalc, service_times, time_windows, demands, vehicle_capacities, vehicle_depots, depots, n_periods, availability, time_limit; max_iterations=350)
    current_solution, period_assignment = initial_solution_multiperiod(timecalc, service_times, time_windows, demands,
        vehicle_capacities, vehicle_depots, depots, n_periods, availability, time_limit,num_vehicles)
    best_solution = deepcopy(current_solution)
    best_cost = solution_cost(best_solution, timecalc, fixed_cost, variable_cost,num_vehicles)
    
    println("Initial solution cost: ", best_cost)
    
    iteration = 0
    while iteration < max_iterations
        iteration += 1
        improved = false
        # List of neighborhood operators (each returns a new solution and a flag)
        neighborhoods = [
            (sol) -> intra_route_2opt_multiperiod(sol, timecalc, service_times, time_windows, demands, vehicle_capacities, depots, time_limit,num_vehicles),
            (sol) -> inter_route_relocate_multiperiod(sol, timecalc, service_times, time_windows, demands, vehicle_capacities, depots, time_limit, availability, n_periods,num_vehicles),
            (sol) -> intra_route_relocate_multiperiod(sol, timecalc, service_times, time_windows, demands, vehicle_capacities, depots, time_limit,num_vehicles),
            (sol) -> inter_route_swap_multiperiod(sol, timecalc, service_times, time_windows, demands, vehicle_capacities, depots, time_limit, availability,num_vehicles)
        ]
        
        for neighborhood in neighborhoods
            new_solution, op_improved = neighborhood(current_solution)
            if op_improved
                current_solution = new_solution
                current_cost = solution_cost(current_solution, timecalc, fixed_cost, variable_cost,num_vehicles)
                if current_cost < best_cost
                    best_solution = deepcopy(current_solution)
                    best_cost = current_cost
                end
                improved = true
                break
            end
        end
        
        if !improved
            current_solution = shake_multiperiod(current_solution, timecalc, service_times, time_windows, demands, vehicle_capacities, depots, time_limit, availability, n_periods,num_vehicles)
        end
        
        if iteration % 50 == 0
            println("Iteration $iteration: best cost = $best_cost")
        end
    end
    return best_solution, best_cost, period_assignment
end

# ---------------------------
# Visualization: Plot Multi-Period Routes
# ---------------------------
function plot_multi_period_vrp(x_coords::Vector{Float64}, y_coords::Vector{Float64}, solution::Dict{Int, Vector{Vector{Int}}}, depot_coords, n_periods; file_name::String = "multi_period_VNS_vrptw_multi_product_solution.png")
    colors = [:red, :blue, :green, :orange, :purple, :cyan, :magenta, :black, :pink, :brown]
    line_styles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
    
    plt = scatter(x_coords, y_coords, marker=:circle, label="Customers", legend=:outertopright)
    
    # Mark depots.
    for (i, depot) in enumerate(depot_coords)
         scatter!(plt, [depot[1]], [depot[2]], marker=:star5, label="Depot $i", markersize=10, color=colors[i])
    end
    
    # Plot each route for every period.
    for p in 1:n_periods
        if haskey(solution, p)
            for (r_idx, route) in enumerate(solution[p])
                if length(route) > 1
                    cyclic_route = vcat(route, route[1])
                    route_x = [x_coords[node] for node in cyclic_route]
                    route_y = [y_coords[node] for node in cyclic_route]
                    label_str = "P$p, Route $r_idx"
                    plot!(plt, route_x, route_y, label=label_str, lw=1.5, color=colors[r_idx], linestyle=line_styles[mod1(p, length(line_styles))],title="Multi-Period VNS VRPTW multi productSolution")
                end
            end
        end
    end
    savefig(plt, file_name)
    println("Multi-Period VRPTW multi product solution plot saved as: $file_name")
end

# ---------------------------
# Run VNS for Multi-Period Multi-Depot VRPTW multi product
# ---------------------------
t = @elapsed begin
best_sol, best_sol_cost, period_assignment = vns_multiperiod_vrptw(timecalc, service_times, time_windows, demands,
    vehicle_capacities, vehicle_depots, 1:n_depots, n_periods, availability, time_limit_per_period, max_iterations=2000)

end
println("Elapsed time: $t seconds")


println("\nBest solution cost from VNS: ", best_sol_cost)
for p in 1:n_periods
    println("\nPeriod $p:")
    if haskey(best_sol, p)
        for (i, route) in enumerate(best_sol[p])
            #println("  Vehicle/Route $i: ", route, " Demand: ", calculate_route_demand(route, demands, 1:n_depots))
          # Calculate the route demand as a vector and print it.
          rd = calculate_route_demand(route, demands, 1:n_depots)
          println("  Vehicle/Route $i: ", route, " Demand: ", rd)
        
        end
    end
end

plot_multi_period_vrp(x_coords, y_coords, best_sol, depot_coords, n_periods)
