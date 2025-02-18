using Random, LinearAlgebra, Plots

# ---------------------------
# Instance Import from File
# ---------------------------
function load_instance(file_path::String)
    # Read all lines and remove empty lines
    lines = filter(x -> !isempty(strip(x)), readlines(file_path))
    
    # The file structure:
    # Line 1: instance name (e.g., "C101")
    # Line 2: header "VEHICLE NUMBER CAPACITY"
    # Line 3: vehicle info (e.g., "25 200")
    # Line 4: header for customer data
    # Line 5 onward: customer data rows, each with:
    #    CUST NO.  XCOORD  YCOORD  DEMAND  READY TIME  DUE DATE  SERVICE TIME
    
    instance_name = strip(lines[1])
    vehicle_info_line = strip(lines[3])
    parts = split(vehicle_info_line)
    n_vehicles = parse(Int, parts[1])
    vehicle_capacity = parse(Int, parts[2])
    
    # Customer data starts at line 5
    customer_data_lines = lines[5:end]
    n_nodes = length(customer_data_lines)  # includes depot as first row
    
    # Initialize arrays. We assume the file already lists the depot as customer 0.
    x_coords = zeros(n_nodes)
    y_coords = zeros(n_nodes)
    demands = zeros(Int, n_nodes)
    service_times = zeros(n_nodes)
    time_windows = Vector{Tuple{Float64,Float64}}(undef, n_nodes)
    
    for (i, line) in enumerate(customer_data_lines)
        parts = split(strip(line))
        # parts: [CUST_NO, XCOORD, YCOORD, DEMAND, READY_TIME, DUE_DATE, SERVICE_TIME]
        # Parse the values
        # Note: The depot should have demand 0 and service time 0.
        cust_no = parse(Int, parts[1])
        x = parse(Float64, parts[2])
        y = parse(Float64, parts[3])
        d = parse(Int, parts[4])
        ready = parse(Float64, parts[5])
        due = parse(Float64, parts[6])
        st = parse(Float64, parts[7])
        
        x_coords[i] = x
        y_coords[i] = y
        demands[i] = d
        service_times[i] = st
        time_windows[i] = (ready, due)
    end
    
    return x_coords, y_coords, service_times, time_windows, demands, n_vehicles, vehicle_capacity
end

# ---------------------------
# Import the instance data
# ---------------------------
file_path = "C_1_10_1.TXT"   # ensure this file is in your working directory
x_coords, y_coords, service_times, time_windows, demands, n_vehicles, vehicle_capacity = load_instance(file_path)
n_nodes = length(x_coords)

println("Imported Customer Data:")
for i in 1:n_nodes
    println("Node $i: (x=$(x_coords[i]), y=$(y_coords[i])), Service Time=$(service_times[i]), Demand=$(demands[i]), Time Window=$(time_windows[i])")
end

# ---------------------------
# Utility Functions (unchanged)
# ---------------------------

# Compute Euclidean distance matrix
function calculate_distance_matrix(x_coords, y_coords)
    n = length(x_coords)
    dist = zeros(n, n)
    for i in 1:n, j in 1:n
        dist[i,j] = sqrt((x_coords[i]-x_coords[j])^2 + (y_coords[i]-y_coords[j])^2)
    end
    return dist
end

dist = calculate_distance_matrix(x_coords, y_coords)

# Check feasibility of a route
function is_route_feasible(route::Vector{Int}, dist, service_times, time_windows, demands, capacity)
    current_time = 0.0
    total_demand = 0
    # route is assumed to start and end with depot (node 1)
    for k in 1:(length(route)-1)
        i = route[k]
        j = route[k+1]
        travel_time = dist[i,j]
        arrival = current_time + travel_time
        tw_start, tw_end = time_windows[j]
        service_start = max(arrival, tw_start)
        if service_start > tw_end
            return false
        end
        current_time = service_start + service_times[j]
        if j != 1  # accumulate demand (skip depot)
            total_demand += demands[j]
            if total_demand > capacity
                return false
            end
        end
    end
    return true
end

# Compute route cost (total distance)
function route_cost(route::Vector{Int}, dist)
    cost = 0.0
    for k in 1:(length(route)-1)
        cost += dist[route[k], route[k+1]]
    end
    return cost
end

# Total cost of a solution (sum over all routes)
function solution_cost(solution::Vector{Vector{Int}}, dist)
    total = 0.0
    for route in solution
        total += route_cost(route, dist)
    end
    return total
end

# ---------------------------
# Initial Solution Construction (unchanged)
# ---------------------------
function initial_solution(dist, service_times, time_windows, demands, capacity, num_vehicles)
    n_nodes = size(dist,1)
    unassigned = collect(2:n_nodes)   # all customers (excluding depot at index 1)
    solution = Vector{Vector{Int}}()
    
    for v in 1:num_vehicles
        route = [1]   # start at depot
        current_time = 0.0
        current_load = 0
        changed = true
        while changed && !isempty(unassigned)
            changed = false
            best_cust = nothing
            best_increase = Inf
            best_new_time = 0.0
            last = route[end]
            for cust in unassigned
                travel_time = dist[last, cust]
                arrival = current_time + travel_time
                tw_start, tw_end = time_windows[cust]
                service_start = max(arrival, tw_start)
                if service_start > tw_end
                    continue
                end
                new_load = current_load + demands[cust]
                if new_load > capacity
                    continue
                end
                # Simple cost measure: extra distance added by visiting 'cust'
                cost_increase = dist[last, cust] + dist[cust, 1] - dist[last, 1]
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
                current_load += demands[best_cust]
                changed = true
            end
        end
        push!(route, 1)  # return to depot
        push!(solution, route)
        if isempty(unassigned)
            break
        end
    end
    # For any customers not assigned, create individual routes
    for cust in unassigned
        push!(solution, [1, cust, 1])
    end
    return solution
end

# ---------------------------
# Neighborhood Operators (unchanged)
# ---------------------------
function intra_route_2opt(solution, dist, service_times, time_windows, demands, capacity)
    best_solution = deepcopy(solution)
    best_cost = solution_cost(solution, dist)
    improved = false
    for r in 1:length(solution)
        route = solution[r]
        n_route = length(route)
        if n_route <= 3
            continue
        end
        for i in 2:(n_route-2)
            for j in (i+1):(n_route-1)
                new_route = copy(route)
                new_route[i:j] = reverse(new_route[i:j])
                if is_route_feasible(new_route, dist, service_times, time_windows, demands, capacity)
                    new_solution = deepcopy(solution)
                    new_solution[r] = new_route
                    new_cost = solution_cost(new_solution, dist)
                    if new_cost < best_cost
                        best_solution = new_solution
                        best_cost = new_cost
                        improved = true
                    end
                end
            end
        end
    end
    return best_solution, improved
end

function inter_route_relocate(solution, dist, service_times, time_windows, demands, capacity)
    best_solution = deepcopy(solution)
    best_cost = solution_cost(solution, dist)
    improved = false
    for r1 in 1:length(solution)
        route1 = solution[r1]
        for pos in 2:(length(route1)-1)
            cust = route1[pos]
            new_route1 = copy(route1)
            deleteat!(new_route1, pos)
            if !is_route_feasible(new_route1, dist, service_times, time_windows, demands, capacity)
                continue
            end
            for r2 in 1:length(solution)
                if r1 == r2
                    continue
                end
                route2 = solution[r2]
                for ins in 2:length(route2)
                    new_route2 = copy(route2)
                    insert!(new_route2, ins, cust)
                    if is_route_feasible(new_route2, dist, service_times, time_windows, demands, capacity)
                        new_solution = deepcopy(solution)
                        new_solution[r1] = new_route1
                        new_solution[r2] = new_route2
                        new_cost = solution_cost(new_solution, dist)
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

function intra_route_relocate(solution, dist, service_times, time_windows, demands, capacity)
    best_solution = deepcopy(solution)
    best_cost = solution_cost(solution, dist)
    improved = false
    for r in 1:length(solution)
        route = solution[r]
        if length(route) <= 3
            continue
        end
        for pos in 2:(length(route)-1)
            cust = route[pos]
            new_route = copy(route)
            deleteat!(new_route, pos)
            for ins in 2:length(new_route)
                candidate_route = copy(new_route)
                insert!(candidate_route, ins, cust)
                if is_route_feasible(candidate_route, dist, service_times, time_windows, demands, capacity)
                    new_solution = deepcopy(solution)
                    new_solution[r] = candidate_route
                    new_cost = solution_cost(new_solution, dist)
                    if new_cost < best_cost
                        best_solution = new_solution
                        best_cost = new_cost
                        improved = true
                    end
                end
            end
        end
    end
    return best_solution, improved
end

function inter_route_swap(solution, dist, service_times, time_windows, demands, capacity)
    best_solution = deepcopy(solution)
    best_cost = solution_cost(solution, dist)
    improved = false
    for r1 in 1:length(solution)-1
        for r2 in (r1+1):length(solution)
            route1 = solution[r1]
            route2 = solution[r2]
            for pos1 in 2:(length(route1)-1)
                for pos2 in 2:(length(route2)-1)
                    new_route1 = copy(route1)
                    new_route2 = copy(route2)
                    new_route1[pos1], new_route2[pos2] = new_route2[pos2], new_route1[pos1]
                    if is_route_feasible(new_route1, dist, service_times, time_windows, demands, capacity) &&
                       is_route_feasible(new_route2, dist, service_times, time_windows, demands, capacity)
                        new_solution = deepcopy(solution)
                        new_solution[r1] = new_route1
                        new_solution[r2] = new_route2
                        new_cost = solution_cost(new_solution, dist)
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

function shake(solution, dist, service_times, time_windows, demands, capacity)
    new_solution = deepcopy(solution)
    routes_with_customers = [r for r in 1:length(new_solution) if length(new_solution[r]) > 2]
    if isempty(routes_with_customers)
        return new_solution
    end
    r = rand(routes_with_customers)
    route = new_solution[r]
    pos = rand(2:(length(route)-1))
    cust = route[pos]
    deleteat!(route, pos)
    new_solution[r] = route
    r2 = rand(1:length(new_solution))
    route2 = new_solution[r2]
    ins = rand(2:length(route2))
    insert!(route2, ins, cust)
    if is_route_feasible(route2, dist, service_times, time_windows, demands, capacity)
        new_solution[r2] = route2
    end
    return new_solution
end

# ---------------------------
# VNS Procedure for VRPTW (unchanged)
# ---------------------------
function vns_vrptw(dist, service_times, time_windows, demands, capacity, num_vehicles; max_iterations=5)
    current_solution = initial_solution(dist, service_times, time_windows, demands, capacity, num_vehicles)
    best_solution = deepcopy(current_solution)
    best_cost = solution_cost(best_solution, dist)
    
    iteration = 0
    while iteration < max_iterations
        iteration += 1
        println("Iteration $iteration: cost = $best_cost")
        neighborhoods = [intra_route_2opt, inter_route_relocate, intra_route_relocate, inter_route_swap]
        improvement = false
        for neighborhood in neighborhoods
            new_solution, improved = neighborhood(current_solution, dist, service_times, time_windows, demands, capacity)
            if improved
                current_solution = new_solution
                current_cost = solution_cost(current_solution, dist)
                if current_cost < best_cost
                    best_solution = deepcopy(current_solution)
                    best_cost = current_cost
                end
                improvement = true
                break
            end
        end
        if !improvement
            current_solution = shake(current_solution, dist, service_times, time_windows, demands, capacity)
        end
    end
    return best_solution, best_cost
end

# ---------------------------
# Run VNS for VRPTW with imported data
# ---------------------------
best_sol, best_sol_cost = vns_vrptw(dist, service_times, time_windows, demands, vehicle_capacity, n_vehicles, max_iterations=2000)

println("\nBest solution cost from VNS: ", best_sol_cost)
for (i, route) in enumerate(best_sol)
    println("Route $i: ", route)
end

using Plots

function plot_vrp_solution(x_coords, y_coords, solution, file_name="vrp_solution.png")
    n_nodes = length(x_coords)

    # Generate a color palette with enough distinct colors for all routes
    colors = distinguishable_colors(length(solution))

    # Plot customers and depot
    scatter(x_coords, y_coords, marker=:circle, label="Customers", legend=:topleft)
    scatter!([x_coords[1]], [y_coords[1]], marker=:star5, label="Depot", markersize=10, color=:red)

    # Draw each route with a different color
    for (i, route) in enumerate(solution)
        x_route = [x_coords[node] for node in route]
        y_route = [y_coords[node] for node in route]
        plot!(x_route, y_route, label="", lw=1.5, color=colors[i])
    end

    # Save the plot as a PNG file
    savefig(file_name)
    println("VRP solution plot saved as $file_name")
end

# Call the function with the best solution
plot_vrp_solution(x_coords, y_coords, best_sol)
