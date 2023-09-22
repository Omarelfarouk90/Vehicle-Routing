# Vehicle-Routing
The aim is to optimize the Vehicle routing using Google OR  Machine learning and Simulated annealing

Algorithm used for solving the vehicle routing problem
1. Apply the branch and bound algorithm with Google OR library in Python
2. Implementing the constraints as follows:
•	The electric vehicles are deployed from the deporting location fully charged.
•	The electric vehicles return to the deployed point after finishing their route.
•	The electric vehicle will always stay 30 min in any charging point regardless of the amount of charge available for safety measures.
•	Maximum number of charging points available is 5 charging points
•	Maximum time duration is 4 hrs, (as it starts from 8 am and finishes at noon).
3. Plotting the results of the optimized route along with the time duration
Case 1: Assuming all electric vehicle for passing along all locations and returning to the deployment point, considering charging time of 30 min and range of 40 km distance traveled per electric vehicle charging capacity
Initial iteration:
Assuming 1 vehicle for passing along all locations and returning to the deployment point.
 
•	The results of the 1st iteration were provided as follows.
•	Total distance: 189 km
•	Total time without charging: 6.3 hrs
•	Total time with charging: 8.3 hrs
•	Number of charging points:4
Which would result in an infeasible solution due to the time constraint
The second iteration will involve k means clustering in order to reduce the total time by selecting an electric vehicle for each region, The results for K=2 and K=3 were omitted as well because of the time constraint for each region.
Implementing k means clustering raphically by dividing the route in 4 regions.
 
 


Total distance: 65 km
Total time without charging: 2.1666666666666665 hrs
Total time with charging: 2.6666666666666665 hrs
number of charging stations 1
charging stations location (0.2, 24.8)
	 

Total distance: 76 km
Total time without charging: 2.533333333333333 hrs
Total time with charging: 3.033333333333333 hrs
number of charging stations 1
location of charging stations (17.54, 29.43)

 

Total distance: 51 km
Total time without charging: 1.7 hrs
Total time with charging: 2.2 hrs
number of charging stations 1
location of the charging stations (3.69, 14.33)
	 

Total distance: 57 km
Total time without charging: 1.9 hrs
Total time with charging: 2.4 hrs
Number of  charging point: 1 point
charging point location (17.02, 12.33)

As a result , 
Total distance of 4 vehicles combined =249 km
Maximum time =3.03 hrs
Number of charging stations =4 , one for each route
Location of charging stations = (3.69, 14.33), (0.2, 24.8), (17.54, 29.43), (17.02, 12.33)























Case 2: Assuming all Vehicle for passing along all locations and returning to the deployment point, considering charging time of 30 min and range of 60 km distance traveled per electric vehicle charging capacity


 
 

Total distance: 91 km
Total time without charging: 3.033333333333333 hrs
Total time with charging: 3.533333333333333 hrs
number of charging stations 1
charging stations location [(3.3, 17.86)])
	 

Total distance: 80 km
Total time without charging: 2.6666666666666665 hrs
Total time with charging: 3.1666666666666665 hrs
number of charging stations 1
location of charging stations ([(14.2, 19.64)])

	 


Total distance: 74 km
Total time without charging: 2.466666666666667 hrs
Total time with charging: 2.966666666666667 hrs
Number of  charging point: 1 point
charging point location ([(10.24, 3.02)]) coordinates

Total covered distance = 245 km
Maximum time =3.53 hrs
Number of vehicles used =3
Location of charging stations(3.3, 17.86), (14.2, 19.64),(10.24, 3.02)


Case 3: Assuming all Vehicle for passing along all locations and returning to the deployment point, considering charging time of 15 min and range of 60 km distance traveled per electric vehicle charging capacity


 
 

Total distance: 65 km
Total time without charging: 2.1666666666666665 hrs
Total time with charging: 2.4166666666666665 hrs
number of charging stations 1
charging stations location ([(0.2, 24.8)])
	 

Total distance: 76 km
Total time without charging: 2.533333333333333 hrs
Total time with charging: 2.783333333333333 hrs
number of charging stations 1
location of charging stations ([(17.54, 29.43)])

	 


Total distance: 57 km
Total time without charging: 1.9 hrs
Total time with charging: 2.15 hrs
Number of  charging point: 1 point
charging point location ([(17.02, 12.33)]) coordinates

Total covered distance = 198 km
Maximum time =2.78 hrs
Number of vehicles used =3
Number of charging stations used
Location of charging stations((0.2, 24.8), (17.54, 29.43),( 17.02, 12.33)


In conclusion, the charging time and the electric vehicle range affect the time and the number of vehicles used significantly.

