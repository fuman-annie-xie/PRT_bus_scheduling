This project is to provide an optimization plan for Pittsburgh Regional Transit. We want to maximize the efficiency with additional focus on equitable access to public transportation. Utilizing datasets from the Western Pennsylvania Regional Data Center and the American Community Survey, we focus on evaluating bus stop usage, scheduled trip counts, and demographic data such as poverty and minority rates within Allegheny County at a census-tract level.
<img width="886" alt="Screen Shot 2024-03-11 at 2 26 17 PM" src="https://github.com/fuman-annie-xie/PRT_bus_scheduling/assets/114703755/559556dd-70c5-451c-8637-104e06f527c2">

Based on our optimization, we calculated the skip-rates for each route and the number of passes for each bus stop. We also provide a visualization for stop probability for a selected route as shown below. PRT can then quickly
identify bus routes/stops that may need to be adjusted and make adjustments accordingly. With the help of Optiguide and an iterative approach, PRT can adjust the model with few efforts, which makes sure that they can provide the most needed services in a timely manner even with changing objectives and constraints.

<img width="809" alt="Screen Shot 2024-03-11 at 2 28 39 PM" src="https://github.com/fuman-annie-xie/PRT_bus_scheduling/assets/114703755/d548b6e5-1865-4bee-8910-80c2321fb70d">

To see our data cleaning process, Gurobi optimization and the visualized results of our base model, run DA_Final_Project.ipynb
Note that you may need to install some packages (census, us and shapely) to run the notebook. If you run into any problems, feel free to email us.

To run our optiguide Q&A, run optiguide_regional_transit.ipynb

The rest files are either data sources or helper scripts that should not be modified.
