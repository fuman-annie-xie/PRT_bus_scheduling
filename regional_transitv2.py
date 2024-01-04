import pandas as pd
import numpy as np
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import gurobipy as gp
# from census import Census
# from us import states
# from shapely.geometry import Point, Polygon
# from pyproj import CRS

def data_cleaning():
    # data from Western Pennsylvania Regional Data Center
    df = pd.read_csv('./wprdc_stop_data.csv')
    schedule = pd.read_csv('./schedule_daily_agg.csv', low_memory=False)
    # fous on inbound stops and weekday
    df = df[(df.datekey == 202104) &
            ((df.direction == "Inbound") | (df.direction == 'Both')) &
            (df.serviceday == "Weekday")].dropna()
    df = df.dropna()
    route_set = df.route_name.unique()
    stopid_set = df.stop_id.unique()
    #Route stops riders
    route_stopid_rider = pd.DataFrame(index = list(route_set), columns = list(stopid_set))#R_ij
    stop_routes_dict = {}
    for index, row in df.iterrows():
        stop_id = row['stop_id']
        routes_ser = set(
            route.strip() for route in row['routes_ser'].split(','))  # Split the string into a set of routes
        stop_routes_dict[stop_id] = routes_ser
    route_stopid = dict.fromkeys(route_set, "")  # j_i
    route_stopcount = dict.fromkeys(route_set, 0)  # J_i
    # get R_ij for each route and each stop on this route
    for i in range(len(df)):
        stop_id = df.iloc[i, df.columns.get_loc("stop_id")]
        route_name = str(df.iloc[i, df.columns.get_loc("route_name")])
        route_stopid[route_name] += stop_id + ','
        route_stopcount[route_name] += 1
        direction = str(df.iloc[i, df.columns.get_loc("direction")])
        if direction == 'Both':
            route_stopid_rider.loc[route_name, stop_id] = df.iloc[i, df.columns.get_loc("avg_ons")] / 2
        else:
            route_stopid_rider.loc[route_name, stop_id] = df.iloc[i, df.columns.get_loc("avg_ons")]
    route_info = pd.DataFrame(index = list(route_set), columns = ['stopid', 'stopcount']).reset_index()
    for i in range(len(route_info)):
        route_info.iloc[i, 1] = route_stopid[route_info.iloc[i, 0]]
        route_info.iloc[i, 2] = route_stopcount[route_info.iloc[i, 0]]
    route_info.columns = ['RouteCode', 'Stopids','Stopcount']
    filtered_schedule = schedule[schedule["Date"] == "2021-03-19"].copy()
    filtered_schedule.loc[:, "per_trip_time"] = filtered_schedule["Trip_Mins"] / filtered_schedule["Trip.Count"]
    # remove the metro
    filtered_schedule = filtered_schedule[~filtered_schedule['RouteCode'].isin(['RED', 'BLUE', 'SLVR'])]
    # remove a duplicate
    filtered_schedule = filtered_schedule[~filtered_schedule.duplicated(subset=["RouteCode"], keep="first")]
    filtered_schedule = filtered_schedule.reset_index()
    # merge route info, which contains routecode, trip count, stop count,
    # average trip time, and each stop id (will be used in data viz)
    merged_route_info = route_info.merge(filtered_schedule, on = 'RouteCode')
    return merged_route_info,route_stopid_rider,stop_routes_dict

def create_route_df(i, merged_route_info, route_stopid_rider):
    routecode = merged_route_info.loc[i, "RouteCode"]
    tripcount = merged_route_info.loc[i, "Trip.Count"] // 2
    R_ij = route_stopid_rider.loc[routecode].dropna()
    route_df = pd.DataFrame(index=np.arange(0, tripcount), columns=R_ij.index)
    for t in range(tripcount):
        ratio = t / tripcount

        if ratio < 0.2 or ratio > 0.8:
            p = 1 / 10
        elif 0.4 <= ratio <= 0.6:
            p = 2 / 10
        else:
            p = 3 / 10

        R_ijt = R_ij * p / (tripcount / 5)
        route_df.iloc[t,] = R_ijt
    return route_df

def route_collection(merged_route_info,route_stopid_rider):
    route_df_collection = {}
    for i in range(len(merged_route_info)):
        routecode = merged_route_info.loc[i, "RouteCode"]
        route_df_collection[routecode] = create_route_df(i, merged_route_info, route_stopid_rider)
    return route_df_collection

def get_stop_map(dir):
    return gpd.read_file(dir)

# generate the indices
merged_route_info,route_stopid_rider,stop_routes_dict = data_cleaning()
R = route_collection(merged_route_info,route_stopid_rider)
routes = list(merged_route_info['RouteCode'])
stops = {}
trips = {}
Time = {}
for i in range(len(routes)):
    route = routes[i]
    stops[route] = int(merged_route_info[merged_route_info['RouteCode'] == route].loc[:,'Stopcount'])
    trips[route] = int(merged_route_info[merged_route_info['RouteCode'] == route].loc[:,'Trip.Count'])//2
    Time[route] = float(merged_route_info[merged_route_info['RouteCode'] == route].loc[:,'per_trip_time'])
stop_map = get_stop_map('./stop_map.json')
# Create the model within the Gurobi environment
params = {
"WLSACCESSID": '902b5959-51a7-4529-a2a2-3dc4c8a5052a',
"WLSSECRET": 'dbd9c5cd-8b98-4ab7-a9c4-bbcc6814ee78',
"LICENSEID": 2411593,
}
env = gp.Env(params=params)
model = gp.Model('Bus_Scheduling', env=env)

X = {}
for i in range(len(routes)):
    for j in range(stops[routes[i]]):
        for t in range(trips[routes[i]]):
            X[i, j, t] = model.addVar(name=f"route_{i}_stop_{j}_trip_{t}", vtype=gp.GRB.BINARY)

# OPTIGUIDE DATA CODE GOES HERE
# Set the objective function
# ticket price value
ticket_price = 2.75

# average cost value
avg_cost = 0.2

objective = (
    sum(
        R[routes[i]].iloc[t, j] * X[i, j, t] * ticket_price
        for i in range(len(routes))
        for j in range(stops[routes[i]])
        for t in range(trips[routes[i]])
    )
    - sum(
        (
            sum(X[i, j, t] for j in range(stops[routes[i]]))
            / stops[routes[i]]
            * avg_cost
            * Time[routes[i]]
        )
        for i in range(len(routes))
        for t in range(trips[routes[i]])
    )
)

model.setObjective(objective, gp.GRB.MAXIMIZE)

# Constraints
# Add 3 types of constraints
# Type 1: each stop must be passed at least once
pov_threshold = 0.1
min_threshold = 0.1
min_passes = 10

for stop_id, routes_set in stop_routes_dict.items():
    total_count = 0
    flag1 = 0
    for route in routes_set:
        if route in routes:
            i = routes.index(route)
            # make sure the stop is passed by the route
            if stop_id in R[route].columns:
                flag1 = 1
                j = R[route].columns.get_loc(stop_id)
                for t in range(trips[route]):
                    total_count += X[i, j, t]
    if flag1 == 1:
        model.addConstr(total_count >= 1, name=f"must_pass_constraint_{stop_id}")

# Type 2: each stop must be passed at least 10 times if local poverty rate is higher than 9%
#         if the scheduled stops < 10, then they all should be passed
    if (stop_map[stop_map.stop_id == stop_id].poverty_rate).all() > pov_threshold:
        total_count_pov = 0
        max_count_pov = 0
        flag2 = 0
        for route in routes_set:
            if route in routes:
                i = routes.index(route)
                if stop_id in R[route].columns:
                    flag2 = 1
                    j = R[route].columns.get_loc(stop_id)
                    for t in range(trips[route]):
                        total_count_pov += X[i, j, t]
                        max_count_pov += 1
        if flag2 == 1:
            model.addConstr(total_count_pov >= min(max_count_pov, min_passes), name=f"pov_constraint_{stop_id}")

# Type 3: each stop must be passed at least 10 times if local minority rate is higher than 20%
    if (stop_map[stop_map.stop_id == stop_id].minority_rate).all() > min_threshold:
        total_count_min = 0
        max_count_min = 0
        flag3 = 0
        for route in routes_set:
            if route in routes:
                i = routes.index(route)
                if stop_id in R[route].columns:
                    flag = 3
                    j = R[route].columns.get_loc(stop_id)
                    for t in range(trips[route]):
                        total_count_min += X[i, j, t]
                        max_count_min += 1
        if flag3 == 1:
            model.addConstr(total_count_min >= min(max_count_min, min_passes), name=f"minority_constraint_{stop_id}")

# Optimize the model
model.optimize()

m = model

# OPTIGUIDE CONSTRAINT CODE GOES HERE

# Solve
m.update()
model.optimize()

if m.status == gp.GRB.OPTIMAL:
    # gather the optimal solutions
    stopcount = 0
    cnt = 0
    # the policy for each route will be stored the dictionary below
    route_result_collection = {}
    for i in range(len(routes)):
        route = routes[i]
        temp_res = np.zeros((stops[route], trips[route]))
        for j in range(stops[routes[i]]):
            for t in range(trips[routes[i]]):
                cnt += 1
                temp_res[j, t] = int(X[i, j, t].x)
                if int(X[i,j,t].x) == 1:
                    stopcount += 1
                # print(f"x_route#{routes[i]}_stop#{j}_trip#{t} = {X[i, j, t].x}")
        route_result_collection[route] = temp_res

    # print a summary for the solution
    percentage_skipped = (1 - stopcount / cnt) * 100

    # Print the percentage as a formatted string
    print(f"{percentage_skipped:.2f}% stops will be skipped.")
    # route
    sorted_routes = sorted(
        [(route, route_result_collection[route].sum() / route_result_collection[route].size) for route in routes],
        key=lambda x: x[1], reverse=True)
    percentage_skipped_each_route = "Percentage of stops that can be skipped on each route."
    for route, per_skipped in sorted_routes:
        percentage_skipped_each_route += f"For route {route:4}, {per_skipped * 100:.2f}% stops will be skipped."
    print(percentage_skipped_each_route)

    # gather optimization results for each stop 
    stop_passes = {}
    for stop_id, routes_set in stop_routes_dict.items():
        total_count = 0
        for route in routes_set:
            if route in routes:
                i = routes.index(route)
                # make sure the stop is passed by the route
                if stop_id in R[route].columns:
                    j = R[route].columns.get_loc(stop_id)
                    for t in range(trips[route]):
                        total_count += int(X[i,j,t].x)
        stop_passes[stop_id] = total_count
    count_1 = sum(value == 1 for value in stop_passes.values())
    print("Number of stops that are only passed once:", count_1)
    count_3 = sum(value <= 3 for value in stop_passes.values())
    print("Number of stops that are passed less than three times:", count_3)
else:
    print("Not solved to optimality. Optimization status:", m.status)
