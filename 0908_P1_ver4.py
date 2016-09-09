# -*- coding: utf-8 -*-
#1) Ready step


"""
Readme

instance file solomon 100 instance
population size 100
drone : depot to customer P2P
"""

saved_file_name = 'test'#raw_input("type saved file name(AhnYoungJae,xoxoq,user1):")
user_name_for_directory = 'xoxoq' #raw_input("type user name:")
data_name = 'RC105_h_3' #raw_input("type data no:")
modal_pb = 0.31 #float(raw_input("type possibility of modal choice in initial solution(if 1 = only vehicle):"))
route_reducing_iteration = 10 #int(raw_input("type Route Reducing number:"))
local_search_iteration = 100 #int(raw_input("type Local search number:"))
EA_population_size = 10 #int(raw_input("type population size:"))
instance_customer_size = 3 # int(raw_input("type number of customer in the instance:"))
good_route_type = 1 #int(raw_input("choose good route parameter [1: low total coste], [2: low time cost], [3: low route cost]:"))
good_solution_type  = 1 #int(raw_input("choose good solution parameter [1: low total coste], [2: low time cost], [3: low route cost]:"))

#1-0) import the needed module 
from geopy.distance import vincenty
#from scipy.spatial import distance
import copy
import math
import random
import operator
import itertools
import time
import datetime

#1-1) define the parameter and its values

D_d = 1.2  #드론의 거리상수 
D_v = 1.7  #차량의 거리상수 
C_d = 1    #드론의 단위 운행거리 당 비용
C_v = 8  #차량의 단위 운행거리 당 비용
Q_k = 1500 #차량 k의 용량 
Q_d = 25   #드론 d의 용량 
ET_a = 0.1    #배송 차량이나 드론이 ET_i 보다 먼저 간 경우의 비용을 계산 하는데 사용되는 비용계수 (a>0)
LT_b = 0.1    #배송 차량이나 드론이, LT_i 보다 늦게 간 경우의 비용을 계산 하는데 사용되는 비용 계수 (b>0)
ET_c = -0.1    #배송 차량이나 드론이, 〖ET〗_i 보다 일찍 간 경우 추가적인 이윤을 계산하기 위한 계수 (c<0) 
M = 25   #드론 이 한번에 비행할 수 있는 거리범위
EL = 0.3 #시간당 비이익 
speed_d = 3 #6#1 #416 #드론의 이동 속도: 0.9km/min or 25km/h (h=60분)
speed_v = 2 #3 #1 #286  #차량 이동 속도: 0.29km/min or 17.2km/h (h=60분)
Max_time = 0  #data1[9][5]: 드론이 하루 중 운행을 할 수 있는 최대 시간. 즉 드론 라우트의 최대 시간. 
search_range = 35
VF = 2000 #차량의 대당 고정 비용
DF = 1000 #드론의 대당 고정 비용
non_profit_cost = 0.3
population_size = 100

""" #ReadMe#
calculated data
1.distance
type = dict
structure 
key = (a,b) a = node b = node  # note that cycle also possible
value = distance between a and b (float type)
e.g.distance1[(a,b)] = distance between a and b
2.time_v
type = dict
structure 
key = (a,b) a = node b = node  # note that cycle also possible
value = moving time between a and b by vehicle (float type)
3.time_d
type = dict
structure 
key = (a,b) a = node b = node  # note that cycle also possible
value = moving time  between a and b by drone (float type)
4.nodes_instance_dict
type = dict
structure 
key = (a) a = node  
value = list of information of node in the given data
[node number, x_coordinate, y_coordinate, weight, time_window_start_time, time_window_end_time, normal delivery index, fast_delivery_index, node type]
normal delivery index and fast_delivery_index is mutual exclusive relationship and it takes binary value{0,1}
node type {-2: drone possible node by depot(-2),-1: drone possible node by depot(-1),1: vehilce must node, 3: depot,0:distribution center}
[fitness, cost, [entime_weight], mode_index]

checker
calculator
provider
"""


#1-2) open instance file

def read_instance_file(user_name_for_directory,data_name):
	instance = open("C:/Users/" + user_name_for_directory + "/Desktop/data/" + data_name +".txt",'r')
	instance_lines = instance.readlines() 
	global instance_data
	instance_data = []
	customer_start_line = 9
	for line in range(0, len(instance_lines)):
		unsplited_line = instance_lines[line]   # b is variable for writing data to alist.
		splited_line = unsplited_line.split()
		int_splited_line = []
		if line >= customer_start_line:
			for element in splited_line:
				int_splited_line.append(int(element))
			instance_data.append(int_splited_line)
		else:
			instance_data.append(splited_line)
	global nodes_instance_dict
	nodes_instance_dict = {}
	for node_information in range(customer_start_line,len(instance_data)):
		nodes_instance_dict[node_information - 11] = instance_data[node_information]
	global distance
	global time_d
	global time_v
	distance = {}
	time_d = {}
	time_v = {}
	global Vehicle_customer
	global Drone_customer_1
	global Drone_customer_2
	global Depot
	global Distribution_center
	Drone_customer_1 = {}
	Drone_customer_2 = {}
	Vehicle_customer = {}
	Deopt = {}
	Distribution_center = {}
	for start_node in range(customer_start_line,len(instance_data)):
		for end_node in range(customer_start_line,len(instance_data)):
			#start_node = int(instance_data[start_node][0])
			#end_node = int(instance_data[end_node][0])
			meter_distance = round(math.sqrt(((int(instance_data[start_node][1])-int(instance_data[end_node][1]))**2+((int(instance_data[start_node][2])-int(instance_data[end_node][2]))**2))),4)
			distance[(int(instance_data[start_node][0]),int(instance_data[end_node][0]))] = meter_distance #문제 생기는 곳(distribution center 수 증가 시 )
			time_d[(int(instance_data[start_node][0]),int(instance_data[end_node][0]))] = round((float(meter_distance)/float(speed_d)),4)
			time_v[(int(instance_data[start_node][0]),int(instance_data[end_node][0]))] = round((float(meter_distance)/float(speed_v)),4)
	for node in range(customer_start_line,len(instance_data)):
		if int(instance_data[node][0]) > 0:
			if int(instance_data[node][3]) >= Q_d:
				Vehicle_customer[int(instance_data[node][3])] = instance_data[node]
				Vehicle_customer[int(instance_data[node][3])].append(1)
				instance_data[node].append(1)
				nodes_instance_dict[int(instance_data[node][0])].append(1)
			else:
				if round(distance[int(instance_data[node][-1]),-1],4) <= M:
					Drone_customer_1[int(instance_data[node][3])] = instance_data[node]
					Drone_customer_1[int(instance_data[node][3])].append(-1)
					instance_data[node].append(-1)
					nodes_instance_dict[int(instance_data[node][0])].append(-1)
				if round(distance[int(instance_data[node][-1]),-2],4) <= M:
					Drone_customer_1[int(instance_data[node][3])] = instance_data[node]
					Drone_customer_1[int(instance_data[node][3])].append(-2)
					instance_data[node].append(-2)
					nodes_instance_dict[int(instance_data[node][0])].append(-2)
		else:
			if int(instance_data[node][0]) == 0:
				Deopt[int(instance_data[node][3])] = instance_data[node]
				Deopt[int(instance_data[node][3])].append(2)
				instance_data[node].append(0)
				nodes_instance_dict[int(instance_data[node][0])].append(0)
			if int(instance_data[node][0]) < 0:
				Distribution_center[int(instance_data[node][3])] = instance_data[node]
				Distribution_center[int(instance_data[node][3])].append(3)
				instance_data[node].append(3)
				nodes_instance_dict[int(instance_data[node][0])].append(3)
	global Max_time
	Max_time = int(nodes_instance_dict[0][5])  
	for node in nodes_instance_dict.keys():
		if int(node) > 0:
			if int(nodes_instance_dict[node][4]) < 0:
				nodes_instance_dict[node][4] = 0
			if int(nodes_instance_dict[node][5]) > Max_time:
				nodes_instance_dict[node][5] = Max_time
				if int(nodes_instance_dict[node][5]) - int(nodes_instance_dict[node][4]) <= 100:
					nodes_instance_dict[node][4] = Max_time - 100
	for customer_key in nodes_instance_dict.keys():
		nodes_instance_dict[customer_key] = copy.deepcopy(nodes_instance_dict[customer_key][:9])
	return 'done'

def customer_time_classifier(nodes_instance_dict):
	time1_customer_list = []
	time2_customer_list = []
	time3_customer_list = []
	time4_customer_list = []
	normal_customer_list = []
	for customer in nodes_instance_dict.keys():
		if customer > 0:
			if nodes_instance_dict[customer][7] == 0:
				normal_customer_list.append([nodes_instance_dict[customer][0], nodes_instance_dict[customer][4] ,nodes_instance_dict[customer][5], 0])
			if nodes_instance_dict[customer][7] == 1:
				time1_customer_list.append([nodes_instance_dict[customer][0], nodes_instance_dict[customer][4] ,nodes_instance_dict[customer][5], 1])
			if nodes_instance_dict[customer][7] == 2:
				time2_customer_list.append([nodes_instance_dict[customer][0], nodes_instance_dict[customer][4] ,nodes_instance_dict[customer][5], 2])
			if nodes_instance_dict[customer][7] == 3:
				time3_customer_list.append([nodes_instance_dict[customer][0], nodes_instance_dict[customer][4] ,nodes_instance_dict[customer][5], 3])
			if nodes_instance_dict[customer][7] == 4:
				if 120 <= nodes_instance_dict[customer][4] < 260:
					time1_customer_list.append([nodes_instance_dict[customer][0], nodes_instance_dict[customer][4] ,nodes_instance_dict[customer][5], 4])
				if 260 <= nodes_instance_dict[customer][4] < 400:
					time2_customer_list.append([nodes_instance_dict[customer][0], nodes_instance_dict[customer][4] ,nodes_instance_dict[customer][5], 4])
				if 400 <= nodes_instance_dict[customer][4] <= 540:
					time3_customer_list.append([nodes_instance_dict[customer][0], nodes_instance_dict[customer][4] ,nodes_instance_dict[customer][5], 4])
				time4_customer_list.append([nodes_instance_dict[customer][0], nodes_instance_dict[customer][4] ,nodes_instance_dict[customer][5], 4])
	time1_customer_list.sort(key = operator.itemgetter(1))
	time2_customer_list.sort(key = operator.itemgetter(1))
	time3_customer_list.sort(key = operator.itemgetter(1))
	time4_customer_list.sort(key = operator.itemgetter(1))
	return [time1_customer_list, time2_customer_list,time3_customer_list,time4_customer_list,normal_customer_list]

def initial_route_maker(nodes_instance_dict):
	nodes_info = customer_time_classifier(nodes_instance_dict)
	time1_customer_list = nodes_info[0]
	time2_customer_list = nodes_info[1]
	time3_customer_list = nodes_info[2]
	time4_customer_list = nodes_info[3]
	normal_customer_list = nodes_info[4]
	print "고객분류 완료!!"
	Vehicle_route_set = {1:[],2:[],3:[],4:[]}
	operation_team = 1
	out_sourced_customer = []
	for target_customer_list in [time1_customer_list,time2_customer_list,time3_customer_list]:
		operation_team += 1
		while len(target_customer_list) != 0:
			route = [0,0,[0,0,0],operation_team]
			depot_random_selector = random.random()
			if depot_random_selector < 0.5:
				route.append(-1)
			else:
				route.append(-2)
			route.append(0)
			route.append(route[4])
			del_list = []
			for inserted_customer_info in target_customer_list:
				route_test = copy.deepcopy(route)
				current_customer = int(inserted_customer_info[0])
				#print "선택된 고객", current_customer
				route_test.insert(-1, current_customer)
				#print "라우트 ", route_test
				if route_feasible_checker(route_test) == 1:
					print "제거 전 ",len(time4_customer_list) + len(target_customer_list)
					if inserted_customer_info in time4_customer_list:
						time4_customer_list.remove(inserted_customer_info)
						#print "time4에 있는 고객 "
					if inserted_customer_info in target_customer_list:
						target_customer_list.remove(inserted_customer_info)
					route = copy.deepcopy(route_test)
					#print "제거 후 ",len(time4_customer_list) + len(target_customer_list)
				#raw_input("type anything")  #<- 문제가 생기면 여기의 채커를 사동 시킬 것 수동으로 확인 해야 함.
			Vehicle_route_set[operation_team].append(route)
	print "우선 고객 끝 "
	print "고객 set 1 남은 고객 ", len(time1_customer_list)
	print "고객 set 2 남은 고객 ", len(time2_customer_list)
	print "고객 set 3 남은 고객 ", len(time3_customer_list)
	print "고객 set 4 남은 고객 ", len(time4_customer_list)
	for customer_info in normal_customer_list + time4_customer_list:
		customer_node = customer_info[0]
		print customer_node
		indicator = 0
		for route_set_key in Vehicle_route_set.keys():
			for route_list in range(0,len(Vehicle_route_set[route_set_key])):
				depot = Vehicle_route_set[route_set_key][route_list][4]
				if (distance[depot, -1] + distance[depot,customer_node])*D_v/speed_v > nodes_instance_dict[customer_node][5] - route_start_time_calculator(Vehicle_route_set[route_set_key][route_list]) and (distance[depot, -2] + distance[depot,customer_node])*D_v/speed_v > nodes_instance_dict[customer_node][5] - route_start_time_calculator(Vehicle_route_set[route_set_key][route_list]):
					out_sourced_customer.append(customer_node)
					indicator = 1
					print customer_node
					print (distance[depot, -1] + distance[depot,customer_node])*D_v/speed_v, ">", nodes_instance_dict[customer_node][5] - route_start_time_calculator(Vehicle_route_set[route_set_key][route_list])
					print (distance[depot, -2] + distance[depot,customer_node])*D_v/speed_v, ">", nodes_instance_dict[customer_node][5] - route_start_time_calculator(Vehicle_route_set[route_set_key][route_list])
					break
				if customer_inserter_in_route(Vehicle_route_set[route_set_key][route_list], customer_node) != 0:
					Vehicle_route_set[route_set_key][route_list] = customer_inserter_in_route(Vehicle_route_set[route_set_key][route_list], customer_node)
					indicator = 1
					if customer_info[3] == 4:
						time4_customer_list.remove(customer_info)
					else:
						normal_customer_list.remove(customer_info)
					break
			if indicator == 1:
				break
		raw_input("type anything")  #<- 문제가 생기면 여기의 채커를 사동 시킬 것 수동으로 확인 해야 함.
	print "처리 불가 고객 ",out_sourced_customer
	print "고객 set 4 남은 고객 ", len(time4_customer_list)
	print "일반 고객 수  ", len(normal_customer_list)
	if len(time4_customer_list) + len(normal_customer_list) > 0:
		print time4_customer_list,';', normal_customer_list
	return Vehicle_route_set

def customer_inserter_in_route(route, customer):
	for position in range(5,len(route) - 1):
		route_test = copy.deepcopy(route)
		test = copy.deepcopy(route_test.insert(position, customer))
		if route_feasible_checker(test) == 1:
			return test
	return 0

def infeasible_checker(customer_node):
	dist_depot1 = distance[depot, -1] + distance[depot,customer_node]
	dist_depot2 = distance[depot, -2] + distance[depot,customer_node]
	end_time = nodes_instance_dict[customer_node][5]
	operation_start_time = route_start_time_calculator[0,0,0nodes_instance_dict[customer_node][7]]

	if dist_depot1*D_v/speed_v > end_time - operation_start_time
					depot = Vehicle_route_set[route_set_key][route_list][4]
				if (distance[depot, -1] + distance[depot,customer_node])*D_v/speed_v > nodes_instance_dict[customer_node][5] - route_start_time_calculator(Vehicle_route_set[route_set_key][route_list]) and (distance[depot, -2] + distance[depot,customer_node])*D_v/speed_v > nodes_instance_dict[customer_node][5] - route_start_time_calculator(Vehicle_route_set[route_set_key][route_list]):
					out_sourced_customer.append(customer_node)
					indicator = 1
					print customer_node
					print (distance[depot, -1] + distance[depot,customer_node])*D_v/speed_v, ">", nodes_instance_dict[customer_node][5] - route_start_time_calculator(Vehicle_route_set[route_set_key][route_list])
					print (distance[depot, -2] + distance[depot,customer_node])*D_v/speed_v, ">", nodes_instance_dict[customer_node][5] - route_start_time_calculator(Vehicle_route_set[route_set_key][route_list])
					break


def route_feasible_checker(route):
	# check inserted customer ET_i and route`s depot visit time
	"""
	만족 조건
	1. 삽입 노드의 생성 시간보다 라우트의 창고 방문 시간이 더 빨라야 한다.
	2. 삽입 노드로 인해 기존 노드들의 시간 제한이 위배되지 않아야 한다.
	3. 각 시간대 라우트가 제 시간에 종료되어야 한다.
	"""
	start_time = route_start_time_calculator(route)
	customer_list = customer_list_in_route_provider(route)
	for customer in customer_list:
		if nodes_instance_dict[customer][4] == start_time:
			#print "접수 전 배달 고객  ", customer
			return 0
		if target_customer_served_time_calculator(route , customer) > nodes_instance_dict[customer][5]:
			#print "종료 시점을 넘긴 배달 고객 ", customer
			return 0
	if route_end_time_calculator(route) > route_end_time_calculator_for_1(route):
		#print "종료시간 초과 "
		return 0
	return 1


#2-1) define needed function

def route_start_time_calculator(route):
	start_time = 0
	if route[3] == 1:
		start_time = 0
	if route[3] == 2:
		start_time = 280
	if route[3] == 3:
		start_time = 420
	if route[3] == 4:
		start_time = 560
	return start_time

def route_end_time_calculator_for_1(route):
	end_time = 0
	if route[3] == 1:
		end_time = 260
	if route[3] == 2:
		end_time = 400
	if route[3] == 3:
		end_time = 540
	if route[3] == 4:
		end_time = 840
	return end_time

def route_end_time_calculator(route):   # input: route
	route_end_time = route_start_time_calculator(route)
	for node in range(4,len(route)-1):
		route_end_time = route_end_time + round(time_v[(int(route[node]),int(route[node + 1]))],4)
	return route_end_time  # output: route end(finished) time


def customer_list_in_route_provider(route):     # input: route
	customer_in_route = []
	for node in route[5:]:
			if int(node) > 0:
				customer_in_route.append(node)
	return customer_in_route    #output: list of customer in the inputted route

def route_distance_calculator(route):    # input: route
	route_distance = 0
	for node_index in range(4,len(route)-1):
		route_distance = route_distance + round(distance[(int(route[node_index]),int(route[node_index + 1]))],4)
	if route_distance == 0:
		print 'route_distance is zero'
		print route
	return route_distance   # output:total distance of inputted route

def swap_feasibility_checker(route1, customer1, route2, customer2):
	if route1[3] + route2[3] == 0:
		if distance[(customer2, route1[4])] > M or distance[(customer1, route2[4])] > M:
			return 0
		endtime1 = route1[2] - 2*time_d[(customer1, route1[4])] + 2*time_d[(customer2, route1[4])]
		endtime2 = route2[2] - 2*time_d[(customer2, route2[4])] + 2*time_d[(customer1, route2[4])]
		if endtime1 > Max_time or endtime2 > Max_time:
			return 0
		return 1
	if route1[3] + route2[3] == 1:
		if customer1 in Vehicle_customer_list or customer2 in Vehicle_customer_list:
			return 0
		if route1[3] == 1:
			v_route = copy.deepcopy(route1)
			d_route = copy.deepcopy(route2)
			v_customer = customer1
			d_customer = customer2
		if route1[3] == 0:
			v_route = copy.deepcopy(route2)
			d_route = copy.deepcopy(route1)
			v_customer = customer2
			d_customer = customer1			
		v_weight = v_route[2][1] - int(data2[v_customer][3]) + int(data2[d_customer][3])
		v_index = int(v_route[6:].index(v_customer)) + 6
		endtime1 = v_route[2] - time_v[(v_customer, v_route[v_index - 1])] - time_v[(v_customer, v_route[v_index + 1])] + time_v[(d_customer, v_route[v_index - 1])] + time_v[(d_customer, v_route[v_index + 1])]
		endtime2 = d_route[2] - 2*time_d[(d_route[4],d_customer)] + 2*time_d[(d_route[4],v_customer)]
		if v_weight > Q_k or distance[(d_route[4],v_customer)] > M or endtime1 > Max_time or endtime2 > Max_time:
			return 0
		return 1
	if route1[3] + route2[3] == 2:
		weight1 = route1[2][1] - int(nodes_instance_dict[customer1][3]) + int(nodes_instance_dict[customer2][3])
		weight2 = route2[2][1] - int(nodes_instance_dict[customer2][3]) + int(nodes_instance_dict[customer1][3])
		index1 = int(route1[6:].index(customer1)) + 6
		index2 = int(route2[6:].index(customer2)) + 6
		endtime1 = route1[2][0] - time_v[(customer1, route1[index1 - 1])] - time_v[(customer1, route1[index1 + 1])] + time_v1[(customer2, route1[index1 - 1])] + time_v1[(customer2, route1[index1 + 1])]
		endtime2 = route2[2][0] - time_v[(customer2, route2[index2 - 1])] - time_v[(customer2, route2[index2 + 1])] + time_v1[(customer1, route2[index2 - 1])] + time_v1[(customer1, route2[index2 + 1])]
		if weight1 > Q_k or weight2 > Q_k or endtime1 > Max_time or endtime2 > Max_time:
			return 0
		return 1

def insert_feasibility_checker(route, inserted_node_list, inserted_position):
	inserted_position = int(inserted_position)
	vehicle = []
	drone = []
	if route[3] == 0:
		added_time = 0
		for inserted_customer in inserted_node_list:
			if customer in Vehicle_customer_list or distance[customer,route[4]] > M:
				return 0
			added_time = added_time + 2*time_d[customer,route[4]] + 2*Drone_ready_time
		if route[2] + added_time > Max_time:
			return 0
		return 1
	if route[3] == 1:
		added_time = len(inserted_node_list)*Vehicle_ready_time
		added_weight = 0
		for inserted_customer_index in range(0,len(inserted_node_list)):
			added_time = added_time + time_v[inserted_node_list[inserted_customer_index],route[4]] 
			added_weight = added_weight + int(nodes_instance_dict[inserted_node_list[inserted_customer_index]][3])
		route_end_time = route[2][0] - time_v[route[inserted_position - 1],route[inserted_position]] + added_time + time_v[route[inserted_position - 1],inserted_node_list[0]] + time_v[inserted_node_list[-1],route[inserted_position]]
		if route_end_time > Max_time or route[2][1] + added_weight > Q_k:
			return 0
		return 1


def count_served_customer_calculator(route): # input: route
	return len(customer_list_in_route_provider(route)) # output: the number of customer in the inputted route

def route_cost_calculator(route): # input: route
	return route_distance_calculator(route)*D_v*C_v

def target_customer_served_time_calculator(route, target_customer):  # input: route and a customer node in the route
	customer_list = customer_list_in_route_provider(route)
	#start_time = route_start_time_calculator(route)
	if target_customer in customer_list:
		if int(route[3]) >= 1:
			reroute = route[0:6]
			for customer in route[6:]:
				reroute.append(customer)
				if customer == target_customer:
					return round(route_end_time_calculator(reroute),4)  # output: arrive time to the customer
					break

def route_head_provider(route): # input: route
	if route[3] == 1:
		return route[:6]
	else:
		print 'head_customer_divider/ head is none vehicle or drone'
		print route
		return 'none'

def sol_set_info_provider(sol_set):
	sol_info = []
	for sol_key in sol_set.keys():
		if sol_key > 0:
			route_cost = []
			time_cost = []
			drone_number = 0
			vehicle_number = 0
			idletime_cost = []
			for route_key in sol_set[sol_key].keys():
				if route_key > 0:
					route_cost.append(round(route_cost_calculator(sol_set[sol_key][route_key]),4))
					time_cost.append(round(time_cost_calculator(sol_set[sol_key][route_key]),4))
					if sol_set[sol_key][route_key][3] == 0:
						drone_number = drone_number + 1
					if sol_set[sol_key][route_key][3] == 1:
						vehicle_number =  vehicle_number + 1
					idletime_cost.append((Max_time - route_end_time_calculator(sol_set[sol_key][route_key]))*non_profit_cost)
		sol_info.append([i, round(sum(route_cost),4),round(sum(time_cost),4),round(sum(idletime_cost),4),vehicle_number,drone_number,test_index])
	return sol_info

def route_costruct_by_head_customer_list_provider(head_, customer_list_): # input: route, customer list
	route = copy.deepcopy(head_)
	for customer in customer_list_:
		route.append(customer)
	route.append(route[4])
	return route #output: a route service inputted customer list with inputted head information

def find_good_route(sol): # input: a solution
	fitness_list = []
	for time_slot_key in sol.keys():
		for route in range(0,len(sol[time_slot_key])):
			fitness_list.append([sol[time_slot_key][route][0],route,time_slot_key])
	fitness_list.sort(key = operator.itemgetter(0))
	return fitness_list[0][1:] #output: a route`s index which has maximum fitness in the routes of inputted solution

def sol_cost_calculator(sol):   # input: a solution
	for time_slot_key in sol.keys():
		for route_index in range(0,len(sol[time_slot_key])):
			route_cost = route_cost_calculator(sol[time_slot_key][route_index])
			idle_cost = (route_end_time_calculator_for_1(sol[time_slot_key][route_index]) - route_end_time_calculator(sol[time_slot_key][route_index]))*non_profit_cost
			sol[time_slot_key][route_index][1] = route_cost + idle_cost
	return sol

def sol_set_cost_calculator(sol_set):  # input: Solution_set
	for sol_key in sol_set.keys():
		for time_slot_key in sol.keys():
			for route_index in range(0,len(sol_set[sol_key][time_slot_key])):
				route_cost = route_cost_calculator(sol_set[sol_key][time_slot_key][route_index])
				idle_cost = (route_end_time_calculator_for_1(sol_set[sol_key][time_slot_key][route_index]) - route_end_time_calculator(sol_set[sol_key][time_slot_key][route_index]))*non_profit_cost
				sol_set[sol_key][time_slot_key][route_index][1] = route_cost + idle_cost
	return sol_set

def sol_fitness_allocate_provider(sol): # input: a solution
	for time_slot_key in sol.keys():
		for route_index in range(0,len(sol[time_slot_key])):
			sol[time_slot_key][route_index][0] = sol[time_slot_key][route_index][1]/count_served_customer_calculator(sol[time_slot_key][route_index])
	return sol

def sol_set_fitness_allocate_provider(sol_set): # input: Solution_set
	for sol_key in sol_set.keys():
		for time_slot_key in sol.keys():
			for route_index in range(0,len(sol[time_slot_key])):
				sol[time_slot_key][route_index][0] = sol[time_slot_key][route_index][1]/count_served_customer_calculator(sol[time_slot_key][route_index])
	return sol_set 

def sol_set_empty_route_deleter(sol_set): # input: Solution_set
	for sol_key in sol_set.keys():
		for time_slot_key in sol.keys():
			for route_index in range(0, len(sol[time_slot_key])):
				if count_served_customer_calculator(sol[route_key]) == 0:
					del sol[time_slot_key][route_index]
	return sol_set

def sol_empty_route_deleter(sol): # input: a solution
	for time_slot_key in sol.keys():
		for route_index in range(0, len(sol[time_slot_key])):
			if count_served_customer_calculator(sol[time_slot_key][route_index]) == 0:
				del sol[time_slot_key][route_index]
	return sol

def select_random_2_sol_key_list_provider(sol_set): # input: Solution set
	key_list = sol_set.keys()
	random.shuffle(key_list)
	pair_list = []
	for pair_index in range(0,len(key_list)):
		if pair_index % 2 == 0:
			pair_list.append([key_list[pair_index],key_list[pair_index + 1]])
	return pair_list # output: random pair list [sol1_index, sol2_index]

def all_customer_checker(sol_set):
	for sol_key in sol_set.keys():
		customer_list = []
		for route_key in sol_set[sol_key].keys():
			if int(route_key) > 0:
				for node in sol_set[route_key][route_key][5:]:
					if int(node) > 0:
						customer_list.append(node)
		if len(customer_list) != instance_customer_size or int(min(customer_list)) != 1 or int(max(customer_list)) != instance_customer_size:
			print "this is sol" ,sol_key, "length:",len(customer_list), "min:",int(min(customer_list)), "max:", int(max(customer_list))

################Generation###############################

def generation2(sol_set):   # input: Solution_set
	sol_set_pair_list = select_random_2_sol_key_list_provider(sol_set)
	for sol_pair in range(0,len(sol_set_pair_list)):
		sol1_index = sol_set_pair_list[sol_pair][0]
		sol2_index = sol_set_pair_list[sol_pair][1]
		sol1 = copy.deepcopy(sol_set[sol1_index])
		sol2 = copy.deepcopy(sol_set[sol2_index])
		sol1_good_route = copy.deepcopy(sol1[find_good_route(sol1)[1]][find_good_route(sol1)[0]])
		sol2_good_route = copy.deepcopy(sol2[find_good_route(sol2)[1]][find_good_route(sol2)[0]])
		sol1_route_customer = customer_list_in_route_provider(sol1_good_route)
		sol2_route_customer = customer_list_in_route_provider(sol2_good_route)
		for time_slot_key in sol2.keys():
			for route in sol2[time_slot_key]:
				head = copy.deepcopy(route[:6])
				customer_list = customer_list_in_route_provider(route)
				for customer_index in range(0,len(customer_list)):
					if int(customer_list[customer_index]) in sol1_route_customer:
						customer_list[customer_list.index(customer_list[customer_index])] = -1
				reroute_customer_list = []
				for handled_customer in customer_list:
					if int(handled_customer) > 0:
						reroute_customer_list.append(handled_customer)	
				route = route_costruct_by_head_customer_list_provider(head,reroute_customer_list)
		for time_slot_key in sol1.keys():
			for route in sol1[time_slot_key]:
				head = copy.deepcopy(route[:6])
				customer_list = customer_list_in_route_provider(route)
				for customer_index in range(0,len(customer_list)):
					if int(customer_list[customer_index]) in sol2_route_customer:
						customer_list[customer_list.index(customer_list[customer_index])] = -1
				reroute_customer_list = []
				for handled_customer in customer_list:
					if int(handled_customer) > 0:
						reroute_customer_list.append(handled_customer)	
				route = route_costruct_by_head_customer_list_provider(head,reroute_customer_list)
		sol1[sol2_good_route[1]].append(sol2[sol2_good_route[1]][sol2_good_route[0]])		
		sol2[sol1_good_route[1]].append(sol1[sol1_good_route[1]][sol1_good_route[0]])
		sol_set[sol1_index] = copy.deepcopy(sol1)
		sol_set[sol2_index] = copy.deepcopy(sol2)
		sol_empty_route_deleter(sol_set[sol1_index])
		sol_empty_route_deleter(sol_set[sol2_index])
		sol_cost_calculator(sol_set[sol1_index])
		sol_cost_calculator(sol_set[sol2_index])
	return 'done' #output: generation processed Solution_set *note that it isn`t return any value.


#operator part
global Solution_set
Solution_set = {}
read_instance_file(user_name_for_directory,data_name)
print "done1"
for sol_set_key in range(1,instance_customer_size + 1):
	print 'sol_set_key', sol_set_key
	Solution_set[sol_set_key] = initial_route_maker(nodes_instance_dict)
print "done2"

global len_list
len_list = {}

for sol in Solution_set.keys():
	len_list[sol] = []
	for route_key in Solution_set[sol].keys():
		for route in Solution_set[sol][route_key]:
			if feasible_checker1(route) == 0:
				print "해",sol,"라우트 ", route, "불가능 해 "
			for customer in route[6:]:
				if customer > 0:
					len_list[sol].append(customer)
	if len(len_list[sol]) != 100:
		print "set 없이 ",sol, "에는",100 - len(len_list[sol]),"명의 고객이 없다."
	if len(set(len_list[sol])) != 100:
		print sol, "에는",100 - len(set(len_list[sol])),"명의 고객이 없다."


raw_input('oh!')
generation2(Solution_set)
