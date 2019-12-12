#-*- coding: utf-8 -*-
import random
import time
import simpy
import math
import operator
import numpy as np
import copy
import matplotlib.pyplot as plt
import itertools
from itertools import tee
import getpass
import os
import openpyxl
import datetime
from sklearn.cluster import KMeans
"""
고객을 생성하는 부분
INPUT : 시간(혹은 주어진 조건)에 따른 고객의 lamda
PROCESS : 포아송 분포에 따라 고객 발생 interval을 생성, 고객의 위치 생성
OUTPUT : 고객 발생 interval / 고객 위치
"""
class Customer_Gen(object):
	def __init__(self, env, name, input_location = None):
		self.name = name #각 고객에게 unique한 이름을 부여할 수 있어야 함. dict의 key와 같이 
		self.time_info = [env.now , None , None , None , None, None] #[발생시간, 차량에 할당 시간, 차량에 실린 시간, 목적지 도착 시간, 고객이 받은 시간]
		self.E_time_info = [env.now , None , None , None , None, None] #request로 할당되는 시점에서의 예상치 시간 정보
		if name == 0:
			self.location = [25,25] #차고지는 [25,25]
		elif input_location != None:
			self.location = input_location
		else:
			self.location = [random.choice(np.arange(0,50,0.01)) ,random.choice(np.arange(0,50,0.01)) ] #현재 고객의 생성은 난수에 의함.
		self.assigned = False
		self.loaded = False
		self.done = False
		self.weight = random.randrange(1,7) #혹은 배송의 특성을 할당할 수도 있음.
		self.service_time = random.randrange(1,7) #혹은 배송의 특성을 할당할 수도 있음.
def nextTime(lamda):
	#lamda should be float type.
	#lamda = input rate per unit time
	return -math.log(1.0 - random.random()) / lamda
def initial_Customer_Generator(env):
	num = 1
	for i in range(0,100):
		#print ("GO")
		c = Customer_Gen(env, num)
		customer_list.append(c)
		num += 1
def Cusotmer_Generator(env, lamda_list):
	num = len(customer_list)
	while True :
		#print ("customer")
		c = Customer_Gen(env, num)
		customer_list.append(c)
		#할당된 시간에 맞추어 lamda가 변경되는 경우에
		current_time_slot = int(env.now // 60)
		if current_time_slot > 24 or env.now > 1440:
			current_time_slot = int(current_time_slot - 24*(env.now//1440)) #하루는 24시간이니까.
		interval = 0
		if len(lamda_list) <= current_time_slot:
			print(lamda_list)
			print ("error", len(lamda_list), current_time_slot)
		if type(lamda_list[current_time_slot]) == int or float:
			interval = nextTime(60) * 60
		else:
			interval = nextTime(float(lamda_list[current_time_slot])) * 60 #분으로 변환
		yield env.timeout(interval) # 고객의 도착 간격 분포에서 무작위로 뽑힌 t 값이 지난 후에 작동
		num += 1
"""
주어지는 비용에 따라 lamda가 바뀌는 쿠팡 플렉스 생성 점
INPUT : 시스템이 지불하고자 하는 비용(즉 수수료)에 따른 고객의 lamda
PROCESS : 포아송 분포에 따라 고객 발생 interval을 생성, 시작점 생성, 숙력도(혹은 에러율 생성)
OUTPUT : 쿠팡플랙스 발생 interval / 시작 위치/ 에러율 
"""
CoupangFlex1 = []
class CoupangFlex(object):
	def __init__(self, env, name):
		self.CP = simpy.Resource(env, capacity = 1)
		self.location = [random.choice(np.arange(0,50,0.01)) ,random.choice(np.arange(0,50,0.01)) ]
		self.error_late = 1 + round(random.random(),4)
		self.Done = False
		self.withdraw = False
		self.name = random.random()
	def CP_Done(self, env, customer):
		self.Done = True
		customer.time_info[1] = env.now
		with self.CP.request() as req:
			#고객의 짐을 싣기 위해 차고지로 이동
			#print ("THIS CP DONE", env.now)
			customer.assigned = True
			yield req
			moving_duration1 = (distance(self.location, depot.location)/speed)*self.error_late
			yield env.timeout(moving_duration1)
			customer.time_info[2] = env.now
			customer.loaded = True
			#짐을 싣고 고객에게 이동
			moving_duration2 = (distance(depot.location, customer.location)/speed)*self.error_late
			yield env.timeout(moving_duration2)
			customer.time_info[3] = env.now
			#도착 후 고객에게 서비스
			service_time = customer.service_time*self.error_late #CP의 숙련도에 따라 서비스 시간 차이 생김
			yield env.timeout(service_time)
			customer.time_info[4] = env.now
			customer.done = True
	def CP_withdraw(self, env, et,lt):
		withdraw_time = random.choice(np.arange(et,lt,0.5))
		yield env.timeout(withdraw_time)
		if self.Done == False and len(self.CP.users) == 0: #아직도 할당되지 않은 경우
			self.withdraw = True
def IntervalCP(expected_fee):
	while True:
		name = 0
		ob = CoupangFlex(env, name)
		env.process(ob.CP_withdraw(env,20,40)) # env, 철회 최소 시간, 철회 최대 시간
		#print ("ob",ob)
		CoupangFlex1.append(ob)
		#expected fee는 1000원부터 12000까지의 값을 가질 수 있다.)
		x = round( expected_fee / 1000,2)
		#lamda = 1/(round(math.exp(6 - x),2)+1)# expected_fee를 받아서, lamda를 도출하는 식이 들어가야 함.
		#yield env.timeout(nextTime(lamda))
		lamda = random.randrange(7,15) #실험의 단순화를 위해 우선 난수로 생성.
		yield env.timeout(lamda)
		name += 1
class mock_driver(object):
	def __init__(self, env, name):
		self.name = name
		self.route = [] #[[유형,이름,위치,도착시간,출발시간,종료시점,상태유형],...,] 유형 : 0(창고)/1(고객) ; 상태유형 : 0 서비스 X /1 : 서비스 됨
class driver(object):
	def __init__(self, env, name):
		self.name = name
		self.route = [] #[[유형,이름,위치,도착시간,출발시간,종료시점,상태유형],...,] 유형 : 0(창고)/1(고객) ; 상태유형 : 0 서비스 X /1 : 서비스 됨
		self.done = []
		self.call_back_time = []
		self.etc = []
		self.veh = simpy.Resource(env, capacity = 1)
		self.load = 0
		self.last_location = [50,50]
		self.served = []
		self.end_time = []
		self.now_return = [[False, 0 ]] #	[현재 회차 중인지 여부, 차량의 마지막 회차 시간]
		self.released_ct = []
		self.queue_len = [0]
	def Driving(self, env, customer, next_coord = None, fake_parameter = False):
		if fake_parameter == True:
			print("Fake in Veh # ", self.name ,"/Ct name :",customer.name,"/T :", env.now,)
		customer.loaded = True
		customer.time_info[1] = env.now
		customer.time_info[2] = env.now
		customer.assigned = self.name
		with self.veh.request() as req:
			#print ("Veh#", self.name,"CT", customer.name , "Start", env.now)
			self.loaded = True
			#customer.assigned = self.name #할당되자 마자 
			self.load += customer.weight
			req.info = customer
			yield req #users에 들어간 이후에 작동
			moving_duration = distance(self.last_location, customer.location)/speed
			yield env.timeout(moving_duration)
			customer.time_info[3] = env.now
			service_time = customer.service_time
			yield env.timeout(service_time)
			customer.time_info[4] = env.now
			self.load -= customer.weight
			self.served.append(customer.name)
			self.last_location = customer.location
			customer.done = True
			#print ("Veh#", self.name,"CT", customer.name , "End", env.now)
			#print("Veh#",customer.name ,customer.loaded ,customer.done)
			if len(self.veh.put_queue) == 0:
				print("Empty queue", self.name, "At", env.now, len(self.veh.put_queue), customer.name)
				current_location = WhereAmI(self.last_location, customer.location, customer.time_info[1])
				env.process(self.return2point(env,current_location))
				#me_to_depot = self.return2point(env,current_location)
				#yield env.timeout(me_to_depot)
				self.last_location = depot.location
				#self.ETE.append(env.now)
			self.end_time.append(env.now)
	def return2point(self, env, current_location): #현재 운행을 정지하고 point로 돌아가는 함수
		if len(self.veh.users) > 0:
			#기존에 실려 있던 고객들의 주문을 다시 처리가 안된 고객으로 돌린다.
			if len(self.veh.put_queue) > 0:
				for ct in self.veh.put_queue:
					self.released_ct.append(ct.info.name)
					for time_index in range(1,len(customer_list[ct.info.name].time_info)):
						customer_list[ct.info.name].time_info[time_index] = None
				self.veh.put_queue = []
			#현재 향하고 있는 고객도 release시킴.
			self.released_ct.append(self.veh.users[0].info)
			self.veh.release(self.veh.users[0])
			with self.veh.request() as req:
				yield req #users에 들어간 이후에 작동
				moving_duration = distance(current_location, depot.location)/speed
				#정확히는 현재의 위치에서 부터 차고지로 돌아가는 역할을 해야 함.
				yield env.timeout(moving_duration)
		else:
			pass #종료 시킬 것.
	def Re_routing(self, env, un_assigned_customer_list, customer_infos):
		un_assigned_customer_list.append(self.released_ct)
		re_ct = []
		for ct_name in un_assigned_customer_list:
			re_ct.append(customer_infos[ct_name])
		location_list = []
		for ct_name in re_ct:
			location_list.append(customer_infos[ct_name].location)
		route = NN_Algo(location_list, re_ct, start = True)
		for ct_name in route:
			env.process(self.Driving(env, customer_infos[ct_name]))		
def WhereAmI(org_start, org_end, departure_time):
	#시작점과 출발점 사이의 함수 구하기
	org_end_time = departure_time + distance(org_start, org_end)/speed
	if org_end_time == 0:
		print ("org end time == 0",org_start, org_end, departure_time)
		#x_delta = abs(org_end[0] - org_start[0])*(env.now/org_end_time)
		#y_delta = abs(org_end[1] - org_start[1])*(env.now/org_end_time)
		x_delta = 25
		y_delta = 25
	else:
		x_delta = abs(org_end[0] - org_start[0])*(env.now/org_end_time)
		y_delta = abs(org_end[1] - org_start[1])*(env.now/org_end_time)
	return [min(org_start[0], org_end[0])+x_delta , min(org_start[1], org_end[1])+ y_delta]
"""
상황에 따라 고객을 쿠팡플렉스로 전담하고자 하는 시스템
INPUT : 시스템이 지불하고자 하는 비용(즉 수수료)에 따른 고객의 lamda
PROCESS : 포아송 분포에 따라 고객 발생 interval을 생성, 시작점 생성, 숙력도(혹은 에러율 생성)
OUTPUT : 쿠팡플랙스 발생 interval / 시작 위치/ 에러율 
"""
def clustered_coord(veh_set, customer_infos, given_ct_set = None):
	#kmeans로 초기 라우트 설정
	#클러스터링 하고
	print ("CLUSTER", len(customer_list))
	node_coor = []
	if given_ct_set == None:
		for ct in customer_infos:
			node_coor.append(ct.location)
	else:
		for ct in given_ct_set:
			node_coor.append(customer_infos[ct].location)
	node_coor = np.asarray(node_coor)
	kmeans = KMeans(n_clusters = len(veh_set), random_state = 0).fit(node_coor)
	cluster_assigned = [[] for _ in range(len(set(kmeans.labels_)))]
	index = 0
	for clt in kmeans.labels_:
		if given_ct_set == None:
			cluster_assigned[clt].append(customer_infos[index])
		else:
			cluster_assigned[clt].append(customer_infos[index])
		index += 1
	return cluster_assigned
def distance(x1,x2):
	return round(math.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2),4)
def NN_Algo(points, original_info, start = None):
	"""
	optimized_travelling_salesman2(...)
	optimized_travelling_salesman2(points, original_info, start=None) -> customer visit sequence list 
	Return customer visit sequence list
	Notice! 
	This function use Nearest Neighborhood algorithm.
	"""
	#라우트는 nearest neighborhood -> 변경 가능 유전 알고리즘이나 뭐 etc...
	if start is None:
		start = points[0]
	else:
		points.insert(0, depot.location)
		start = depot.location
	must_visit = points
	path = [start]
	#print (points, path)
	must_visit.remove(start)
	while must_visit:
		nearest = min(must_visit, key=lambda x: distance(path[-1], x))
		path.append(nearest)
		must_visit.remove(nearest)
	full_path = []
	#print ("LEN PATH", len(path))
	for node in path:
		for full_info in original_info:
			if node == full_info.location and full_info.name not in full_path :
				full_path.append(full_info.name)
				break
	#full_path.append(original_info[0]) # this is 
	return full_path # full path 는 고객들의 이름 임.
def call_back_feasibility(vehicle, customer_infos, policy_t = 120):
	feasible_check_list = []
	if len(vehicle.veh.users) == 0:
		return True
	else:
		#print ("CHEK",vehicle.last_location, depot.location)
		added_time = 1.7*distance(vehicle.last_location, depot.location)/speed 
		# 1.7은 일정의 거리 상수 만약 이를 2로 하면 상당히 보수적인 기준. 
		# 즉 차고지를 재방문하는 경우에 증가하는 시간에 대한 기준 임.
		arrival_time = env.now
		for ct in vehicle.veh.users + vehicle.veh.put_queue:
			#print ("CHECK",ct.info.location)
			arrival_time += distance(vehicle.last_location, ct.info.location)/speed
			arrival_time += ct.info.service_time
			if ct.info.time_info[0] + policy_t >= arrival_time:
				#배송 최대 시간 >= 기존 예상 배송 시간 + 예상 증가 시간
				pass
			else:
				return False
		return True
#복귀시킬 차량 선택 
def AllNodeInsertCostCal(av_veh_index_set, veh_set, customer_infos):
	if len(av_veh_index_set) == 0 :
		print ("No available veh/ Current time : ", env.now)
		pass
	else:
		insert_info = []
		for ct in customer_infos:
			if ct.assigned == False:
				cal = []
				for veh_index in av_veh_index_set:
					#print ("CAL", veh_set[veh_index], veh_set[veh_index].veh.users + veh_set[veh_index].veh.put_queue)
					route = [veh_set[veh_index]]
					if len(route) > 0:
						#index, cost = insertion_cost(route ,ct)
						route_index, index, cost = route_insert_cost(ct, route ,sol_para = False)
						cal.append([veh_index, ct.name ,index,cost]) #차량 번호 ; 삽입 고객 이름 ;삽입되는 위치; 삽입 비용
				if len(cal)> 0:	
					cal.sort(key = operator.itemgetter(3))  
					insert_info.append(cal[0])
		return insert_info #회차 가능 차량들에 대해, 최소 비용 삽입 위치 계산.
def idle_veh_run(veh_set, customer_infos):
	while True:
		idle_name = []
		for veh in veh_set:
			if len(veh.veh.users) + len(veh.veh.put_queue) == 0: #idle한 차량
				idle_name.append(veh.name)
		if len(idle_name) > 0:
			print( "IDLE VEH", idle_name, "NOW : ", env.now)
			unas_cts = []
			unas_cts_eles = []
			for ct in customer_infos:
				if ct.assigned == False and ct.done == False:
					unas_cts.append(ct.name)
					unas_cts_eles.append(ct.location)
			route = NN_Algo(unas_cts_eles, customer_infos, start = None)
			#라우트의 feasible_check를 하고, 가능해만 발생 시킬 것.
			arrival_time = env.now
			idle_count = 0
			for ct_name in route:
				#예상 도착 시간을 계산 후 불가능한 고객은 라우트에서 제거
				add_time = customer_infos[ct_name].service_time
				if route.index(ct_name) == 0:
					add_time += distance(depot.location , customer_infos[ct_name].location)/speed
				elif route.index(ct_name) == len(route) - 1:
					add_time += distance(customer_infos[ct_name].location, depot.location)/speed
				else:
					add_time += distance(customer_infos[ct_name].location, depot.location)/speed
				#만족하는 경우에만 수행함
				if arrival_time + add_time <= customer_infos[ct_name].time_info[0] + policy_t:
					arrival_time += add_time
					env.process(veh_set[idle_name[0]].Driving(env, customer_infos[ct_name]))
					idle_count += 1
					#print ("ASSIGN")
				else: #시간의 초과로 배송 할 수 없는 고객임.
					customer_infos[ct_name].assigned = False
					pass
			print("Now ",env.now ,"/Idle done : ", idle_count)
		yield env.timeout(10)

def customer_re_assigner2(veh_set ,customer_infos, insert_infos, rev_beta, assign_type = 'all', min_return_time = 60):
	#수정 중
	#threshold_1 : 고객 삽입 시 고려 하는 기준 (이것 보다 큰 경우 삽입 취소)
	#insert info 내용. : 차량 번호 ; 삽입 고객 이름 ;삽입되는 위치; 삽입 비용
	#차량 회차 의사결정시점까지 발생한 고객 
	print("customer_re_assigner start")
	#1 회차 기준을 만족하는 차량을 선택
	CallBackAva = []
	CallBackAvaName = []
	already_loaded = []
	for veh in veh_set:
		if veh.now_return[-1][1] < env.now - min_return_time: # 기준보다 더 이전에 회차 했어야함.
			CallBackAva.append(veh)
			CallBackAvaName.append(veh.name)
			for ct in veh.veh.users + veh.veh.put_queue:
				already_loaded.append(ct.info.name)
	#할당 방식 결정
	unloaded_ct_name = []
	for ct_info in insert_infos:
		if assign_type == 'all':
			#아직 할당이 안된 모든 고객을 고려
			unloaded_ct_name.append(ct_info[1])
		elif 'some':
			#아직 할당이 안된 모든 고객 중 현재 선택된 차량에 삽입 되려던 고객만 선택
			if ct_info[0] in CallBackAvaName and ct_info[0] == veh.name:
				unloaded_ct_name.append(ct_info[1])
		else:
			print("assign type is not setted")
	#여러 경우 중 이미 차량에 실리거나 CP에 할당되거나, 제거된 고객은 제거.
	print("need more job1")
	unloaded_ct_name2 = []
	for u_ct in unloaded_ct_name:
		if customer_infos[u_ct].time_info[0] > env.now - 120: #최소 발생 후 2시간 이내 할당
			pass
		elif customer_infos[u_ct].assigned == True or customer_infos[u_ct].done == True: #고객이 어느 것에도 할당 되지 않음
			pass
		else:
			unloaded_ct_name2.append(u_ct)
	#C_old U C_new로 이루어진 초기해 방식을 사용하여, 구성
	test = clustered_coord(CallBackAva, customer_infos , given_ct_set =  already_loaded + unloaded_ct_name2)
	mock_routes = []
	route_index = 0
	for ele in test:
		eles = []
		mock_route = mock_driver(env, route_index)
		for node in ele:
			eles.append(node.location)
		route = NN_Algo(eles, customer_list, start = True)
		for ct_name in route:
			#고객들을 차량에 request로 할당해야 함.
			if ct_name == 0:
				pass
			else:
				mock_route.route.append([customer_list[ct_name].name])
		mock_routes.append(mock_route)
		route_index += 1
	#r`_k와 r_k비교
	hamming_score = []
	print("veh, mock test LEN", len(veh_set),len(mock_routes), len(test))
	for veh in veh_set:

		ham_score = hamming_distance(veh_set[veh.name], mock_routes[veh.name])
		hamming_score.append(ham_score)
	#max hamming score을 회차 시켜 고객을 할당.
	max_index = hamming_score.index(max(hamming_score))
	coord_list = []
	for ct_name in mock_routes[max_index].route:
		if type(ct_name) != int:
			print ("ERR", ct_name)
		coord_list.append(customer_infos[ct_name[0]].location)
	route_node_seq = NN_Algo(coord_list, customer_infos, start = True) #<- route는 고객의 이름 순서
	veh_set[max_index].veh.put_queue = []
	for ct_name in route_node_seq:
		env.process(veh_set[max_index].Driving(env, customer_infos[ct_name]))
	veh_set[max_index].now_return.append([True, env.now])
	yield env.timeout(1)
	print("customer_re_assginer end", env.now, "To veh #", veh.name, "; Current len :", len(veh.veh.users)+ len(veh.veh.put_queue))
	#return veh_set
def hamming_distance(route1, route2):
	#sol1 = 실제 운행 중; sol2 = 가상으로 구축된 고객.
	if route1.name != route2.name:
		print ("NOT SAME ROUTE ERROR")
		return None
	r1_list = []
	r2_list = []
	for node in route1.veh.users + route1.veh.put_queue:
		#if len(node.info.name.split(";")) > 1:
		if type(node.info.name) != int : # 가짜 노드를 구분하기 위한 작업.
			name_info = node.info.name.split(";")
			r1_list.append([int(name_info[1]), int(name_info[2])])
		else:
			r1_list.append([node.info.name])
	for node_name in route2.route: #route2 는 mock.route에 있는 ct name list e.g.[[11], [12], [13], [14], [19]]
		#if len(node.info.name.split(";")) > 1:
		if type(node_name[0]) != int : # 가짜 노드를 구분하기 위한 작업.
			name_info = node.info.name.split(";")
			r2_list.append([int(name_info[1]), int(name_info[2])])
		else:
			r2_list.append([node_name[0]])
	ham_score = 0
	node_index = 0
	print ("check len",len(r1_list), len(r2_list))
	#r1과 	r2의 길이에 상관 없이 작동하는 방식 짧은 리스트에서 긴 리스트를 확인하는 방법 #수정 필요
	sort = [r1_list,r2_list]
	sort.sort(key = len)
	short_r = sort[0]
	long_r = sort[1]
	for node_name in short_r: 
		if node_name == long_r[node_index]:
			ham_score += 1
		else:
			pass
		node_index += 1
	"""
	for node1 in r1_list: #r1과 	r2의 길이가 다름.
		if node1 == r2_list[node_index]:
			ham_score += 1
		else:
			pass
		node_index += 1
	"""
	return ham_score
def Algo_ver2(env, veh_set, customer_list, lamda_list, policy_t, stack_time, CPs, end_time = 288, lamda_matrix_size = 50):
	#input : env, veh_set, lamda_list, e_t, a_t, stack_time
	#process : E_Com과 A_Com을 작동 시킴.
	#output :동작 함수
	len_cp_list = 10
	CP_infos = [] #CP로 처리된 고객 정보 [[#고객 이름, #CP할당 시간],...,]
	rev_beta = 0
	while True and env.now < end_time:
		if env.now % 5 == 0:
			#print("Loop start", env.now)
			pass
		# E_Com부
		#1.1 단위시간 lamda_info 갱신
		if env.now % 30 == 0 and env.now > 0:
			duration = 30
			current_time_slot = int((env.now/30)/48) - 24*int((env.now/30)/48)
			ite = 10
			e_num_info = E_customer_generator(env, duration, lamda_list[current_time_slot], ite)
			# lamda_list[current_time_slot] <- 현재 시간대의 lamda_list
			#1.2 해당 lamda_info와 현재 고객에 대해 가상해 생성
			alpha = 0.95 ; beta = 0.5 ; max_ite_num = 50 ; max_imp_num = 10
			rev_beta, rev_sol, len_cp_list = E_sol_constructor2(env, veh_set, customer_list, e_num_info,  alpha,beta, len_cp_list ,max_ite_num, max_imp_num, insert_type = "OneByOne")
		# A_Com 부
		#일정 간격으로 lamda_list 갱신
		if env.now % 30 == 0: #<-lamda_list를 갱신하는 시간 간격
			tem_lamda_list = Customer_lamda_recorder(env, lamda_list, customer_list)
			lamda_list.append(tem_lamda_list)
		#실제 라우트 가동 부
		#stack_time = 이 시간 보다 적으면 CP에 할당하는 기준
		if env.now % 30 == 0: #<-회차를 확인하는 시간 간격
			#1:새롭게 생성된 고객들을 확인하는 부분
			unas_ct_set = []
			for ct in customer_list:
				if ct.assigned == False:
					unas_ct_set.append(ct)
			#2:회차가 가능한 차량을 선별
			av_veh_set = []
			for veh in veh_set:
				if call_back_feasibility(veh, customer_list, policy_t = policy_t) == True: #가능한 차량 
					av_veh_set.append(veh.name)
			#3:가능한 차량 중 어떤 차량을 선택할지 결정
			if len(av_veh_set) > 0: #가능한 차량이 존재하는 경우
				#특정 차량에 고객을 할당 or CP를 활용하여 할당할지를 결정.
				print("Now", env.now, "/ Ava veh #", len(av_veh_set))
				last_call_back_time = []
				for veh in veh_set:
					last_call_back_time.append(env.now - veh.now_return[-1][1])
				if env.now - min(last_call_back_time) < 100: #최근 회차가 너무 일찍 발생했음. 
					CP_infos = CP_assign(customer_list, CPs)
				#빠짐으로써 라우트 효율성을 증가시키는 고객도 빠져야 함.
				if env.now > 120: #회차 시작. 단 차량의 회차가 너무 자주 이루어 질 수는 없음.
					#CallBacker(av_veh_set, veh_set, policy_t, customer_list, rev_beta, CP_infos)
					CallBacker2(av_veh_set, veh_set, policy_t, customer_list, rev_beta, CP_infos)
		yield env.timeout(1)
def CP_assign(customer_infos, CPs):
	CP_infos = []
	unan_cts = []
	for customer in customer_infos:
		if customer.assigned == False and customer.loaded == False:
			unan_cts.append([customer, env.now - customer.time_info[0]])
	unan_cts.sort(key = operator.itemgetter(1), reverse = True) #오래된 고객 일수록 시급함.
	ava_CPs = []
	for CP in CPs:
		if CP.Done == False and CP.withdraw == False:
			ava_CPs.append(CP) #ava_CPs = CP에 할당 될 수 있는 고객.
	print("NOW", env.now ,"/ un_ct :" , len(unan_cts), "Ava CP", len(ava_CPs))
	ct_index = 0
	for CP in ava_CPs:
		if ct_index < ava_CPs:
			env.process(CP.CP_Done(env, unan_cts[ct_index][0]))
			print("CP : ", CP.name, "->", unan_cts[ct_index][0].name)
			CP_infos.append([unan_cts[ct_index][0].name,env.now])
		#else:
			#pass # 이미 할당 가능한 고객은 모두 할당함.
			#break
		ct_index += 1
	return CP_infos
def CallBacker2(av_veh_set, veh_set, policy_t, customer_list, rev_beta, CP_infos):
	insert_infos = AllNodeInsertCostCal(av_veh_set, veh_set, customer_list) #회차가 가능한 차량들에 대해 미할당된 고객들 삽입 예상 비용 계
	#inser_infos = [[veh_index, ct.name ,index, cost],...,]
	#가장 많은 수의 고객이 삽입 될 수 있는 차량에 삽입.
	count = []
	for veh in veh_set:
		count.append([veh.name, 0])
		veh.queue_len.append(len(veh.veh.users + veh.veh.put_queue))
	#차량 번호 ; 삽입 고객 이름 ;삽입되는 위치; 삽입 비용
	for info in insert_infos:
		count[info[0]][1] += 1
	count.sort(key = operator.itemgetter(1))
	veh_index = None
	#현재는 시간 기준을 만족하는 차량 아무것이나 바로 할당 -> 즉 수정이 필요함.
	for veh_info in count:
		if veh_set[veh_info[0]].now_return[-1][1] + 20 < env.now : #최소 한시간 전에는 회차 했어야 함.
			veh_index = veh_info[0]
			break
	#res = customer_re_assginer(veh_set[veh_index], customer_list, insert_infos, rev_beta, assign_type = 'all', min_return_time = 60)
	env.process(customer_re_assigner2(veh_set, customer_list, insert_infos, rev_beta, assign_type = 'some', min_return_time = 60))
	if env.now == 0: #Print route change
		for veh in veh_set:#print t = 0  
			print ("Veh:", veh.name , "/ Queue:", len(veh.veh.users + veh.veh.put_queue))
	else:
		for veh in veh_set:#if cutsomer is added in queue
			if veh.queue_len[-1] < len(veh.veh.users) + len(veh.veh.queue) :
				print ("Veh:", veh.name , "/ Queue:",veh.queue_len[-1],"->", len(veh.veh.users + veh.veh.put_queue),"/t=",env.now)
	#if res != None:
	if True == True:
		#회차된 차량에서 불 필요한 고객 제거 
		#CP_infos = 사용 가능한 쿠팡 플렉스 수? -> 제거 될 수 있는 고객의 수
		#len_cp_list : 가능한 CP수 (사용 가능한)
		len_cp_list = 3
		print("Insert test")
		Lreturned_times = []
		for veh in veh_set:
			Lreturned_times.append([veh.name] + veh.now_return[-1])
		Lreturned_times.sort(key= operator.itemgetter(1), reverse = True)
		veh_set[Lreturned_times[0][0]]
		veh_set[Lreturned_times[0][0]], CP_infos = route_customer_deleter(veh_set[Lreturned_times[0][0]], CP_infos ,len_cp_list, rev_beta)
#20190828 개선 부분
def E_sol_constructor2(env, current_sol, customer_infos, lamda_list, alpha,beta, CP_num ,max_ite_num, max_imp_num, insert_type = None):
	#beta와 |CP|의 크기가 결정 되어야 함.
	ite = 0 
	non_imp_num = 0
	E_loaded_customer = []
	lamda_index_list = []
	cp_list = []
	#print ("lamda_list", type(lamda_list), lamda_list[0][2], np.shape(lamda_list))
	size = np.shape(lamda_list)
	for i in range(0,size[0]):
		for j in range(0,size[1]):
			lamda_index_list.append([i,j])
	rev_sol = []
	for veh in veh_set:
		rev_sol.append(driver_duplicater2(env, veh))
		#print( "driver_ORG : ",veh.name ,"/ Len:",len(veh.veh.users) + len(veh.veh.put_queue) )
		#print( "driver_duplicate : ",rev_sol[-1][0] ,"/ Len:",len(rev_sol[-1][1]) )
	#Pr{lamda >= 1} >= alpha를 만족하는 E고객만을 선별
	E_customer_infos = []
	for index in lamda_index_list:
		lamda = lamda_list[index[0],index[1]]
		if lamda > alpha:
			f_num = str(env.now) + str(index)
			#가상의 고객 설정
			median_location = [index[0]*2.5 + 1.75,index[1]*2.5 + 1.75] # 50x50구역을 가로 새로 10개로 나눔.
			e_c = Customer_Gen(env,'Fake' + f_num, input_location = median_location)
			E_customer_infos.append(e_c) #임의의 고객을 생성 함.
	print("Now :", env.now, "/E_ct_num : ", len(E_customer_infos))
	prev_cost = CP_under_beta2(rev_sol, sol_type = "mock") #처음에는 연산이 수행되어야 하기 때문에 아주 큰수를 임의로 할당.
	check_ite = 0
	while ite < max_ite_num and non_imp_num < max_imp_num:
		if check_ite > 10:
			break
		if check_ite % 5 == 0:
			print ("check_ite", check_ite)
		check_ite += 1
		after_cost = 0
		if insert_type == "OneByOne":
		#한번에 삽입하는 방식
		#시간 지연에 대해 연산을 멈추게 하는 장치를 삽입.
			start_time = time.time()
			count_index = 0
			while time.time() - start_time < 180  and count_index < 1: #3분을 최대 연산시간으로 놈
				#print ("OneByOne /", env.now )
				f1_time = time.time()
				#print (" START route_insert_cost ")
				if len(E_customer_infos) > 0:
					print("T",env.now, "E_ct #", len(E_customer_infos))
					for e_c in E_customer_infos:
						route_index, insert_index, insert_cost = route_insert_cost2(e_c , rev_sol, sol_para = True)
						if insert_cost < beta:
							E_loaded_customer.append([route_index, insert_index, insert_cost ,e_c])
						else:
							cp_list.append(e_c)
					#print (" END route_insert_cost  ; ComT:" , str(time.time() - f1_time))
					E_loaded_customer.sort(key = operator.itemgetter(2))
					for insert_info in E_loaded_customer:
						rev_sol[insert_info[0]] = node_inserter2(env, rev_sol[insert_info[0]], insert_info[3], customer_infos, insert_info[1],E_loaded_customer)
						#여기서 처음 rev_sol에 직접 변화가 발생 <- insert
						pass
					E_loaded_customer = [] #E_loaded_customer를 한번 지워주어야 함. 다시 계산 되기 때문
				else:
					#print("Empty E_ct")
					pass
				count_index += 1
		elif insert_type == "Sequential":
		#순서대로 삽입하는 방식
			print ("Sequential")
			for _ in range(0,10): #이는 한 beta에서의 작동방식
				mock_sol = rev_sol #구조가 간단화 된 리스트 형식의 라우트 
				random.shuffle(E_customer_infos)
				mock_e_c = []
				mock_cp_list = []
				for e_c in E_customer_infos:
					route_index, insert_index, insert_cost = route_insert_cost2(e_c , rev_sol, sol_para = True)
					if insert_cost < beta:
						mock_e_c.append([route_index, insert_index, insert_cost ,e_c])
						rev_sol[insert_info[0]] = node_inserter2(env, rev_sol[insert_info[0]], insert_info[3], customer_infos, insert_info[1],E_loaded_customer)
					else:
						mock_cp_list.append(e_c)
				mock_after_cost = CP_under_beta2(mock_sol,sol_type = "mock")
				if prev_cost < mock_after_cost:
					pass
				else: #개선 발생
					rev_sol = mock_sol
					after_cost = mock_after_cost
					break
		else:
			print ("Select Insert Type")
			pass
		ite += 1
		after_cost = CP_under_beta2(rev_sol, sol_type = "mock")
		if prev_cost < after_cost:
			non_imp_num += 1
		else: #개선이 발생함.
			non_imp_num = 0
			beta = beta * 1.05
			CP_num = len(cp_list)
		prev_cost = after_cost # 해 전제 비용 갱신
	print("E_sol_constructor Done")
	return beta, rev_sol, CP_num
#20191209 개선 부분 ; 노트 필기 내용 대로 라우트 구성
def E_sol_constructor3(env, current_sol, customer_infos, lamda_list, alpha,beta, CP_num ,max_ite_num, max_imp_num, insert_type = None):
	#beta와 |CP|의 크기가 결정 되어야 함.
	ite = 0 
	non_imp_num = 0
	E_loaded_customer = []
	lamda_index_list = []
	cp_list = []
	#print ("lamda_list", type(lamda_list), lamda_list[0][2], np.shape(lamda_list))
	size = np.shape(lamda_list)
	for i in range(0,size[0]):
		for j in range(0,size[1]):
			lamda_index_list.append([i,j])
	rev_sol = []
	for veh in veh_set:
		rev_sol.append(driver_duplicater2(env, veh))
		#print( "driver_ORG : ",veh.name ,"/ Len:",len(veh.veh.users) + len(veh.veh.put_queue) )
		#print( "driver_duplicate : ",rev_sol[-1][0] ,"/ Len:",len(rev_sol[-1][1]) )
	#Pr{lamda >= 1} >= alpha를 만족하는 E고객만을 선별
	E_customer_infos = []
	for index in lamda_index_list:
		lamda = lamda_list[index[0],index[1]]
		if lamda > alpha:
			f_num = str(env.now) + str(index)
			#가상의 고객 설정
			median_location = [index[0]*2.5 + 1.75,index[1]*2.5 + 1.75] # 50x50구역을 가로 새로 10개로 나눔.
			e_c = Customer_Gen(env,'Fake' + f_num, input_location = median_location)
			E_customer_infos.append(e_c) #임의의 고객을 생성 함.
	print("Now :", env.now, "/E_ct_num : ", len(E_customer_infos))
	prev_cost = CP_under_beta2(rev_sol, sol_type = "mock") #처음에는 연산이 수행되어야 하기 때문에 아주 큰수를 임의로 할당.
	check_ite = 0
	while ite < max_ite_num and non_imp_num < max_imp_num:
		if check_ite > 10:
			break
		if check_ite % 5 == 0:
			print ("check_ite", check_ite)
		check_ite += 1
		after_cost = 0
		if insert_type == "OneByOne":
		#한번에 삽입하는 방식
		#시간 지연에 대해 연산을 멈추게 하는 장치를 삽입.
			start_time = time.time()
			count_index = 0
			while time.time() - start_time < 180  and count_index < 1: #3분을 최대 연산시간으로 놈
				#print ("OneByOne /", env.now )
				f1_time = time.time()
				#print (" START route_insert_cost ")
				if len(E_customer_infos) > 0:
					print("T",env.now, "E_ct #", len(E_customer_infos))
					for e_c in E_customer_infos:
						route_index, insert_index, insert_cost = route_insert_cost2(e_c , rev_sol, sol_para = True)
						if insert_cost < beta:
							E_loaded_customer.append([route_index, insert_index, insert_cost ,e_c])
						else:
							cp_list.append(e_c)
					#print (" END route_insert_cost  ; ComT:" , str(time.time() - f1_time))
					E_loaded_customer.sort(key = operator.itemgetter(2))
					for insert_info in E_loaded_customer:
						rev_sol[insert_info[0]] = node_inserter2(env, rev_sol[insert_info[0]], insert_info[3], customer_infos, insert_info[1],E_loaded_customer)
						#여기서 처음 rev_sol에 직접 변화가 발생 <- insert
						pass
					E_loaded_customer = [] #E_loaded_customer를 한번 지워주어야 함. 다시 계산 되기 때문
				else:
					#print("Empty E_ct")
					pass
				count_index += 1
		elif insert_type == "Sequential":
		#순서대로 삽입하는 방식
			print ("Sequential")
			for _ in range(0,10): #이는 한 beta에서의 작동방식
				mock_sol = rev_sol #구조가 간단화 된 리스트 형식의 라우트 
				random.shuffle(E_customer_infos)
				mock_e_c = []
				mock_cp_list = []
				for e_c in E_customer_infos:
					route_index, insert_index, insert_cost = route_insert_cost2(e_c , rev_sol, sol_para = True)
					if insert_cost < beta:
						mock_e_c.append([route_index, insert_index, insert_cost ,e_c])
						rev_sol[insert_info[0]] = node_inserter2(env, rev_sol[insert_info[0]], insert_info[3], customer_infos, insert_info[1],E_loaded_customer)
					else:
						mock_cp_list.append(e_c)
				mock_after_cost = CP_under_beta2(mock_sol,sol_type = "mock")
				if prev_cost < mock_after_cost:
					pass
				else: #개선 발생
					rev_sol = mock_sol
					after_cost = mock_after_cost
					break
		else:
			print ("Select Insert Type")
			pass
		ite += 1
		after_cost = CP_under_beta2(rev_sol, sol_type = "mock")
		if prev_cost < after_cost:
			non_imp_num += 1
		else: #개선이 발생함.
			non_imp_num = 0
			beta = beta * 1.05
			CP_num = len(cp_list)
		prev_cost = after_cost # 해 전제 비용 갱신
	print("E_sol_constructor Done")
	return beta, rev_sol, CP_num
def node_inserter2(env, veh, node, customer_infos, insert_index, fake_customer_infos):
	#실제 simpy부분은 건드리지 않고 route 부분만 건드림.
	veh[1].insert(insert_index, [node.name, node.location, 0, 0])
	node_index = 0
	for info in veh[1]:
		#print ("NI check",info)
		if node_index >= insert_index:
			service_time = 0
			if type(info[0]) == int:
				ct = customer_infos[info[0]]
				service_time = ct.service_time
			else:
				service_time = 3
			if insert_index == 0: #삽입된 위치가 맨 처음인 경우
				info[2] = env.now
				info[3] = env.now + service_time
			else: # 삽입된 index 이후로는 시간을 다시 갱신
				t1 = distance(veh[1][node_index -1][1], veh[1][node_index][1])
				veh[1][node_index][2] = veh[1][node_index -1][3] + t1
				veh[1][node_index][3] = veh[1][node_index][2] + service_time
		node_index += 1
	return veh
def driver_duplicater2(env, veh):
	#input : env, veh
	#process : construct mock route 
	#output : mock veh
	mock_driver = [[1000 + veh.name], []] #[info part , route part]
	if len(veh.veh.users) > 0: #현재 수행중인 고객이 존재 하기 때문에 이를 반영
		csn = veh.veh.users[0].info #current_servicing_node
		mock_driver[1].append([csn.name, csn.location, env.now, env.now +csn.service_time]) #[위치, 현재 시간, 출발시간(해당 노드)]
		for order in veh.veh.put_queue:
			node = order.info
			if len(mock_driver[1]) > 0:
				arrive_time = mock_driver[1][-1][3] + distance(mock_driver[1][-1][1], node.location)
				dep_time = arrive_time + node.service_time
				mock_driver[1].append([node.name, node.location, arrive_time, dep_time])
			else:
				arrive_time = env.now
				dep_time = env.now
				mock_driver[1].append([0, depot.location, arrive_time, dep_time])
		return mock_driver
	else:
		#veh already arrived to depot.
		print("driver_duplicater2 ; veh already arrived to depot")
		res = [[1000 + veh.name], [[0, depot.location, env.now, env.now]]]
		return res
def CP_under_beta2(sol, sol_type = "mock"):
	cost = []
	if sol_type == "mock":
		for veh in sol:
			route_cost = 0
			if len (veh[1]) > 1:
				for node_index in range(1,len(veh[1])):
					before = veh[1][node_index - 1][1]
					after = veh[1][node_index][1]
					route_cost += distance(before , after)
			cost.append(route_cost)
		return sum(cost)
	elif sol_type == "real":
		for veh in sol:
			route = []
			route_cost = 0
			for ct in veh.veh.users + veh.veh.queue:
				route.append(ct.location)
			if len (route) > 1:
				for node_index in range(1,len(route)):
					before = route[node_index - 1][1]
					after = route[node_index][1]
					route_cost += distance(before , after)
			cost.append(route_cost)
		return sum(cost)
	else:
		print("CP_under_beta2 input error")
		return None
def route_insert_cost2(node, sol, sol_para = True):
	#input : 노드, 해
	#output : 삽입위치, 증가비용
	sol_cost_list = []
	route_index = 0
	if sol_para == False:
		route_index = sol.name
	for veh_class in sol:
		route =  veh_class[1]
		route_cost_list = []
		if len(route) > 0:
			for index in range(0,len(route) + 1):
				cost = 0
				if index == 0 :
					first_node = route[index][1]
					cost = distance(depot.location, node.location) + distance(node.location, first_node) - distance(depot.location, first_node)
				elif 0 < index < len(route):
					before_node = route[index - 1][1]
					after_node = route[index][1]
					cost = distance(before_node, node.location) + distance(node.location, after_node) - distance(before_node, after_node)
				else:
					end_node = route[index - 1][1]
					cost = distance(end_node, node.location) + distance(node.location, depot.location) - distance(end_node, depot.location)
				route_cost_list.append([route_index, index, cost])
			route_cost_list.sort(key = operator.itemgetter(2))
			sol_cost_list.append(route_cost_list[0])
		else:
			print("Empty route")
	if len(sol_cost_list) > 0:
		sol_cost_list.sort(key = operator.itemgetter(2))
		return sol_cost_list[0]
	else:
		#print ("NO ROUTE IN route_insert_cost")
		return None
def node_inserter(env, veh, node, customer_infos, insert_index, fake_customer_infos):
	#print("Insert", node.name, "At", env.now)
	org_len = len(veh.veh.users) + len(veh.veh.put_queue)
	rev_veh = veh
	ct_list = []
	for order in veh.veh.put_queue:
		if type(order.info.name) != int:
			#customer_infos가 fake고객으로 이루어짐.
			#print ("Error node_inserter", order.info.name)
			for fake_c in fake_customer_infos:
				if fake_c.name == order.info.name:
					ct_list.append(fake_c)
					break
		else:
			ct_list.append(customer_infos[order.info.name])
	ct_list.insert(insert_index - 1, node)
	rev_veh.veh.put_queue = []
	for node in ct_list:
		env.process(rev_veh.Driving(env, node)) #veh.Driving()은 env와 class customer을 받는다.
	rev_len = len(rev_veh.veh.users) + len(rev_veh.veh.put_queue)
	if org_len + 1 != rev_len:
		#print("Vh # : ",rev_veh.name , "/ Node Inserter Error", org_len, ":", rev_len)
		pass
	return rev_veh
def E_customer_generator(env, duration, pro_info, ite):
	#input i,j크기의 pro_info <- numpy.array(가상 lamda_list의 크기)
	#output : 각 지역 i,j에 발생한 고객을 담은 list <- 각 요소들은 class customer
	total_lamda_list = []
	for _ in range(0,ite):
		zero_case = 0
		rev_lamda_list = []
		mean_val = []
		check = []
		for row in pro_info:
			lamda_row = []
			for col in row:
				#해당 람다에 맞게 값을 생성
				lamda = E_customer_gen(env, col, duration)
				lamda_row.append(lamda)
				check.append(lamda)
				if lamda == 0:
					zero_case += 1
			rev_lamda_list.append(lamda_row)
			mean_val.append(sum(lamda_row))
		#print("zero case", zero_case/float(ite),"among",250, "mean", sum(mean_val)/2500.0)
		#check.sort()
		#print (check)
		rev_lamda_list = np.array(rev_lamda_list)
		total_lamda_list.append(rev_lamda_list) #모든 좌표에 대해서 생성된 고객의 수가 저장된 2 행렬
	print ("ITE", ite, len(total_lamda_list), np.shape(total_lamda_list[0]))
	# total_lamda_list가 0 인 경우에는 연산이 안되므로 error출력
	if len(total_lamda_list) == 0:
		print ("total_lamda_list is None")
		return None
	#계산된 행렬의 값이 1보다 큰 경우를 count하는 방식
	result = np.zeros(np.shape(total_lamda_list[0]))
	index_list = []
	for i in range(0,len(result)):
		for j in range(0,len(result[0])):
			index_list.append([i,j])
	"""
	ite_index = 0
	for index in index_list:
		for info in total_lamda_list:
			print ("check_index",index, type(info),info[index[0]][index[1]])
			if info[index] >= 1 :
				result[index] += 1
	"""
	for index in index_list:
		for ite_index in range(0,len(total_lamda_list)):
			val = total_lamda_list[ite_index][index[0],index[1]]
			if val >= 1 :
				result[index[0],index[1]] += 1.0
	"""
	단순히 평균을 내는 방식
	for lamda_matrix in total_lamda_list:
		result += lamda_matrix
	"""
	rev_result = np.true_divide(result,len(total_lamda_list))
	print ("Shape : ", rev_result.shape, "/ mean :",rev_result.mean(),"/ Sum :",rev_result.sum())
	return rev_result
	
def E_customer_gen(env, lamda, duration):
	#input : lamda, 시간대 길이
	#output: 해당 시간대 발생한 고객의 수 <- int
	num = 0
	current_time = env.now
	end_time = env.now + duration
	ite = 0
	while current_time < end_time:
		if ite > 0:
			num += 1
		interval = nextTime(float(lamda)) * 60#분으로 변환
		current_time += interval # 고객의 도착 간격 분포에서 무작위로 뽑힌 t 값이 지난 후에 작동
		ite += 1
	return num
def Customer_lamda_recorder(env, lamda_infos, customer_infos):
	#Input : 
	#Output : t시간대 마다, 고객들의 lamda가 기록된 array의 list
	time_slot = int((env.now/60)/24) - 24*int((env.now/60)/24)
	lamda_info = np.zeros(lamda_infos[time_slot].shape)
	#print ("len_1", len(customer_infos),"len_2",len(lamda_info))
	for ct in customer_infos:
		#print ("ct_check",ct.location[0], ct.location[1])
		#print ("ct_check",ct.location[0]/2.5, ct.location[1]/2.5)
		lamda_info[int(ct.location[0]/2.5),int(ct.location[1]/2.5)] += 1
	for row_index in range(0,len(lamda_info)):
		for col_index in range(0,len(row)):
			#현재는 단순 평균으로 lamda를 갱신 
			lamda_infos[time_slot][row_index,col_index] = (lamda_infos[time_slot][row_index,col_index] + lamda_info[row_index,col_index])/2.0
	"""
	while True:
		for ct in customer_infos:
			#print ("ct_check",ct.location[0], ct.location[1])
			#print ("ct_check",ct.location[0]/2.5, ct.location[1]/2.5)
			lamda_info[int(ct.location[0]/2.5),int(ct.location[1]/2.5)] += 1
		for row_index in range(0,len(lamda_info)):
			for col_index in range(0,len(row)):
				#현재는 단순 평균으로 lamda를 갱신 
				lamda_infos[time_slot][row_index,col_index] = (lamda_infos[time_slot][row_index,col_index] + lamda_info[row_index,col_index])/2.0
	"""
	return lamda_infos
def route_insert_cost(node, sol, sol_para = True):
	#input : 노드, 해
	#output : 삽입위치, 증가비용
	sol_cost_list = []
	route_index = 0
	if sol_para == False:
		route_index = sol[0].name
	for veh_class in sol:
		#큐에 있는 고객 중 성분을 구분
		#print("route_insert_cost")
		real_ct_num = 0 
		fake_ct_num = 0
		for ct in veh_class.veh.users + veh_class.veh.put_queue:
			if type(ct.info.name) != int:
				fake_ct_num += 1
			else:
				real_ct_num += 1
		#print("Veh :", veh_class.name , "Len :", len(veh_class.veh.users + veh_class.veh.put_queue), "R_ct :" , real_ct_num, "/ F_ct : ", fake_ct_num)
		f1_time = time.time()
		route =  veh_class.veh.users + veh_class.veh.put_queue
		route_cost_list = []
		if len(route) > 0:
			for index in range(0,len(route) + 1):
				cost = 0
				if index == 0 :
					cost = distance(depot.location, node.location) + distance(node.location, route[index].info.location) - distance(depot.location, route[index].info.location)
				elif 0 < index < len(route):
					#print("list_out",route[index + 1])
					cost = distance(route[index - 1].info.location, node.location) + distance(node.location, route[index].info.location) - distance(route[index - 1].info.location, route[index].info.location)
				else:
					cost = distance(route[index - 1].info.location, node.location) + distance(node.location, depot.location) - distance(route[index - 1].info.location, depot.location)
				route_cost_list.append([route_index, index, cost])
			route_cost_list.sort(key = operator.itemgetter(2))
			sol_cost_list.append(route_cost_list[0])
		#print ("Ct  : ", str(time.time() - f1_time))
	if len(sol_cost_list) > 0:
		sol_cost_list.sort(key = operator.itemgetter(2))
		return sol_cost_list[0]
	else:
		print ("NO ROUTE IN route_insert_cost")
		return None, None, None
def Few_longest_node_set(sol, cp_infos, cp_num, benefit_beta = 0, route_para = True):
	#수정 버전
	#차고지에서 가장 멀리 떨어져 있는 고객을 제거.
	if route_para == True: #input되는 sol이 라우트 단위임.
		pass
	rev_sol = sol
	del_infos = []
	print (rev_sol)
	for veh in rev_sol:
		node_index = 0
		#print("FW", veh)
		print (veh)
		for node in veh.veh.put_queue:
			del_infos.append([veh.name, node_index, distance(depot.location, node.info.location), node.info])
			node_index += 1
	del_infos.sort(key = operator.itemgetter(2), reverse = True)
	added_num = 0
	for node_info in del_infos:
		if added_num > cp_num - len(cp_infos):
			break
		#해당 노드를 경로에서 제거
		elif node_info[2] >= benefit_beta:
			#rev_sol[node_info[0]] = env.process(node_deleter(env, rev_sol[node_info[0]], node_info[1], node_info[3]))
			rev_sol[node_info[0]] = node_deleter(env, rev_sol[node_info[0]], node_info[1], node_info[3])
			#해당 정보를 cp_infos에 추가
			cp_infos.append(node_info[3])
			added_num += 1
	return rev_sol, cp_infos
def node_deleter(env, veh, del_index, del_node_info):
	#단 여기서 del_index는 put_queue에서 만 인덱스를 카운트 한 것임.
	node_list = []
	for customer_index in range(0,len(veh.veh.put_queue)):
		if customer_index == del_index:
			pass
		else:
			node_list.append(veh.veh.put_queue[customer_index].info)
	veh.veh.put_queue = []
	for node in node_list:
		env.process(veh.Driving(env, node))
	return veh

def route_customer_deleter(veh, cp_infos ,len_cp_list, rev_beta):
	rev_veh = [veh]
	if type(veh) == list:
		print ("TYPE ERROR")
		print(veh)
	#if len(veh) == 1:
	#	veh = veh[0]
	for _ in range(0,len_cp_list): #현재 가능한 CP만큼 고객을 제거 해 보자.
		if random.random() < 0.5:
			rev_veh , rev_cp_infos = Few_longest_node_set(rev_veh, cp_infos, len_cp_list, benefit_beta = rev_beta)
		else:
			rev_veh , rev_cp_infos  = Worst_node_set(rev_veh, cp_infos, len_cp_list, benefit_beta = rev_beta)
	"""
	for order in veh.veh.users + veh.veh.put_queue:
		if random.random() < 0.5:
			rev_veh , rev_cp_infos = Few_longest_node_set(rev_veh, cp_infos, len_cp_list, benefit_beta = rev_beta)
		else:
			rev_veh , rev_cp_infos  = Worst_node_set(rev_veh, cp_infos, len_cp_list, benefit_beta = rev_beta)
	"""
	return rev_veh[0], rev_cp_infos
def Worst_node_set(sol, cp_infos, cp_num, benefit_beta = 0):
	#수정 버전
	#한번에 한개의 고객을 제거 하는 것.
	rev_sol = sol
	for ite in range(0,cp_num - len(cp_infos)):
		del_infos = []
		for veh in rev_sol:
			node_index = 0
			#print("WNS", veh)
			route = veh.veh.users + veh.veh.put_queue
			for node in route: #해에서 제거 되었을 때, 가장 비용 감소가 큰 노드 식별.
				if node_index == 0:
					pass
				elif node_index == len(route) - 1:
					if node.info.name == 0:
						pass
					else:
						cost = distance(route[node_index].info.location, depot.location)
						del_infos.append([veh.name, node_index, cost, node.info.name])
					break #더 이상 진행하면 안됨.
				else:
					print("lt1",node_index, len(route))					
					org_a = distance(route[node_index - 1].info.location,route[node_index].info.location)
					org_b = distance(route[node_index].info.location, route[node_index + 1].info.location)
					org_c = distance(route[node_index - 1].info.location,route[node_index + 1].info.location)
					cost = org_a + org_b - org_c
					del_infos.append([veh.name, node_index, cost, node.info.name])
				node_index += 1
		if len(del_infos) > 0:
			del_infos.sort(key = operator.itemgetter(2), reverse = True)
			if del_infos[0][2] >= benefit_beta :
				#rev_sol[del_infos[0][0]] = env.process(node_deleter(env, rev_sol[del_infos[0][0]], del_infos[0][1], del_infos[0][3]))#제거
				rev_sol[del_infos[0][0]] = node_deleter(env, rev_sol[del_infos[0][0]], del_infos[0][1], del_infos[0][3])#제거
				cp_infos.append(del_infos[0][3])
			else:
				#하나씩 계산을 하기 때문에, 한 개의 노드라도 benefit_beta를 넘지 못하면 이미 다음 노드들은 해당 값보다 클 수 없음.
				return rev_sol , cp_infos
	return rev_sol, cp_infos
def normalized_pareto_count_rank(data1,data2):
	#각각을 정규화 시키기.
	nor_data1 = []
	max_data1 = max(data1)
	min_data1 = min(data1)
	for data in data1:
		nor_data1.append((data1 - min_data1)/(max_data1 - min_data1))
	nor_data2 = []
	max_data2 = max(data2)
	min_data2 = min(data2)
	for data in data2:
		nor_data1.append((data2 - min_data2)/(max_data2 - min_data2))
	combined_data = []
	for index in range(0,len(data1)):
		combined_data.append([nor_data1[index],nor_data2[index]])
	res = []
	index = 0
	for point in combined_data:
		scroe = 0
		for point_com in combined_data:
			if point != point_com:
				if point[0] > point_com[0] and point[1] > point_com[1]:
					scroe += 1
		res.append([index, scroe])
		index += 1
	return res
def system_checker(env, veh_set, customer_infos):
	while True:
		print ("current time ", env.now, "sys check")
		for veh in veh_set:
			print ("Veh:", veh.name , "/ Queue:", len(veh.veh.users + veh.veh.put_queue))
		assigned = 0
		done = 0
		loaded_but_not_delivered = 0
		nope = 0
		for ct in customer_infos:
			if ct.loaded == True:
				assigned += 1
				if ct.done == True:
					done += 1
				else:
					loaded_but_not_delivered += 1
			else:
				nope += 1
		print ("Now :",env.now,"/ Total : ", len(customer_infos), "/ assigned :", assigned, "/done : ",done,"/Not yet", loaded_but_not_delivered, "/nope : ", nope)
		yield env.timeout(60)
def veh_checker(env, veh_set):
	while True:
		for veh in veh_set:
			f_ct = 0
			r_ct = 0
			for order in veh.veh.users + veh.veh.put_queue:
				if type(order.info.name) == int:
					r_ct += 1
				else:
					f_ct += 1
			#print ("For check -> ",env.now ,"r_ct : ", r_ct, "/ f_ct : ", f_ct)
		yield env.timeout(1)
def driver_duplicate(env, veh_class):
	rev_veh = driver(env, veh_class.name)
	#나머지 키 복수 작업
	rev_veh.last_location = copy.deepcopy(veh_class.last_location)
	rev_veh.served = copy.deepcopy(veh_class.served)
	rev_veh.now_return = copy.deepcopy(veh_class.now_return)
	rev_veh.released_ct = copy.deepcopy(veh_class.released_ct)
	rev_veh.done = copy.deepcopy(veh_class.done)
	#고객들을 다시 driving으로 할당하기.
	order_info = []
	for order in veh_class.veh.put_queue:
		order_info.append(order.info)
	#현재 수행중인 고객 생성
	if len(veh_class.veh.users) > 0:
		rev_veh.last_location = copy.deepcopy(veh_class.veh.users[0].info.location)
	"""
	if len(order_info) > 0:
		dummy_location = order_info[0].location
		rev_veh.last_location = copy.deepcopy(order_info[0].location)
	if len(veh_class.veh.users) > 0:
		dummy_name = veh_class.veh.users[0]
		dummy_ct = Customer_Gen(env, dummy_name, input_location = )
	"""
	#order_info의 고객들을 driving으로 할당
	if len(order_info) > 0:
		for customer in order_info:
			env.process(rev_veh.Driving(env, customer))
	print ("DUplicate info", len(order_info))
	return rev_veh
#복귀시킨 차량의 라우트 구성
#쿠팡플렉스에 일 할당
speed = 2
env = simpy.Environment()
#필요한 data 저장소 정의
depot = Customer_Gen(env,0)
customer_list = [depot]
request_list = []
veh_set = []
#lamda_list = [60,60,60,60,60,60,60,60,60,60,60,60]
#lamda_maker
int_lamda_list = []
for i in range(0,50):
	row = []
	for j in range(0,50):
		row.append(random.random()*5)
	int_lamda_list.append(row)
lamda_list = []
lamda_list.append(np.array(int_lamda_list))
print ("lamda_list_info",len(lamda_list), len(lamda_list[0]))
policy_t = 120
stack_time = 60
simul_interval = 60
#초기 고객 및 라우트 생성
#1.1 초기 차량 생성
veh_index = 0
for i in range(0,4):
	veh_set.append(driver(env,veh_index))
	veh_index += 1
#1.2 초기 고객 생성
initial_Customer_Generator(env)
#1.3 초기해 생성
test = clustered_coord(veh_set, customer_list)
index = 0
for ele in test:
	eles = []
	for node in ele:
		eles.append(node.location)
	route = NN_Algo(eles, customer_list, start = True)
	for ct_name in route:
		#고객들을 차량에 request로 할당해야 함.
		if ct_name == 0:
			pass
		else:
			env.process(veh_set[index].Driving(env, customer_list[ct_name]))
	index += 1
print ("초기해 및 초기 라우트 할당 완료", env.now)
"""
for veh in veh_set:
	print("Veh. # :" , veh.name , "qeue # " , len(veh.veh.users) + len(veh.veh.put_queue))
"""
env.process(IntervalCP(500))
env.process(Cusotmer_Generator(env, lamda_list))
env.process(Algo_ver2(env, veh_set, customer_list, lamda_list, policy_t, stack_time, CoupangFlex1, end_time = 288, lamda_matrix_size = 50))
env.process(veh_checker(env, veh_set))
env.process(system_checker(env, veh_set,customer_list))
env.process(idle_veh_run(veh_set, customer_list))
env.run(until = 600)
#print ("차량에 할당된 고객 수", "전체 고객 수: ", len(customer_list), "CP수", len(CoupangFlex1))
for veh in veh_set:
	print (veh.name,len(veh.veh.put_queue))
#os.system("pause")
