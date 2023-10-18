import random, math
from _functools import reduce
from copy import copy
from builtins import isinstance
from resource import setrlimit, RLIMIT_AS, RLIMIT_DATA
from heapq import heappop, heappush
import time
import sys
sys.setrecursionlimit(10**7)

class Hanoi:
	NTOWERS = 4
	TOWERS = ["A", "B", "C", "D"]

	def __init__(self, n, puzzle : dict[str, list[int]] = {}, movesList : list[tuple[str, str]] = []):
		self.NDISKS = n
		self.state = copy(puzzle) if puzzle else self.start_state(n)
		self.moves = copy(movesList)
	
	def start_state(self, n):
		return {"A": list(reversed(range(1, n + 1))), "B": [], "C": [], "D": []}

	def solved_state(self, n):
		return {"A": [], "B": [], "C": [], "D": list(reversed(range(1, n + 1)))}

	def display(self) -> str:
		for key in self.state:
			print(key, " | ", *self.state[key])
		print("\n")

	def display_moves(self):
		for move in self.moves:
			print("Move disk from " + move[0] + " to " + move[1])
	
	def clear_moves(self):
		"""Șterge istoria mutărilor pentru această stare."""
		self.moves.clear()

	def apply_move_inplace(self, move : tuple[str, str]):
		"""Aplică o mutare, modificând această stare."""
		source = move[0]
		dest = move[1]

		# Daca se muta de pe turn gol, miscarea este invalida
		if not self.state[source]:
			return None

		source_disk = self.state[source][-1]

		if self.state[dest]:
			dest_disk = self.state[dest][-1]

			# Daca se misca un disc mai mare pe un disc mai mic, miscarea este invalida
			if source_disk > dest_disk:
				return None
	
		self.state[source].pop()
		self.state[dest].append(source_disk)
		self.moves.append(move)
		return self
	
	def apply_move(self, move : tuple[str, str]):
		"""Construiește o nouă stare, rezultată în urma aplicării mutării date."""
		return self.clone().apply_move_inplace(move)
		
	def verify_solved(self, moves : list[tuple[str, str]]) -> bool:
		""""Verifică dacă aplicarea mutărilor date pe starea curentă duce la soluție"""
		for move in moves:
			if self.apply_move_inplace(move) is None:
				return False
		return self.state == self.solved_state(self.NDISKS)

	def clone(self):
		return Hanoi(self.NDISKS, self.state, self.moves)

class NPuzzle:
	"""
	Reprezentarea unei stări a problemei și a istoriei mutărilor care au adus starea aici.
	
	Conține funcționalitate pentru
	- afișare
	- citirea unei stări dintr-o intrare pe o linie de text
	- obținerea sau ștergerea istoriei de mutări
	- obținerea variantei rezolvate a acestei probleme
	- verificarea dacă o listă de mutări fac ca această stare să devină rezolvată.
	"""

	NMOVES = 4
	UP, DOWN, LEFT, RIGHT = range(NMOVES)
	ACTIONS = [UP, DOWN, LEFT, RIGHT]
	names = "UP, DOWN, LEFT, RIGHT".split(", ")
	BLANK = ' '
	delta = dict(zip(ACTIONS, [(-1, 0), (1, 0), (0, -1), (0, 1)]))
	
	PAD = 2
	
	def __init__(self, puzzle : list[int | str], movesList : list[int] = []):
		"""
		Creează o stare nouă pe baza unei liste liniare de piese, care se copiază.
		
		Opțional, se poate copia și lista de mutări dată.
		"""
		self.N = len(puzzle)
		self.side = int(math.sqrt(self.N))
		self.r = copy(puzzle)
		self.moves = copy(movesList)
	
	def display(self, show = True) -> str:
		l = "-" * ((NPuzzle.PAD + 1) * self.side + 1)
		aslist = self.r
		
		slices = [aslist[ slice * self.side : (slice+1) * self.side ]  for slice in range(self.side)]
		s = ' |\n| '.join([' '.join([str(e).rjust(NPuzzle.PAD, ' ') for e in line]) for line in slices]) 
	
		s = ' ' + l + '\n| ' + s + ' |\n ' + l
		if show: print(s)
		return s
	def display_moves(self):
		print([self.names[a] if a is not None else None for a in self.moves])

	def get_nr_of_moves(self):
		return len(self.moves)
		
	def print_line(self):
		return str(self.r)
	
	@staticmethod
	def read_from_line(line : str):
		list = line.strip('\n][').split(', ')
		numeric = [NPuzzle.BLANK if e == "' '" else int(e) for e in list]
		return NPuzzle(numeric)
	
	def clear_moves(self):
		"""Șterge istoria mutărilor pentru această stare."""
		self.moves.clear()
	
	def apply_move_inplace(self, move : int):
		"""Aplică o mutare, modificând această stare."""
		blankpos = self.r.index(NPuzzle.BLANK)
		y, x = blankpos // self.side, blankpos % self.side
		ny, nx = y + NPuzzle.delta[move][0], x + NPuzzle.delta[move][1]
		if ny < 0 or ny >= self.side or nx < 0 or nx >= self.side: return None
		newpos = ny * self.side + nx
		piece = self.r[newpos]
		self.r[blankpos] = piece
		self.r[newpos] = NPuzzle.BLANK
		self.moves.append(move)
		return self
	
	def apply_move(self, move : int):
		"""Construiește o nouă stare, rezultată în urma aplicării mutării date."""
		return self.clone().apply_move_inplace(move)

	def solved(self):
		"""Întoarce varianta rezolvată a unei probleme de aceeași dimensiune."""
		return NPuzzle(list(range(self.N))[1:] + [NPuzzle.BLANK])

	def verify_solved(self, moves : list[int]) -> bool:
		""""Verifică dacă aplicarea mutărilor date pe starea curentă duce la soluție"""
		return reduce(lambda s, m: s.apply_move_inplace(m), moves, self.clone()) == self.solved()

	def clone(self):
		return NPuzzle(self.r, self.moves)
	def __str__(self) -> str:
		return str(self.N-1) + "-puzzle:" + str(self.r)
	def __repr__(self) -> str: return str(self)
	def __eq__(self, other):
		return self.r == other.r
	def __lt__(self, other):
		return True
	def __hash__(self):
		return hash(tuple(self.r))
	

# generare
def genOne(side, difficulty):
	state = NPuzzle(list(range(side * side))[1:] + [NPuzzle.BLANK])
	for i in range(side ** difficulty + random.choice(range(side ** (difficulty//2)))):
		s = state.apply_move(random.choice(NPuzzle.ACTIONS))
		if s is not None: state = s
	state.clear_moves()
	return state

# heuristic 1 : Hamming
def hamming(start, end):
	cost = 0
	length = len(start.r);

	for i in range(0, length):
		if start.r[i] != end.r[i] and isinstance(start.r[i], int):
			cost +=1

	return cost

def manhattan_distance(a, b):
    dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    return dist

# heuristic 2: Manhatten
def manhattan(start, end):
	cost = 0

	for idx, item in enumerate(start.r):
		if isinstance(item, int):

			# pos of tile in starting position
			i = idx // start.side
			j = idx - i * start.side

			pos_a = (i, j)

			# pos in the tile in ending position
			desired_idx = item - 1
			i = desired_idx // start.side
			j = desired_idx - i * start.side

			pos_b = (i, j)

			dist = manhattan_distance(pos_a, pos_b)
			cost += dist

	return cost

# heuristic 3: basic cost for Hanoi towers - probably a very badly implemented heuristic for it tho'
# basically it counts the nr of rings that are not on the last tower for a problem with 4 towers
def hanoi_cost(start):
	correct_rings = start.state['D']
	cost = start.NDISKS - correct_rings
	return cost

def get_children(parent):
	children = []

	child1 = parent.apply_move(parent.UP)
	if child1:
		children.append(child1)

	child2 = parent.apply_move(parent.DOWN)
	if child2:
		children.append(child2)

	child3 = parent.apply_move(parent.LEFT)
	if child3:
		children.append(child3)

	child4 = parent.apply_move(parent.RIGHT)
	if child4:
		children.append(child4)

	return children

# A*
def astar(start, end, h):
	frontier = []
	heappush(frontier, (0 + h(start, end), start))

	discovered = {start : (None, 0)}
	while frontier:
		dist, parent = heappop(frontier)
		current_dist = discovered[parent][1]

		if parent == end:
			break

		children = get_children(parent)

		for child in children:
			new_dist = current_dist + 1
			if child not in discovered or new_dist < discovered[child][1]:
				discovered[child] = (parent, new_dist)
				heappush(frontier, (new_dist + h(child, end), child))

	# states_saved = len(discovered) + len(frontier)
	states_saved = len(discovered)
	return parent, states_saved

# Beam Search
def beam_search(start, end, B, h, limit):
	beam = [start]
	discovered = {start : (None, 0)}

	while beam and len(discovered) < limit:
		succ = {}
		selected = []
		for s in beam:
			children = get_children(s)
			for child in children:
				if child == end:
					# states_saved = len(discovered) + len(beam)
					states_saved = len(discovered)
					return child, states_saved
				if child not in discovered:
					succ[child] = h(child, end)

		for k , v in sorted(succ.items(), key=lambda item: item[1]):
			selected.append(k)
			discovered[k] = h(k, end)
			if len(selected) == B:
				break

		beam = selected

	# states_saved = len(discovered) + len(beam)
	states_saved = len(discovered)
	return None, states_saved


# Generalized Limited Discrepancy Search
def GLDS(start, end, h, limit):
	discovered = {start: (None, 0)}
	discrepancies = 0

	# adding a 10 mins time limit
	start_time = time.time()
	while True and (time.time() - start_time) < 600:
		# print("Discrepancies :" + str(discrepancies))
		result = iteration(start, end, discrepancies, h, discovered, limit)
		if result:
			states_saved = len(discovered)
			return result, states_saved
		discrepancies += 1

	return None, len(discovered)

# Iterations for GLDS
def iteration(state, end, discrepancies, h, visited, limit):
	succ = []
	children = get_children(state)

	for s in children:
		if s == end:
			visited.pop(state, None)
			return s
		if s not in visited:
			succ.append(s)

	if len(succ) == 0:
		visited.pop(state, None)
		return None
	if len(visited) > limit:
		visited.pop(state, None)
		return None

	best_h = h(succ[0], end)
	best = succ[0]

	for s in succ:
		if h(s, end) < best_h:
			best_h = h(s, end)
			best = s

	if discrepancies == 0:
		visited[best] = best_h
		return iteration(best, end, 0, h, visited, limit)
	else:
		succ.remove(best)
		while len(succ) != 0:
			s_h = h(succ[0], end)
			ss = succ[0]
			for s in succ:
				if h(s, end) < best_h:
					s_h = h(s, end)
					ss = s
			succ.remove(ss)
			visited[ss] = s_h
			result = iteration(ss, end, discrepancies-1, h, visited, limit)
			# visited.pop(ss, None)
			if result:
				visited.pop(state, None)
				return result
		visited[best] = best_h
		return iteration(best, end, discrepancies, h, visited, limit)

# BLDS
def BLDS(start, end, h, B, limit):
	discovered = {start: (None, 0)}
	discrepancies = 0

	start_time = time.time()
	while True and (time.time() - start_time) < 600:
		result = B_iteraion([start], end, discrepancies, B, h, discovered, limit)
		if result:
			return result, len(discovered)
		discrepancies += 1
		# print("In BLDS, discrepancies: " + str(discrepancies))

	return None, len(discovered)

# Iterations for BLDS
def B_iteraion(level, end, discrepancies, B, h, visited, limit):
	succ = {}
	new_visited = visited.copy()
	for s in level:
		children = get_children(s)
		for child in children:
			if child == end:
				return child
			if child not in new_visited:
				new_visited[child] = h(child, end)
				succ[child] = h(child, end)
	# print("Nr of children saved in succ is: " + str(len(succ)))
	if len(succ) == 0:
		return None
	if len(new_visited) + min(B, len(succ)) > limit:
		return None
	# sort succ by h
	succ = dict(sorted(succ.items(), key=lambda item: item[1]))

	if discrepancies == 0:
		next_level = []
		for i in range(0, B):
			if i >= len(succ):
				break
			next_level.append(list(succ.keys())[i])

		return B_iteraion(next_level, end, 0, B, h, new_visited, limit)
	else:
		already_explored = B
		while already_explored < len(succ):
			# print("In while in iteration, already_explored: " + str(already_explored) + " < " +  str(len(succ)))
			n = min(len(succ) - already_explored, B)
			new_visited = visited.copy()

			next_level = []
			for i in range(already_explored, already_explored+n):
				if len(succ) <= i:
					break
				key = list(succ.keys())[i]
				new_visited[key] = h(key, end)
				next_level.append(key)

			# print("next lvl is :" + str(len(next_level)))
			val = B_iteraion(next_level, end, discrepancies-1, B, h, new_visited, limit)
			if val: 
				return val

			already_explored = already_explored + len(next_level)

		# print("exited while")
		new_visited = visited.copy()
		next_level = []
		for i in range(0, B):
			if len(succ) <= i:
					break
			key = list(succ.keys())[i]
			new_visited[key] = h(key, end)
			next_level.append(key)

		return B_iteraion(next_level, end, discrepancies, B, h, new_visited, limit)

# main

MLIMIT = 3 * 10 ** 9 # 3 GB RAM limit
setrlimit(RLIMIT_DATA, (MLIMIT, MLIMIT))


f = open("files/problems4.txt", "r")
input = f.readlines()
f.close()
problems = [NPuzzle.read_from_line(line) for line in input]
# problems[0].display()

i = 0

Blist = [1, 10, 50, 100, 500, 1000]
limit = 100000
if(problems[i].side == 5):
	limit = 500000
if(problems[i].side == 6):
	limit = 1000000
# print(problems[i].side)

print("Testing for BLDS for N = " + str(problems[i].side))
for i in range (0, 5):
	for B in Blist:
		print("test " + str(i) + " with B = " + str(B) + ":")
		start_time = time.time()
		result, states_saved = BLDS(problems[i], problems[i].solved(), manhattan, B, limit)
		end_time = time.time() - start_time

		if result:
			print("Found solution, path length is " + str(result.get_nr_of_moves()) + ", nr of states saved is "
				+ str(states_saved))
			print("Time(s): " + str(end_time))
		else:
			print("Failed to find solution, nr of states saved is " + str(states_saved))
			print("Time(s): " + str(end_time))

"""
print("Testing GLDS for N = " + str(problems[i].side))

for i in range(0, 5):
	print("test " + str(i) + ":")
	start_time = time.time()
	result, states_saved = GLDS(problems[i], problems[i].solved(), hamming, limit)
	end_time = time.time() - start_time

	if result:
		print("Found solution, path length is " + str(result.get_nr_of_moves()) + ", nr of states saved is "
			+ str(states_saved))
		print("Time(s): " + str(end_time))
	else:
		print("Failed to find solution, nr of states saved is " + str(states_saved))
		print("Time(s): " + str(end_time))
"""
"""
print("Testing GLDS for N = " + str(problems[i].side))
result, states_saved = GLDS(problems[i], problems[i].solved(), manhattan, limit)

if result:
	print("Found solution, path length is " + str(result.get_nr_of_moves()) + ", nr of states saved is "
		+ str(states_saved))
"""
"""
for B in Blist:
	start_time = time.time()
	path, states_saved = beam_search(problems[i], problems[i].solved(), B, manhattan, limit)
	end_time = time.time() - start_time

	print("Beam search for B = " + str(B) + ", limit = " + str(limit) + ":")
	if (path):
		print("runtime: " + str(end_time) + " seconds, nr of saved states: " 
			+ str(states_saved) + ", path length: " + str(path.get_nr_of_moves()))
	else:
		print("Did not find a solution.")
		print("runtime: " + str(end_time) + " seconds, nr of saved states: " 
			+ str(states_saved))
"""
"""
start_time = time.time()
path, states_saved = astar(problems[0], problems[0].solved(), manhattan)
end_time = time.time() - start_time

print("A* with manhattan heuristic ran for " + str(end_time) + " seconds, path length is of " + 
		str(path.get_nr_of_moves()) + ", nr of saved states is " 
		+ str(states_saved))
"""

# solved_problem = problems[0].solved()
# solved_problem.display()

"""
print("Generare:")
random.seed(4242)
p = genOne(3, 4)
p.display()
(p.solved()).display()
# problemele easy au fost generate cu dificultatile 4, 3, respectiv 2 (pentru marimile 4, 5, 6)
# celelalte probleme au toate dificultate 6

print("Hamming cost is: ")
print(hamming(p, p.solved()))
print("Manhattan cost is: ")
print(manhattan(p, p.solved()))

hamming_path = astar(p, p.solved(), hamming)
print("A* with hamming heuristic:")
hamming_path.display_moves()

manhattan_path = astar(p, p.solved(), manhattan)
print("A* with manhattan heuristic:")
manhattan_path.display_moves()

B = 10
limit = 100000
print("Pentru B = 10, limit = 100000:")

beam_path_hamming = beam_search(p, p.solved(), B, hamming, limit)
print("Beam Search with hamming heuristic:")
if beam_path_hamming:
	beam_path_hamming.display_moves()
else:
	print("Did not reach a solution")

beam_path_manhattan = beam_search(p, p.solved(), B, manhattan, limit)
print("Beam Search with manhattan heuristic:")
if beam_path_manhattan:
	beam_path_manhattan.display_moves()
else:
	print("Did not reach a solution")

"""
