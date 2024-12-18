
from contest.capture_agents import CaptureAgent
import contest.util as util
import random, time
from contest.util import nearest_point

from contest.game import Directions, Actions
import contest.game
import json




def create_team(first_index, second_index, is_red,
               first = 'Agent2', second = 'Agent1', num_training = 10, **args):

  # The following line is an example only; feel free to change it.
  return [eval(first)(first_index), eval(second)(second_index)]


class approx_q_learning_offense(CaptureAgent):

  def set_default_file(self):
    self.default_file = "weights.json"


  def rec(self,i,j,depth_map,visited,h):
    sum = 0
    visited[h-i-1][j] = True
    direction = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dy,dx in direction:
      y = i + dy
      x = j + dx
      if visited[h-y-1][x] == False and depth_map[h-y-1][x] == 3:
        self.dead_ends_depth[h-y-1][x] = self.rec(y,x,depth_map,visited,h)
        sum = self.dead_ends_depth[h-y-1][x]
    return sum + 1

  def register_initial_state(self, game_state):
    self.default_file = "weights.json"
  


    w = game_state.data.layout.width
    h = game_state.data.layout.height


    # Initialize state maps
    self.dead_ends_depth = [[0] * w for _ in range(h)]
    depth_map = [[0] * w for _ in range(h)]
    visited = [[False] * w for _ in range(h)]
    walls = game_state.get_walls()
    queue = util.Queue()
    # Start exploration from each unvisited, non-wall cell
    for i in range(h):
        for j in range(w):
            if not game_state.has_wall(j, i):
                neighbors = Actions.get_legal_neighbors((j, i), walls)
                depth_map[h-i-1][j] = len(neighbors)
                if len(neighbors) == 2:
                  queue.push((i, j))
    
    while not queue.is_empty():
      i,j = queue.pop()
      self.dead_ends_depth[h-i-1][j] = self.rec(i,j,depth_map,visited,h)

    self.max_dead_end = max(value for row in self.dead_ends_depth for value in row)

    self.set_default_file()
    self.total_food = len(self.get_food(game_state).as_list())
    width = game_state.data.layout.width
    height = game_state.data.layout.height
    self.max_distance = (width**2 + height**2)**0.5

    self.defence_ancor_point = (0,0)
    neighbors = []
    if game_state.is_on_red_team(self.index):
      neighbors = [(0, 1), (0, -1), (-1, 1), (-1, 0), (-1,-1) ]
    else:
      neighbors = [(0, 1), (0, -1), (1, 1), (1, 0), (1,-1) ]



    self.default_defence_ancor_point1 = self.defence_ancor_point
    self.default_defence_ancor_point2 = self.defence_ancor_point

    # LOWER CHECK POINT
    temp_h = max(1, h - 3)
    if game_state.is_on_red_team(self.index):
      self.defence_ancor_point1 = (w // 2 - 2, temp_h)
    else:
      self.defence_ancor_point1 = (w // 2 + 1, temp_h)

    if game_state.has_wall(self.defence_ancor_point1[0], self.defence_ancor_point1[1]):
      dist = 1
      while game_state.has_wall(self.defence_ancor_point1[0], self.defence_ancor_point1[1]):
        for dx,dy in neighbors:
          self.defence_ancor_point1 = (self.defence_ancor_point1[0] + dist*dx, self.defence_ancor_point1[1] + dist * dy)
          if not game_state.has_wall(self.defence_ancor_point1[0], self.defence_ancor_point1[1]):
            break
        dist += 1
    self.default_defence_ancor_point1 = self.defence_ancor_point1

    # UPPER CHECK POINT
    temp_h = min(h-2 , 3)
    if game_state.is_on_red_team(self.index):
      self.defence_ancor_point2 = (w // 2 - 2, temp_h)
    else:
      self.defence_ancor_point2 = (w // 2 + 1, temp_h)

    if game_state.has_wall(self.defence_ancor_point2[0], self.defence_ancor_point2[1]):
      dist = 1
      while game_state.has_wall(self.defence_ancor_point2[0], self.defence_ancor_point2[1]):
        for dx,dy in neighbors:
          self.defence_ancor_point2 = (self.defence_ancor_point2[0] + dist*dx, self.defence_ancor_point2[1] + dist * dy)
          if not game_state.has_wall(self.defence_ancor_point2[0], self.defence_ancor_point2[1]):
            break
        dist += 1
    self.default_defence_ancor_point2 = self.defence_ancor_point2
    # print(self.default_defence_ancor_point1, self.default_defence_ancor_point2)
    self.mode = "ATTACK"


        

    self.start = game_state.get_agent_position(self.index)
    self.features_extractor = features_extractor(self)
    CaptureAgent.register_initial_state(self, game_state)

  def choose_action(self, game_state):

    legal_actions = game_state.get_legal_actions(self.index)
    if len(legal_actions) == 0:
      return None

    food_left = len(self.get_food(game_state).as_list())

    if food_left <= 0:
    # if food_left <= 2:
      best_dist = 9999
      for action in legal_actions:
        successor = self.get_successor(game_state, action)
        pos2 = successor.get_agent_position(self.index)
        dist = self.get_maze_distance(self.start, pos2)
        if dist < best_dist:
          best_action = action
          best_dist = dist
      return best_action


    action = self.find_best_action(game_state)
    return action


  def get_q_value(self, game_state, action):
    features = self.features_extractor.get_features(game_state, action)
    return features * self.weights

        
  def final(self, state):
    CaptureAgent.final(self, state)


  def get_successor(self, game_state, action):

    successor = game_state.generate_successor(self.index, action)
    pos = successor.get_agent_state(self.index).get_position()
    if pos != nearest_point(pos):
      return successor.generate_successor(self.index, action)
    else:
      return successor

  def compute_value_from_q_values(self, game_state):

    allowed_actions = game_state.get_legal_actions(self.index)
    if len(allowed_actions) == 0:
      return 0.0
    best_action = self.find_best_action(game_state)
    return self.get_q_value(game_state, best_action)

  def find_best_action(self, game_state):

    legal_actions = game_state.get_legal_actions(self.index)
    if len(legal_actions) == 0:
      return None
    action_vals = {}
    best_q_value = float('-inf')
    for action in legal_actions:
      target_q_value = self.get_q_value(game_state, action)
      action_vals[action] = target_q_value
      if target_q_value > best_q_value:
        best_q_value = target_q_value
    best_actions = [k for k, v in action_vals.items() if v == best_q_value]
    # random tie-breaking
    return random.choice(best_actions)

class features_extractor:

  def __init__(self, agent_instance):
    self.agent_instance = agent_instance

  
  def get_features(self, game_state, action):
    
    # extract the grid of food and wall locations and get the ghost locations
    food = self.agent_instance.get_food(game_state)


    walls = game_state.get_walls()
    enemies = [game_state.get_agent_state(i) for i in self.agent_instance.get_opponents(game_state)]
    ghosts= [a for a in enemies if not a.is_pacman and a.get_position() != None]
    enemy_pacman = [a.get_position() for a in enemies if a.is_pacman and a.get_position() != None]
    # if self.agent_instance.get_score(game_state) != 0:
    #   print(self.agent_instance.get_score(game_state))
    features = util.Counter()
    features["bias"] = 1.0
    
    agent_position = game_state.get_agent_position(self.agent_instance.index)
    x, y = agent_position
    dx, dy = Actions.direction_to_vector(action)
    next_x, next_y = int(x + dx), int(y + dy)

    if x == self.agent_instance.default_defence_ancor_point1[0] and y == self.agent_instance.default_defence_ancor_point1[1]:
      self.agent_instance.defence_ancor_point = self.agent_instance.default_defence_ancor_point2
    elif x == self.agent_instance.default_defence_ancor_point2[0] and y == self.agent_instance.default_defence_ancor_point2[1]:
      self.agent_instance.defence_ancor_point = self.agent_instance.default_defence_ancor_point1


    


    # dist = self.closest_food((next_x, next_y), food, walls)
    dist = self.closest_food2((next_x, next_y), food, walls, game_state)
    if dist is not None:
      # 
      # pomalku
      features["closest-food"] = float(dist) / (walls.width**2 + walls.height**2)**0.5
    
    # closest_enemy_ghost
    if len(ghosts) > 0:
      min_ghost = min(ghosts, key=lambda g: self.agent_instance.get_maze_distance((next_x, next_y), g.get_position()))
      closest_enemy = self.agent_instance.get_maze_distance((next_x, next_y), min_ghost.get_position())

      if min_ghost.scared_timer > 0:
        features["closest_enemy_ghost"] = -2* (6 - closest_enemy) / 6
      elif closest_enemy <= 5:
        # pomalku
        features["closest_enemy_ghost"] = (6 - closest_enemy) / 6
        features["closest-food"] = 999
      elif closest_enemy == 0:
        features["dead"] = 1
        

    
    features["closest_enemy_pacman"] = 0
    # closest_enemy_pacman
    if len(enemy_pacman) > 0:
      closest_enemy = min([self.agent_instance.get_maze_distance((next_x, next_y), e) for e in enemy_pacman])
      if closest_enemy <= 5 and game_state.get_agent_state(self.agent_instance.index).scared_timer == 0:
        # poveke
        features["closest_enemy_pacman"] = (6 - closest_enemy) / 6
    
    features["closest_enemy_pacman_vertical"] = 0
    # verical distance to enemy VERTICAL    
    if len(enemy_pacman) > 0 :
      closest_enemy = min([abs(next_y - e[1]) for e in enemy_pacman])
      if closest_enemy <= 5 and game_state.get_agent_state(self.agent_instance.index).scared_timer == 0:
        # poveke
        features["closest_enemy_pacman_vertical"] = (6 - closest_enemy) / 6


    next_state = self.agent_instance.get_successor(game_state, action)

    # pomalku
    features["food_left"] = len(food.as_list()) / self.agent_instance.total_food

    # poveke
    features["score"] = (self.agent_instance.get_score(next_state) + self.agent_instance.total_food) / self.agent_instance.total_food * 2

    capsules = self.agent_instance.get_capsules(next_state)    

    features["closest_capsule"] = 0

    if len(capsules) > 0:
      closest_cap = min([self.agent_instance.get_maze_distance((next_x, next_y), cap) for cap in capsules])
      # if closest_cap <= 5:
      #   features["closest_capsule"] = (6 - closest_cap) / 6
      features["closest_capsule"] = (self.agent_instance.max_distance - closest_cap) / self.agent_instance.max_distance
    


    features["dead_end"] = 0
    closest_way_home = 0
    if game_state.get_agent_state(self.agent_instance.index).is_pacman:
      # pomalo
      features["num_carring"] = game_state.get_agent_state(self.agent_instance.index).num_carrying / self.agent_instance.total_food
      middle = walls.width // 2 - 1 if  self.agent_instance.red else walls.width // 2
      closest_way_home = min([self.agent_instance.get_maze_distance((next_x, next_y), (middle, y)) for y in range(walls.height) if not game_state.has_wall(middle, y)])
      features["num_carring"] = features["num_carring"]  * closest_way_home / self.agent_instance.max_distance
  


      if self.agent_instance.dead_ends_depth[next_y][next_x] > 0:
        features["dead_end"] = self.agent_instance.dead_ends_depth[next_y][next_x] / self.agent_instance.max_dead_end 

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[game_state.get_agent_state(self.agent_instance.index).configuration.direction]
    if action == rev: features['reverse'] = 1


    if game_state.get_agent_state(self.agent_instance.index).is_pacman:
      features["defence"] = 0
    else:
      features["defence"] = 1
    
    old_game_state = self.agent_instance.get_previous_observation()
    if old_game_state is  not None:
      food_old = self.agent_instance.get_food_you_are_defending(old_game_state).as_list()
      food_new = self.agent_instance.get_food_you_are_defending(game_state).as_list()

      missing_food = list(set(food_old) - set(food_new))
      # print(missing_food)

      if len(missing_food) > 0:
        self.agent_instance.defence_ancor_point = missing_food[0]
    pacman_unknown_position= [a for a in enemies if a.is_pacman]
    if len(pacman_unknown_position) == 0 and   self.agent_instance.defence_ancor_point != self.agent_instance.default_defence_ancor_point1 and self.agent_instance.defence_ancor_point != self.agent_instance.default_defence_ancor_point2:
      dist_ap1 = self.agent_instance.get_maze_distance((next_x, next_y), self.agent_instance.default_defence_ancor_point1)
      dist_ap2 = self.agent_instance.get_maze_distance((next_x, next_y), self.agent_instance.default_defence_ancor_point2)
      if dist_ap1 < dist_ap2:
        self.agent_instance.defence_ancor_point = self.agent_instance.default_defence_ancor_point1
      else:
        self.agent_instance.defence_ancor_point = self.agent_instance.default_defence_ancor_point2

      # print( self.agent_instance.defence_ancor_point)

    # print(walls.width, walls.height)

    features["defence_ancor_point"] = self.agent_instance.get_maze_distance((next_x, next_y), (self.agent_instance.defence_ancor_point))
    index = self.agent_instance.index
    if index == 3 or index == 4:
      # DEFENSIVE AGENT
      if game_state.get_agent_state(index).scared_timer > 0:
        features["closest_enemy_pacman"] = 0
        features["closest_enemy_pacman_vertical"] = 0
        features["defence"] = 0
        features["defence_ancor_point"] = 0
        features["reverse"] = 0
      else:
        features["closest_enemy_pacman"] *= 40
        features["closest_enemy_pacman_vertical"] *= 40
        features["closest_capsule"] = 0
        features["closest-food"] = 0
        # features["closest_enemy_ghost"] = 0
        features["num_carring"] = 0
        features["dead_end"] = 0
        features["stop"] *= 3
    else:

      # OFFENSIVE AGENT
      if len(enemy_pacman) == 0 or  min([self.agent_instance.get_maze_distance((next_x, next_y), e) for e in enemy_pacman]) == 0:
        self.agent_instance.mode = "ATTACK"


      index_def = 3 if index == 1 else 2
      team_mate_position = game_state.get_agent_state(index_def).get_position()
      between_team_mate_distance = self.agent_instance.get_maze_distance((next_x, next_y), team_mate_position)

      if  len(enemy_pacman) > 0 and between_team_mate_distance > 5:
        closest_enemy = min([self.agent_instance.get_maze_distance((next_x, next_y), e) for e in enemy_pacman])
        closest_enemy_pacman_to_team_mate = min([self.agent_instance.get_maze_distance(team_mate_position, e) for e in enemy_pacman])
        if closest_enemy  <= 5 and closest_enemy_pacman_to_team_mate < 5:
          self.agent_instance.mode = "DEFFENSIVE"
      if closest_way_home + 3 >= game_state.data.timeleft / 4:
        self.agent_instance.mode = "TIMEOUT"

      # print(self.agent_instance.mode)
      if self.agent_instance.mode == "DEFFENSIVE":
        features["closest-food"] = 0
        features["closest_enemy_ghost"] = 0
        features["num_carring"] = 0
        features["dead_end"] = 0
        features["closest_enemy_pacman"] *= 40
        features["closest_enemy_pacman_vertical"] *= 40
      elif self.agent_instance.mode == "ATTACK":
        features["defence"] = 0
        features["defence_ancor_point"] = 0
        features["reverse"] = 0


      elif self.agent_instance.mode == "TIMEOUT":
        features["closest-food"] = 0
        features["num_carring"] = 0
        features["dead_end"] = 0
        features["closest_enemy_pacman"] = 0
        features["closest_enemy_pacman_vertical"]  = 0
        features["defence"] = 0
        features["defence_ancor_point"] = 0
        features["reverse"] = 0
        features["home_anchor_point"] = closest_way_home / self.agent_instance.max_distance
        
    return features

  def closest_food2(self, pos, food, walls, game_state):

    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
      pos_x, pos_y, dist = fringe.pop(0)
      if (pos_x, pos_y) in expanded:
        continue
      expanded.add((pos_x, pos_y))
      # if we find a food at this location then exit
      if food[pos_x][pos_y]:
        return dist
      # otherwise spread out from the location to its neighbours
      nbrs = Actions.get_legal_neighbors((pos_x, pos_y), walls)
      for nbr_x, nbr_y in nbrs:
        fringe.append((nbr_x, nbr_y, dist + 1))
    # no food found
    return None

# svetlosin
class Agent1(approx_q_learning_offense):
  def set_default_file(self):
      self.weights ={
            "closest-food": -1500,
            "bias": 100,
            "closest_enemy_ghost": -300,
            "closest_enemy_pacman": 2000,
            "closest_enemy_pacman_vertical": 500,
            "food_left": 0,
            "score": 0,
            "closest_capsule": -40,
            "num_carring": -100,
            "dead_end": -100,
            "defence": 500,
            "defence_ancor_point": -100,
            "stop": -150,
            "reverse": -50,
            "dead": 10000,
            "home_anchor_point": 0
        }

  

    
class Agent2(approx_q_learning_offense):
    def set_default_file(self):
      self.weights = {
      "closest-food": -2500,
      "bias": 0,
      "closest_enemy_ghost": -300,
      "closest_enemy_pacman": 100,
      "closest_enemy_pacman_vertical": 50,
      "food_left": 0,
      "score": 0,
      "closest_capsule": -40,
      "num_carring": -3500,
      "dead_end": -300,
      "defence": 500,
      "defence_ancor_point": 0,
      "stop": -50,
      "reverse": -5,
      "dead": 10000,
      "home_anchor_point": -2000
      }
