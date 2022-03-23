# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    
    #score begins at 0
    score = 0
    #relative score
    score += successorGameState.getScore() - currentGameState.getScore()


    #check if successor is Win
    if successorGameState.isWin():
      return 10000


    #punish for stopping
    if action == Directions.STOP:
      score -= 10
  

    #get distance from successor position to all foods
    food = newFood.asList()
    food_dist = [0]
    for position in food:
      food_dist.append(util.manhattanDistance(newPos, position))

    curr_food = currentGameState.getFood().asList()
    food_dist_curr = [0]
    for curr_pos in curr_food:
      food_dist_curr.append(util.manhattanDistance(newPos, curr_pos))


    #successor: get positions of ghosts
    ghost_pos = []
    for ghost in newGhostStates:
      ghost_pos.append(ghost.getPosition())

    food_left = len(food)
    food_left_curr = len(curr_food)


    #punish for more food left
    score -= (10*food_left)

    #reward for less food in successor state
    if food_left < food_left_curr:
      score += 200

    #reward for min distance from food getting smaller
    if min(food_dist) > 0 and min(food_dist_curr) > 0:
      if (1/(min(food_dist)) < 1/(min(food_dist_curr))):
        score += 500
      else:
        score -= 500
    else:
      if (min(food_dist) < min(food_dist_curr)):
        score += 500
      else:
        score -= 500
  
    #get dist from successor position to ghosts
    ghost_dist = []
    for position in ghost_pos:
      ghost_dist.append(util.manhattanDistance(newPos, position))


    #current: get positions of ghosts
    ghost_pos_curr  = []
    for ghost in currentGameState.getGhostStates():
      ghost_pos_curr.append(ghost.getPosition())

    ghost_dist_curr = []
    for position in ghost_pos_curr:
      ghost_dist_curr.append(util.manhattanDistance(newPos, position))

    total_scared = sum(newScaredTimes)

    if total_scared > 0:
      #move toward scared ghosts
      if (min(ghost_dist_curr) > 0 and min(ghost_dist) > 0):
        if (1/(min(ghost_dist_curr)) < 1/(min(ghost_dist))):
          score += 200
        else:
          score -= 100
      else:
        if (min(ghost_dist_curr) < min(ghost_dist)):
          score += 200
        else:
          score -= 100
    else:
      if (min(ghost_dist_curr) > 0 and min(ghost_dist) > 0):
        if (1/(min(ghost_dist_curr)) < 1/(min(ghost_dist))):
          score -= 50
        else:
          score += 100
      else:
        if (min(ghost_dist_curr) < min(ghost_dist)):
          score -= 50
        else:
          score += 100


    #increase score if next state is big food
    bigFood = len(successorGameState.getCapsules())
    if bigFood > 0:
      print bigFood
    if newPos in currentGameState.getCapsules():
      score += 100*bigFood



    return score
   # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"

    num_ghosts = gameState.getNumAgents()-1

    def maximum(gameState, depth): #pacman at max levels
      maxval = float('-inf')
      depth_curr = depth + 1
      if gameState.isWin() or gameState.isLose() or self.depth == depth_curr:
        #terminate
        return self.evaluationFunction(gameState)

      actions = gameState.getLegalActions(0)
      for action in actions:
        succ = gameState.generateSuccessor(0, action)
        maxval = max(maxval, minimum(succ, depth_curr, 1))
      return maxval


    def minimum(gameState, depth, agentIndex): #ghost at all min levels
      minval = float('inf')
      if gameState.isWin() or gameState.isLose():
        #terminate
        return self.evaluationFunction(gameState)

      actions = gameState.getLegalActions(agentIndex)
      
      for action in actions:
        succ = gameState.generateSuccessor(agentIndex, action)
        
        if agentIndex == gameState.getNumAgents()-1: #next agent will be pacman
          minval = min(minval, maximum(succ, depth))
        else:
          minval = min(minval, minimum(succ, depth, agentIndex+1))

      return minval


    actions = gameState.getLegalActions(0)
    curr_score = float('-inf')

    for action in actions:
      nextState = gameState.generateSuccessor(0, action)
      score = minimum(nextState, 0, 1)
      
      if score > curr_score:
        bestAction = action
        curr_score = score
        
    return bestAction



   #  util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"

    def maximum(gameState, depth, alpha, beta):
      depth_curr = depth + 1
      if gameState.isWin() or gameState.isLose() or self.depth == depth_curr:
        return self.evaluationFunction(gameState)

      maxval = float('-inf')
      actions = gameState.getLegalActions(0)
      alpha1 = alpha

      for action in actions:
        succ = gameState.generateSuccessor(0, action)
        maxval = max(maxval, minimum(succ, depth_curr, 1, alpha1, beta))
        if maxval > beta:
          return maxval
        alpha1 = max(alpha1, maxval)

      return maxval

    
    def minimum(gameState, depth, agentIndex, alpha, beta):
      
      if gameState.isWin() or gameState.isLose():
        #terminate
        return self.evaluationFunction(gameState)

      minval = float('inf')
      actions = gameState.getLegalActions(agentIndex)
      beta1 = beta

      for action in actions:
        succ = gameState.generateSuccessor(agentIndex, action)
        if agentIndex == gameState.getNumAgents()-1:
          minval = min(minval, maximum(succ, depth, alpha, beta1))
          if minval < alpha:
            return minval
          beta1 = min(beta1, minval)
        else:
          minval = min(minval, minimum(succ, depth, agentIndex+1, alpha, beta1))
          if minval < alpha:
            return minval
          beta1 = min(beta1, minval)

      return minval



    actions = gameState.getLegalActions(0)
    curr_score = float('-inf')

    alpha = float('-inf')
    beta = float('inf')
    best_action = actions[0] #default value

    for action in actions:
      nextState = gameState.generateSuccessor(0, action)
      score = minimum(nextState, 0, 1, alpha, beta)

      if score > curr_score: #update values
        bestAction = action
        curr_score = score

      if score > beta:
        return best_action

      alpha = max(alpha, score)

    return bestAction


#     util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"

    def maximum(gameState, depth):
      depth_curr = depth+1
      if gameState.isWin() or gameState.isLose() or self.depth == depth_curr:
        return self.evaluationFunction(gameState)

      maxval = float('-inf')
      actions = gameState.getLegalActions(0)

      for action in actions:
        succ = gameState.generateSuccessor(0, action)
        maxval = max(maxval, expected(succ, depth_curr, 1))
      return maxval

    def expected(gameState, depth, agentIndex):
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
      actions = gameState.getLegalActions(agentIndex)
      expected_sum = 0

      for action in actions:
        succ = gameState.generateSuccessor(agentIndex, action)
        if agentIndex == gameState.getNumAgents()-1:
          e_val = maximum(succ, depth)
        else:
          e_val = expected(succ, depth, agentIndex+1)
        expected_sum += e_val

      if len(actions) == 0:
        return 0

      return float(expected_sum/len(actions))


    actions = gameState.getLegalActions(0)
    curr_score = float('-inf')
    best_action = actions[0]
    
    for action in actions:
      nextState = gameState.generateSuccessor(0, action)
      score = expected(nextState, 0, 1)
      
      if score > curr_score:
        best_action = action
        curr_score = score
    return best_action



#     util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  
  #initial stuff:
  # successorGameState = currentGameState.generatePacmanSuccessor(action)
  newPos = currentGameState.getPacmanPosition()
  newFood = currentGameState.getFood()
  newGhostStates = currentGameState.getGhostStates()
  newScaredTime = [ghostState.scaredTimer for ghostState in newGhostStates]


  #get important states
    #get distance from successor position to all foods
  food = newFood.asList()
  food_dist = [0]
  for position in food:
    food_dist.append(util.manhattanDistance(newPos, position))
  curr_food = currentGameState.getFood().asList()
  food_dist_curr = [0]
  for curr_pos in curr_food:
    food_dist_curr.append(util.manhattanDistance(newPos, curr_pos))

    #get distance from current position to regular, scared ghosts
  ghost_pos_curr = []
  scared_ghost_pos_curr = []
  for ghost in currentGameState.getGhostStates():
    if ghost.scaredTimer:
      scared_ghost_pos_curr.append(ghost.getPosition())
    else:
      ghost_pos_curr.append(ghost.getPosition())
  ghost_dist_curr = []
  scared_ghost_dist_curr = []
  for position in ghost_pos_curr:
    ghost_dist_curr.append(util.manhattanDistance(newPos, position))
  for scposition in scared_ghost_pos_curr:
    scared_ghost_dist_curr.append(util.manhattanDistance(newPos, position))

    #get number of regular food pellets left
  food_left = len(food)
  food_left_curr = len(curr_food)

    #get number of big pellets left
  bigFood_left = len(currentGameState.getCapsules())


  #evaluation function
  currScore = scoreEvaluationFunction(currentGameState)
  FoodDistances = sum(food_dist_curr)
  GhostDistances = sum(ghost_dist_curr)
  ScaredGhostDistances = sum(scared_ghost_dist_curr)
  FoodLeft = food_left_curr
  BigFoodLeft = bigFood_left

  if sum(food_dist_curr) > 0:
    FoodDistances = 1/sum(food_dist_curr)
  if sum(ghost_dist_curr) > 0:
    GhostDistances = 1/sum(ghost_dist_curr)
  if sum(scared_ghost_dist_curr) > 0:
    ScaredGhostDistances = 1/sum(scared_ghost_dist_curr)
    
  score = currScore - FoodDistances - GhostDistances - ScaredGhostDistances - FoodLeft - 5*BigFoodLeft

  return score

#  util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

