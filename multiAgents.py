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
    currPos = currentGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newBigFood = successorGameState.getCapsules()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    
    #get distance to foods
    food = newFood.asList()
    food_dist = [0]
    for position in food:
      food_dist.append(util.manhattanDistance(newPos, position))

    curr_food = currentGameState.getFood().asList()
    curr_food_dist = [0]
    for position in curr_food:
      curr_food_dist.append(util.manhattanDistance(currPos, position))

    food_left = len(food_dist)
    curr_food_left = len(curr_food_dist)


    #distance to big foods
    big_food = newBigFood
    big_food_dist = [0]
    for position in big_food:
      big_food_dist.append(util.manhattanDistance(newPos, position))

    curr_big_food = currentGameState.getCapsules()
    curr_big_food_dist = [0]
    for position in curr_big_food:
      curr_big_food_dist.append(util.manhattanDistance(currPos, position))

    #distance to ghosts
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


    #number of regular food pellets
    food_left = len(food)
    #number of big food pellets
    big_food_left = len(big_food)


    #evaluation function
    currScore = scoreEvaluationFunction(currentGameState)
    # FoodDistances = sum(food_dist)
    FoodDistances = sum(curr_food_dist)
    BigFoodDistances = sum(curr_big_food_dist)
    GhostDistances = sum(ghost_dist_curr)
    ScaredGhostDistances = sum(scared_ghost_dist_curr)
    FoodLeft = curr_food_left

    if action == Directions.STOP:
      currScore -= 10

    ClosestFood = min(curr_food_dist)
    if ClosestFood != 0:
      ClosestFoodRecip = 1.0/ClosestFood
    else:
      ClosestFoodRecip = 0
    
    ClosestGhost = min(ghost_dist_curr)

    # score = currScore - FoodDistances - 3*BigFoodDistances + GhostDistances - 2*ScaredGhostDistances
    # score = currScore + GhostDistances*10/FoodDistances
    score = currScore + 2*ClosestFoodRecip - ClosestGhost

    



    # NEW TRY
    Ghosts = successorGameState.getGhostPositions()
    #find dist to ghosts
    for ghost in Ghosts:
      ghost_dist = manhattanDistance(ghost, newPos)
      if ghost_dist < 2:
        return float('-inf')

    closest_food = float('inf')
    
    for foodItem in food:
      food_dist = manhattanDistance(foodItem, newPos)
      closest_food = min(closest_food, food_dist)
      

    initScore = successorGameState.getScore()
    return initScore + 1.0/closest_food


    # return score

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
  newBigFood = currentGameState.getCapsules()
  newGhostStates = currentGameState.getGhostStates()
  newScaredTime = [ghostState.scaredTimer for ghostState in newGhostStates]

  #succesor game state
  

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

    #distance from successor position to big foods
  big_food = newBigFood
  big_food_dist = [0]
  for position in big_food:
    big_food_dist.append(util.manhattanDistance(newPos, position))
  curr_big_food = currentGameState.getCapsules()
  big_food_dist_curr = [0]
  for curr_pos in curr_big_food:
    big_food_dist_curr.append(util.manhattanDistance(newPos, curr_pos))


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
  bigFood_left = len(big_food)
  bigFood_left_curr = len(curr_big_food)


  #evaluation function
  currScore = scoreEvaluationFunction(currentGameState)
  FoodDistances = sum(food_dist_curr)
  BigFoodDistances = sum(big_food_dist_curr)
  GhostDistances = sum(ghost_dist_curr)
  ScaredGhostDistances = sum(scared_ghost_dist_curr)
  FoodLeft = food_left_curr
  BigFoodLeft = bigFood_left_curr  #bigFood_left
  # NextFoodDistances 


#  if sum(food_dist_curr) > 0:
#    FoodDistances = 1/sum(food_dist_curr)
#  if sum(ghost_dist_curr) > 0:
#    GhostDistances = 1/sum(ghost_dist_curr)
#  if sum(scared_ghost_dist_curr) > 0:
#    ScaredGhostDistances = 1/sum(scared_ghost_dist_curr)
    
  score = currScore - FoodDistances - 3*BigFoodDistances + GhostDistances \
- 2*ScaredGhostDistances - FoodLeft - 5*BigFoodLeft

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

