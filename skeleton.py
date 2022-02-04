import math
from operator import contains, itemgetter
import gym
import random
import requests
import numpy as np
import argparse
import sys
from timeit import default_timer as timer
from datetime import timedelta
from gym_connect_four import ConnectFourEnv

env: ConnectFourEnv = gym.make("ConnectFour-v0")

#SERVER_ADRESS = "http://localhost:8000/"
SERVER_ADRESS = "https://vilde.cs.lth.se/edap01-4inarow/"
API_KEY = 'nyckel'
STIL_ID = ["os7138ce-s"] 

# my code
#
def possible_moves(move, isPlayer):
   possiblemoves = []
   token = (1 if isPlayer else -1)
   for i in range(7):
      if move[0][i] == 0:
         for j in range(6):
            if move[j][i] != 0:
               nextmove = np.copy(move)
               nextmove[j-1][i] = token 
               possiblemoves.append((i, nextmove))
               break
            elif j == 5:
               nextmove = np.copy(move)
               nextmove[j][i] = token 
               possiblemoves.append((i, nextmove))
               break
   return possiblemoves 

def is_terminal(move):
   # test if full
   if len(possible_moves(move, True)) == 0:
      return True
   
   # taken from connect_four_env
   transposed = [list(i) for i in zip(*move)]
   flipped = np.fliplr(move)
   for i in range(6):
      for j in range(7 - 3):
            # test rows
            value = sum(move[i][j:j + 4])
            if abs(value) == 4:
               return True
   for i in range(6 - 3):
      for j in range(7):
            # test cols
            value = sum(transposed[j][i:i + 4])
            if abs(value) == 4:
               return True
   for i in range(6 - 3):
      for j in range(7 - 3):
            # test diagonal
            value = 0
            for k in range(4):
               value += move[i + k][j + k]
               if abs(value) == 4:
                  return True
            # test reverse diagonal
            value = 0
            for k in range(4):
               value += flipped[i + k][j + k]
               if abs(value) == 4:
                  return True

   return False

def eval_move(move):
   # test if full
   if len(possible_moves(move, True)) == 0:
      return 0
   
   # taken from connect_four_env
   evalsum = 0
   transposed = [list(i) for i in zip(*move)]
   flipped = np.fliplr(move)

   # test rows
   for i in range(6):
      for j in range(7 - 3):
            values = move[i][j:j + 4]
            value = sum(values)
            if abs(value) == 4:
               return math.copysign(1, value) * math.inf
            if not contains(values, -1):
               evalsum += math.pow(10, abs(value))
            elif not contains(values, 1):
               evalsum -= math.pow(10, abs(value))

   # test cols 
   for i in range(7):
      for j in range(6 - 3):
            values = transposed[i][j:j + 4]
            value = sum(values)
            if abs(value) == 4:
               return math.copysign(1, value) * math.inf
            if not contains(values, -1):
               evalsum += math.pow(10, abs(value))
            elif not contains(values, 1):
               evalsum -= math.pow(10, abs(value))

   # test diagonals
   for i in range(6 - 3):
      for j in range(7 - 3):
            # test forward diagonals
            values = []
            for k in range(4):
               values.append(move[i + k][j + k])
            value = sum(values)
            if abs(value) == 4:
               return math.copysign(1, value) * math.inf
            if not contains(values, -1):
               evalsum += math.pow(10, abs(value))
            elif not contains(values, 1):
               evalsum -= math.pow(10, abs(value))
            # test backward diagonals
            values = []
            for k in range(4):
               values.append(flipped[i + k][j + k])
            value = sum(values)
            if abs(value) == 4:
               return math.copysign(1, value) * math.inf
            if not contains(values, -1):
               evalsum += math.pow(10, abs(value))
            elif not contains(values, 1):
               evalsum -= math.pow(10, abs(value))
   
   return evalsum

def alpha_beta_pruning(move, depth, alpha, beta, isPlayer):
   if depth == 0 or is_terminal(move):
      return eval_move(move)
   possiblemoves = possible_moves(move, isPlayer)
   if isPlayer:
      value = -math.inf
      for (_, nextmove) in possiblemoves:
         value = max(value, alpha_beta_pruning(nextmove, depth - 1, alpha, beta, False))
         alpha = max(alpha, value)
         if value >= beta:
            break
      return value
   else:
      value = math.inf
      for (_, nextmove) in possiblemoves:
         value = min(value, alpha_beta_pruning(nextmove, depth - 1, alpha, beta, True))
         beta = min(beta, value)
         if value <= alpha:
            break
      return value

def call_server(move):
   res = requests.post(SERVER_ADRESS + "move",
                       data={
                           "stil_id": STIL_ID,
                           "move": move, # -1 signals the system to start a new game. any running game is counted as a loss
                           "api_key": API_KEY,
                       })
   # For safety some respose checking is done here
   if res.status_code != 200:
      print("Server gave a bad response, error code={}".format(res.status_code))
      exit()
   if not res.json()['status']:
      print("Server returned a bad status. Return message: ")
      print(res.json()['msg'])
      exit()
   return res

def check_stats():
   res = requests.post(SERVER_ADRESS + "stats",
                       data={
                           "stil_id": STIL_ID,
                           "api_key": API_KEY,
                       })

   stats = res.json()
   return stats

def opponents_move(env):
   env.change_player() # change to oppoent
   avmoves = env.available_moves()
   if not avmoves:
      env.change_player() # change back to student before returning
      return -1

   # TODO: Optional? change this to select actions with your policy too
   # that way you get way more interesting games, and you can see if starting
   # is enough to guarrantee a win
   action = random.choice(list(avmoves))

   state, reward, done, _ = env.step(action)
   if done:
      if reward == 1: # reward is always in current players view
         reward = -1
   env.change_player() # change back to student before returning
   return state, reward, done

def bluewave_move(state):
   start = timer()
   possibleMoves = possible_moves(state, True)
   
   if len(possibleMoves) == 1:
      return possibleMoves[0][0];
   
   moveEvaluations = []
   for (choice, move) in possibleMoves:
      moveEvaluations.append((choice, alpha_beta_pruning(move, 5, -math.inf, math.inf, False)))
   
   bestChoice = max(moveEvaluations, key=itemgetter(1))[0]
   
   for (choice, value) in moveEvaluations:
      prefix = " <" if choice == bestChoice else ""
      print("choice " + str(choice + 1) + ": " + str(value) + " points" + prefix)
   end = timer()
   print("eval. time: " + str(timedelta(seconds=end-start)))
   print()
   return bestChoice 

def to_emoji(token):
   if token == 1:
      return '🔵'
   elif token == -1:
      return '🔴'
   else:
      return '⬛'

def print_state(state):
   formatted = ["".join(list(map(to_emoji, row))) for row in state]
   for row in formatted:
      print(row)

def play_game(vs_server = False):
   # default state
   state = np.zeros((6, 7), dtype=int)
   currentRound = 0

   # setup new game
   if vs_server:
      # Start a new game
      res = call_server(-1) # -1 signals the system to start a new game. any running game is counted as a loss

      # This should tell you if you or the bot starts
      print(res.json()['msg'])
      botmove = res.json()['botmove']
      state = np.array(res.json()['state'])
   else:
      # reset game to starting state
      env.reset(board=None)
      # determine first player
      student_gets_move = random.choice([True, False])
      if student_gets_move:
         print('🌊 starts!')
         print()
      else:
         print('Bot starts!')
         print()

   # Print current gamestate
   print("Current state (Blue are 🌊's discs, Red are servers): ")
   print_state(state)
   print()

   done = False
   while not done:
      # Select your move
      currentRound += 1
      print("round " + str(currentRound));
      stmove = bluewave_move(state) 

      # make both student and bot/server moves
      if vs_server:
         # Send your move to server and get response
         res = call_server(stmove)
         print(res.json()['msg'])

         # Extract response values
         result = res.json()['result']
         botmove = res.json()['botmove']
         state = np.array(res.json()['state'])
      else:
         if student_gets_move:
            # Execute your move
            avmoves = env.available_moves()
            if stmove not in avmoves:
               print("🌊 tried to make an illegal move! 🌊 has lost the game.")
               break
            state, result, done, _ = env.step(stmove)

         student_gets_move = True # student only skips move first turn if bot starts

         # print or render state here if you like

         # select and make a move for the opponent, returned reward from students view
         if not done:
            state, result, done = opponents_move(env)

      # Check if the game is over
      if result != 0:
         done = True
         if not vs_server:
            print("Game over. ", end="")
         if result == 1:
            print("🌊 won!")
         elif result == 0.5:
            print("It's a draw!")
         elif result == -1:
            print("🌊 lost!")
         elif result == -10:
            print("🌊 made an illegal move and have lost!")
         else:
            print("Unexpected result result={}".format(result))
         if not vs_server:
            print("Final state (Blue are 🌊's discs, Red are servers): ")
      else:
         print("Current state (Blue are 🌊's discs, Red are servers): ")

      # Print current gamestate
      print_state(state)
      print()

def main():
   # Parse command line arguments
   parser = argparse.ArgumentParser()
   group = parser.add_mutually_exclusive_group()
   group.add_argument("-l", "--local", help = "Play locally", action="store_true")
   group.add_argument("-o", "--online", help = "Play online vs server", action="store_true")
   parser.add_argument("-s", "--stats", help = "Show 🌊's current online stats", action="store_true")
   args = parser.parse_args()

   # Print usage info if no arguments are given
   if len(sys.argv)==1:
      parser.print_help(sys.stderr)
      sys.exit(1)

   if args.local:
      play_game(vs_server = False)
   elif args.online:
      play_game(vs_server = True)

   if args.stats:
      stats = check_stats()
      print(stats)

if __name__ == "__main__":
    main()