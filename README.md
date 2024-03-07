# chessAI
ChessAI algorithm for intro to AI

How to use the  code:

#### 1: If you are planning to train the AI using MPI 

Ensure you use a machine with MPI installed, you can download an MPI through openMPI https://www.open-mpi.org/software/ompi/v4.1/

Use mpiexec -n 4 python mpi_ga.py in your console, where "4" is the number of threads you want the program to run on

**WARNING** this will take a while to run

You may stop the training after every time you get a "new generation" message,
as the best model gets saved at that point in model.pkl. You can stop it by entering an escape command lke ctrl+c

#### 2: If you want to play against the AI

Run the notebook titled "Load_and_play_AI.ipynb" this will only work on Windows machines and machines you have admin access

This will open a wb browser window on your local machine,
and will allow you to play against the 2 different AI's. GA move is chosen by the best model we found through training, and
stockfish move is a move chosen by stockfish.

To make your move, choose the desired piece's current location, e2, your desired location for that piece, e4,
and type them into the bar like so: e2e4. This is read by the program as "e2 to e4"

#### 3: If you would like to inspect the code

"Mpi_ga.py" holds all genetic algorithm ecosystem code and generational logic

"simulate_and_evaluate.py" contains all Organism, neural network, and simulation logic

"Intro_to_AI_NB.ipynb" contains an early version of the code without MPI logic
Testing some repo stuff
