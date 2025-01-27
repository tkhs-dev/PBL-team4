import argparse
import subprocess
import re
from collections import defaultdict
from time import sleep


def run_battlesnake_test(cli_path, gym_path, local_path, level_port, games_per_level=20):
    results = defaultdict(int)  # Dictionary to store win/loss/draw counts

    for level, port in level_port.items():
        print(f"Starting tests for Level {level} on port {port}...")

        for i in range(games_per_level):
            try:
                # Run the Battlesnake game using the specified command
                command = [
                    f"{cli_path}/battlesnake", "play",
                    "--name", "you", "--url", f"{local_path}",
                    "--name", "opponent", "--url", f"{gym_path}:{port}",
                    "-t", "1000"
                ]

                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                # Analyze the result output
                output = result.stderr
                if re.search(r"you was the winner", output):
                    results[f"Level {level} Wins"] += 1
                elif re.search(r"It was a draw", output):
                    results[f"Level {level} Draws"] += 1
                else:
                    results[f"Level {level} Losses"] += 1

                print(f"Game {i + 1}/{games_per_level} for Level {level} completed.")
                sleep(0.5)

            except Exception as e:
                print(f"An error occurred during testing on Level {level}: {e}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cli-path', type=str, default='../../rules')
    parser.add_argument('-g', '--games', type=int, default=20)
    parser.add_argument('-l', '--local-path', type=str, default='http://127.0.0.1:8000')
    parser.add_argument('-p', '--gym-path', type=str, default='http://pbl.ics.es.osaka-u.ac.jp')
    args = parser.parse_args()

    # Define ports and levels
    level_port = {
        1: 8001
    }

    # Run the tests and collect results
    results = run_battlesnake_test(args.cli_path, args.gym_path, args.local_path, level_port, args.games)

    # Print the results
    print("\nBattlesnake Performance Test Results:")
    for key, value in results.items():
        print(f"{key}: {value}")
