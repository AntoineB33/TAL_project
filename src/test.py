import subprocess

# List of terminal commands
commands = [
    "ls",              # Example: list directory contents
    "pwd",             # Example: print working directory
    "echo 'Hello World'" # Example: print Hello World
]

# Function to execute the commands
def execute_commands(commands):
    for command in commands:
        try:
            # Execute each command
            result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
            print(f"Command: {command}")
            print(f"Output: {result.stdout}")
            if result.stderr:
                print(f"Error: {result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing {command}: {e}")

# Call the function with the list of commands
execute_commands(commands)
