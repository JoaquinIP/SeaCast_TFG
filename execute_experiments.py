#from argparse import ArgumentParser
import os
import subprocess

if __name__ == "__main__":

    experiments = [
        [
            {"command": "cd", "args": ["./src/seacast_tools/"]},
            {"command": "python", "args": ["create_non_uniform_mesh.py", "--dataset", "atlantic", "--plot", "0", "--mesh_type", "uniform", "--levels", "3", "--crossing_edges", "0", "--uniform_resolution_list", "27, 9, 3", "--n_connections", "1", "--k_neighboors", "1"]},
            {"command": "python", "args": ["move_files.py", "--graph_type", "hierarchical", "--graph", "uniform"]},
            {"command": "cd", "args": ["../.."]},
            {"command": "python", "args": ["train_model.py", "--dataset", "atlantic", "--epochs", "100", "--n_workers", "4", "--batch_size", "1", "--step_length", "1", "--ar_steps", "1", "--lr", "0.001", "--optimizer", "adamw", "--scheduler", "cosine", "--finetune_start", "1", "--model", "hi_lam", "--graph", "hierarchical", "--processor_layers", "4", "--hidden_dim", "128", "--n_nodes", "4", "--custom_run_name", "non_crossig_edges_27_9_3"]}
        ],
        [
            {"command": "cd", "args": ["./src/seacast_tools/"]},
            {"command": "python", "args": ["create_non_uniform_mesh.py", "--dataset", "atlantic", "--plot", "0", "--mesh_type", "uniform", "--levels", "3", "--crossing_edges", "0", "--uniform_resolution_list", "45, 15, 5", "--n_connections", "1", "--k_neighboors", "1"]},
            {"command": "python", "args": ["move_files.py", "--graph_type", "hierarchical", "--graph", "uniform"]},
            {"command": "cd", "args": ["../.."]},
            {"command": "python", "args": ["train_model.py", "--dataset", "atlantic", "--epochs", "100", "--n_workers", "4", "--batch_size", "1", "--step_length", "1", "--ar_steps", "1", "--lr", "0.001", "--optimizer", "adamw", "--scheduler", "cosine", "--finetune_start", "1", "--model", "hi_lam", "--graph", "hierarchical", "--processor_layers", "4", "--hidden_dim", "128", "--n_nodes", "4", "--custom_run_name", "non_crossig_edges_45_15_5"]}  
        ],
        [
            {"command": "cd", "args": ["./src/seacast_tools/"]},
            {"command": "python", "args": ["create_non_uniform_mesh.py", "--dataset", "atlantic", "--plot", "0", "--mesh_type", "uniform", "--levels", "3", "--crossing_edges", "0", "--uniform_resolution_list", "63, 21, 7", "--n_connections", "1", "--k_neighboors", "1"]},
            {"command": "python", "args": ["move_files.py", "--graph_type", "hierarchical", "--graph", "uniform"]},
            {"command": "cd", "args": ["../.."]},
            {"command": "python", "args": ["train_model.py", "--dataset", "atlantic", "--epochs", "100", "--n_workers", "4", "--batch_size", "1", "--step_length", "1", "--ar_steps", "1", "--lr", "0.001", "--optimizer", "adamw", "--scheduler", "cosine", "--finetune_start", "1", "--model", "hi_lam", "--graph", "hierarchical", "--processor_layers", "4", "--hidden_dim", "128", "--n_nodes", "4", "--custom_run_name", "non_crossig_edges_63_21_7"]}  
        ],
        [
            {"command": "cd", "args": ["./src/seacast_tools/"]},
            {"command": "python", "args": ["create_non_uniform_mesh.py", "--dataset", "atlantic", "--plot", "0", "--mesh_type", "uniform", "--levels", "3", "--crossing_edges", "0", "--uniform_resolution_list", "81, 27, 9", "--n_connections", "1", "--k_neighboors", "1"]},
            {"command": "python", "args": ["move_files.py", "--graph_type", "hierarchical", "--graph", "uniform"]},
            {"command": "cd", "args": ["../.."]},
            {"command": "python", "args": ["train_model.py", "--dataset", "atlantic", "--epochs", "100", "--n_workers", "4", "--batch_size", "1", "--step_length", "1", "--ar_steps", "1", "--lr", "0.001", "--optimizer", "adamw", "--scheduler", "cosine", "--finetune_start", "1", "--model", "hi_lam", "--graph", "hierarchical", "--processor_layers", "4", "--hidden_dim", "128", "--n_nodes", "4", "--custom_run_name", "non_crossig_edges_81_27_9"]}  
        ],
        [
            {"command": "cd", "args": ["./src/seacast_tools/"]},
            {"command": "python", "args": ["create_non_uniform_mesh.py", "--dataset", "atlantic", "--plot", "0", "--mesh_type", "uniform", "--levels", "3", "--crossing_edges", "1", "--uniform_resolution_list", "81, 27, 9", "--n_connections", "1", "--k_neighboors", "1"]},
            {"command": "python", "args": ["move_files.py", "--graph_type", "hierarchical", "--graph", "uniform"]},
            {"command": "cd", "args": ["../.."]},
            {"command": "python", "args": ["train_model.py", "--dataset", "atlantic", "--epochs", "100", "--n_workers", "4", "--batch_size", "1", "--step_length", "1", "--ar_steps", "1", "--lr", "0.001", "--optimizer", "adamw", "--scheduler", "cosine", "--finetune_start", "1", "--model", "hi_lam", "--graph", "hierarchical", "--processor_layers", "4", "--hidden_dim", "128", "--n_nodes", "4", "--custom_run_name", "crossig_edges_81_27_9"]}  
        ]
    ]

    cwd = os.getcwd()

    for commands in experiments:
        for step in commands:
            cmd = step["command"]
            args = step["args"]
            print(f"\nExecuting: {cmd} {' '.join(args)}")
            if cmd == "cd":
                # Cambiar directorio
                path = args[0]
                cwd = os.path.abspath(os.path.join(cwd, path))
                print(f"\nDirectory changed to: {cwd}")
            else:
                # Ejecutar comando
                command = [cmd] + args
                print(f"\nEExecuting in directory: {cwd} the command: {' '.join(command)}")

                result = subprocess.run(command, cwd=cwd, text=True)

                if result.returncode != 0:
                    print("Error executing command:", " ".join(command))
                    print("Output:", result.stdout)
                    print("Error:", result.stderr)
                    break

        print("\nAll commands executed successfully in this step. Next Experiment.")
    print("\nAll experiments executed successfully.")