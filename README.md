# SCITAS Kuma User Guide

This guide explains how to use the **SCITAS Kuma GPU cluster** ([Spec](https://scitas-doc.epfl.ch/supercomputers/kuma/)) with:  
- Python & [Micromamba](https://mamba.readthedocs.io/en/latest/index.html) (faster alternative for conda)  
- VS code Remote Window
- PyTorch + CUDA  


## Table of Contents
1. [Connect to the EPFL Network](#1-connect-to-the-epfl-network)
2. [Join the HPC-LAPD Group](#2-join-the-hpc-lapd-group)
3. [SSH Access to Kuma](#3-ssh-access-to-kuma)
4. [File System on Kuma](#4-file-system-on-kuma)
5. [Install Micromamba & Python](#5-install-micromamba--python)
6. [Set Up VS Code and run PyTorch (no GPU yet)](#6-set-up-vs-code-and-run-pytorch-no-gpu-yet)
7. [Set Up Passwordless SSH](#7-set-up-passwordless-ssh)
8. [Interactive GPU Access in VS Code](#8-interactive-gpu-access-in-vs-code)
9. [Running Jobs with GPU on Kuma](#9-running-jobs-with-gpu-on-kuma)

---

## 1. Connect to the EPFL Network
Ensure you are on the **EPFL network** or connected via **VPN** to access Kuma.

## 2. Join the HPC-LAPD Group
1. Visit [EPFL Groups](https://groups.epfl.ch)
2. Search for `hpc-lapd`.
3. If you are not a member, contact the administrators.  
<img src="https://i.imgur.com/ql1qCc4.png" width=50%>  
<img src="https://i.imgur.com/OdVTrx0.png" width=50%>


## 3. SSH Access to Kuma
1. Open **PowerShell** (Windows) or **Terminal** (Linux/macOS).
2. Run: `ssh <username>@kuma.hpc.epfl.ch`
   - Example for Leo Jih-Liang Hsieh: `ssh jlhsieh@kuma.hpc.epfl.ch`
3. Enter your password (characters will not be displayed as you type).
4. You are now in the Kuma frontend (`kuma1` or `kuma2`), but GPU access is not yet available.  
<img src="https://i.imgur.com/59m2486.png" width=70%>

## 4. File System on Kuma
Kuma has different [storage locations](https://scitas-doc.epfl.ch/storage/file-system/):

### **Home Directory (`/home/<username>`)**
- Limited to **100 GB per user**.
- Data is deleted **after 2 years of inactivity** or **6 months after leaving EPFL**.
- Useful commands:
  ```sh
  pwd                  # Show current directory
  cd /home/<username>  # Change to home directory (or `cd ~`)
  ls                   # List file/folder with details (or `ls -l`, `ll`, `ls -la`, `ll -a`)
  du -sh <file/folder> # Check disk usage
  ```

### **Scratch Directory (`/scratch/<username>`)**
- Shared **435 TB** of high-speed storage.
- Suitable for computation-heavy tasks.
- **Files older than 30 days are automatically deleted**.

## 5. Install Micromamba & Python
[Micromamba](https://mamba.readthedocs.io/en/latest/index.html) is a fast alternative to Conda. Follow these steps to install it in your `scratch` directory:

1. Install Micromamba:
   ```sh
   "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
   ```
   - Binary folder: `/scratch/<username>/.local/bin`
   - Shell initialization: `y`
   - Configure conda-forge: `y`
   - Installation prefix: `/scratch/<username>/micromamba`

2. Activate Micromamba:
   ```sh
   source ~/.bashrc
   micromamba activate
   ```
   - You should see `(base)` in your terminal.  
    <img src="https://i.imgur.com/NhKr2x9.png" width=80%>  


3. Create a Python environment:
   ```sh
   micromamba create -n my_env python=3.13
   micromamba activate my_env
   which python  # Note this path for VS Code (Example: `/scratch/jlhsieh/micromamba/envs/my_env/bin/python`)
   ```

4. Install packages:  
   - [SCITAS Lmod tool](https://scitas-doc.epfl.ch/user-guide/using-clusters/software-stack/#modules-and-lmod)  Tool for managing scientific software.  
   - [uv](https://astral.sh/blog/uv)  Extremely fast. Leo prefer `uv pip install` over `pip install`, `conda install`, `micromaba install`.  
   - [PyTorch](https://pytorch.org/get-started/locally/)  Use the PyTorch that match the CDUA on Kuma.  
   ```sh
   pip install uv  # an extremely fast Python package installer
   module load gcc/13.2.0 cuda/12.4.1  # SCITAS Lmod tool
   module list  # Check what tool have been load
   nvcc --version  # Check if there is CUDA
   uv pip install torch torchvision torchaudio  # Check PyTorch website to match CDUA version if needed
   uv pip install ipykernel  # For Jupyter
   ```  
    <img src="https://i.imgur.com/b3M5UyW.png" width=60%>  


## 6. Set Up VS Code and run PyTorch (no GPU yet)
1. Open **VS Code** on your local computer.
2. Click the blue **Open a Remote Window** button (bottom left corner).
3. Click **Connect to Host** > **Add New SSH Host**.
4. Enter: `ssh <username>@kuma.hpc.epfl.ch`.
5. Choose the current user SSH configuration file.  
(Example: `/home/leohsieh/.ssh/config` or `C:\Users\leohsieh\.ssh\config`)
6. Rename the `Host` in SSH config file.  
7. Connect VS code to Remote Host.  

https://github.com/user-attachments/assets/44657123-58f3-4fba-a89f-e4254a20f454

8. Open `/scratch/<username>` folder in VS Code.  
9. Create and test a Python script (`test.py`) to check PyTorch.  
```python
    # %% test.py
    import torch
    import sys

    def check_gpu() -> None:
        print(f"Python version: {sys.version}")
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print("CUDA is available. Here are the details of the CUDA devices:")
            for i in range(torch.cuda.device_count()):
                print(f"Device {i}: {torch.cuda.get_device_name(i)}")
                a = f"  CUDA Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor},"
                b = f"  Multiprocessors: {torch.cuda.get_device_properties(i).multi_processor_count}"
                print(f"{a+b}")
                print(f"  Memory")
                print(f"    {torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):.2f} GB: Total Memory")
                print(f"    {torch.cuda.memory_reserved(i) / (1024 ** 3):.2f} GB: PyTorch current Reserved Memory")
                print(f"    {torch.cuda.memory_allocated(i) / (1024 ** 3):.2f} GB: PyTorch current Allocated Memory")
                print(f"    {torch.cuda.max_memory_reserved(i) / (1024 ** 3):.2f} GB: PyTorch max ever Reserved Memory")
                print(f"    {torch.cuda.max_memory_allocated(i) / (1024 ** 3):.2f} GB: PyTorch max ever Allocated Memory")
        else:
            print("CUDA is NOT available")

    check_gpu()
```

https://github.com/user-attachments/assets/ef5d3385-f6e8-4b45-a39f-1c43203c9745


10. If your Python interpreter in Micromamba is not detected, enter the Python path manually.  
   Example: `/scratch/jlhsieh/micromamba/envs/my_env/bin/python`  
11. Try running `test.py`. You should see `PyTorch version: 2.6.0+cu124`.  
   - (No GPU access yet. You'll see `CUDA is NOT available`).

   https://github.com/user-attachments/assets/e948b0a3-20e0-4455-aa1e-9e92d43a8bdf



## 7. Set Up Passwordless SSH
**This is required** because we need ProxyJump to GPU node later.  
1. Open **PowerShell** (Windows) or **Terminal** (Linux/macOS).
2. Generate an SSH key pair in your local computer:  
   - Both Linux/macOS and Windows PowerShell  
      ```sh
      ssh-keygen -t ed25519 -f ${HOME}/.ssh/my_ssh_key
      ```  
      <img src="https://i.imgur.com/cFMRLZF.png" width=80%>  

3. Copy the public key to Kuma:  
   - Linux/macOS
      ```sh
      ssh-copy-id -i ${HOME}/.ssh/my_ssh_key.pub <username>@kuma.hpc.epfl.ch`
      ```
   - Windows PowerShell  
      ```sh
      type $HOME\.ssh\my_ssh_key.pub | ssh <username>@kuma.hpc.epfl.ch "mkdir -p .ssh && tee -a .ssh/authorized_keys"
      ```

4. Modify the SSH config file as:
   ```
   Host my-kuma-frontend
     HostName kuma.hpc.epfl.ch
     User jlhsieh
     IdentityFile ~/.ssh/my_ssh_key
     IdentitiesOnly yes

   Host my-kuma-node
     HostName kh???
     User jlhsieh
     IdentityFile ~/.ssh/my_ssh_key
     IdentitiesOnly yes
     ProxyJump my-kuma-frontend
   ```  
   <img src="https://i.imgur.com/hXFheou.png" width=50%>  

5. Test the connection without password:
   ```sh
   ssh my-kuma-frontend
   ```
   - If successful, no password is required.


## 8. Interactive GPU Access in VS Code
Use interactive sessions for testing/debugging:

1. Open **PowerShell** (Windows) or **Terminal** (Linux/macOS) and connect to `my-kuma-frontend`:
   ```sh
   ssh my-kuma-frontend
   micromamba activate my_env
   module load gcc/13.2.0 cuda/12.4.1
   ```

2. Request an interactive GPU node:
   ```sh
   Sinteract -p h100 -q debug -m 4G -g gpu:1 -t 0-00:10:00
   ```
   - `-p h100` (Use H100 GPU) or `-p l40s` (Use L40S GPU)
   - `-q debug` (free but 1 hour max) or `-q normal` or `-q long`  # See [QOS detail](https://scitas-doc.epfl.ch/user-guide/using-clusters/slurm-qos-partitions/#kuma-the-gpu-cluster) and [Kuma Pricing & QOS](https://scitas-doc.epfl.ch/blog/2024/10/17/kuma-cluster-full-production--pricing--nov-1st/).  
   - `-m 4G` (4 GB RAM)  
   - `-g gpu:1` (1 GPU) or `-g gpu:2` (2 GPUs)
   - `-t 0-00:30:00` (Time duration format: `D-HH:MM:SS`)
   - For more details: `Sinteract --help` or [link](https://scitas-doc.epfl.ch/user-guide/using-clusters/running-jobs/#sinteract)

3. A Kuma GPU node will be assigned (Example: `kh029`).  
   <img src="https://i.imgur.com/4Qw2bGY.png" width=80%>  

4. Modify the SSH config accordingly, connect to the GPU node in VS Code, and start using the GPU.  
   https://github.com/user-attachments/assets/30150cce-c39e-49b3-9236-986e57b2fcc7
   - The GPU node will close when the time is up or when the Terminal/PowerShell is closed.





## 9. Running Jobs with GPU on Kuma
For long-running jobs, submit batch scripts instead of using interactive mode. This way, you don't need to keep your Terminal/PowerShell open.

1. connect to the Kuma frontend node in VS Code (GPU node not needed)

2. Create `myjob.run` in `/scratch/<username>`:
   ```sh
   #!/bin/bash
   #SBATCH --partition h100
   #SBATCH --qos debug
   #SBATCH --mem 4G
   #SBATCH --gpus 1
   #SBATCH --time 0-00:01:00

   echo "==== Start job ==================================="
   module load gcc/13.2.0 cuda/12.4.1
   micromamba run -n my_env python /scratch/jlhsieh/test.py
   echo "sleep 10 seconds"
   sleep 10
   micromamba run -n my_env python /scratch/jlhsieh/test.py
   echo "==== End job====================================="
   ```
   - For more details: [link](https://scitas-doc.epfl.ch/user-guide/using-clusters/running-jobs/#running-jobs-with-slurm)


3. Open a terminal in VS Code and submit the job:
   ```sh
   sbatch myjob.run
   ```
4. Check job status:
   ```sh
   Squeue
   ```  
   https://github.com/user-attachments/assets/5dc3ef84-ce4c-4eb4-ae77-f50414e109a8


5. You can overwrite job parameters when submitting:
   ```sh
   sbatch --partition=l40s --qos normal --mem=8G --gpus=2 --time=0-00:05:00 myjob.run  # parameters in `myjob.run` will be overwritten
   ```

6. Now your job runs independently, even after you **disconnect from Kuma**.

