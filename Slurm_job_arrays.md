# Research Computing Training

[![drawing](images/ac-grid-5-red.svg)](images/ac-grid-5-red.svg)

## Presentor

Your Name Here

Title

[Research Computing](https://rc.northeastern.edu/research-computing-team/)

## Running Many Jobs at Once with Slurm Job Arrays

Welcome to the [Research Computing Training Series](https://rc.northeastern.edu/training/)! In this training we will learn how to use Slurm job arrays on the Explorer cluster to run many similar jobs from a single `sbatch` script. Job arrays are one of the easiest ways to scale up your work, whether you need to process hundreds of files, test many parameter values, or run the same simulation with different random seeds.

> This training builds on the [Slurm Jobs Array](https://rc-docs.northeastern.edu/en/latest/runningjobs/slurmarray.html) page in the Explorer documentation, which is a good reference to keep open alongside these notes.

This training assumes you have already completed our [Intro to Explorer and the OOD](https://rc.northeastern.edu/training/) session, or are comfortable logging in, moving around the file system, and submitting a basic `sbatch` script.

Today this presentation will cover:

1. [What is a job array?](#what-is-a-job-array)
2. [Your first job array](#your-first-job-array)
3. [Array index syntax](#array-index-syntax)
4. [Using the task ID: processing a list of files](#using-the-task-id-processing-a-list-of-files)
5. [Using the task ID: parameter sweeps](#using-the-task-id-parameter-sweeps)
6. [Job arrays with GPUs](#job-arrays-with-gpus)
7. [Resource allotment and limiting concurrent tasks](#resource-allotment-and-limiting-concurrent-tasks)
8. [Monitoring and canceling arrays](#monitoring-and-canceling-arrays)
9. [Checking resource use and rerunning failed tasks](#checking-resource-use-and-rerunning-failed-tasks)
10. [Running a final step after the array finishes](#running-a-final-step-after-the-array-finishes)
11. [Best practices and common mistakes](#best-practices-and-common-mistakes)
12. [Practice exercise](#practice-exercise)
13. [How to get help](#how-to-get-help)

## What is a job array?

Imagine you have 200 samples that each need to be run through the same program. You could:

A) Write 200 separate `sbatch` scripts and submit them one at a time.

B) Write one `sbatch` script with a `for` loop that runs all 200 samples, one after another, in a single long job.

C) Write **one** `sbatch` script and tell Slurm to run it 200 times, each time with a different task number.

Option C is a **job array**.

| Approach | Downsides |
| -------- | --------- |
| 200 separate scripts | Tedious, error prone, hard to manage |
| One long loop | Samples run one at a time, so it takes much longer; if the job times out you lose track of where it stopped |
| **Job array** | None of the above! Tasks run in parallel (up to the [concurrent task limit](#resource-allotment-and-limiting-concurrent-tasks)), each is tracked independently, and one script does it all |

In a job array, every **task** is its own independent job: it is scheduled separately, gets its own resources, and can run on a different compute node at the same time as its siblings. The only difference between the tasks is an environment variable called `SLURM_ARRAY_TASK_ID`, which holds that task's number. You use that number in your script to decide what each task should work on.

Some good use cases for job arrays:

- Running the same analysis on many input files or samples
- Parameter sweeps (testing many values of a setting)
- Running replicate simulations with different random seeds
- Processing a large dataset split into chunks
> Job arrays are for tasks that are **independent** of each other. If task 2 needs the results from task 1, a job array is not the right tool (but see [the section on job dependencies](#running-a-final-step-after-the-array-finishes)).

## Your first job array

Let's start with the simplest possible job array. Log in to Explorer and create a new directory to work in:

```
mkdir -p /scratch/<username>/job_arrays
cd /scratch/<username>/job_arrays
mkdir logs
```

Remember to substitute your own username. We also created a directory called `logs` where our output files will go. We will explain why in a moment.

Now create a file called `hello_array.sh` (you can use `nano hello_array.sh` or create it with the OOD Files app):

```
#!/bin/bash
#SBATCH --job-name=hello_array
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB
#SBATCH --time=00:05:00
#SBATCH --array=1-5
#SBATCH --output=logs/%x_%A_%a.out

echo "Hello from array task ${SLURM_ARRAY_TASK_ID}"
echo "This task is part of array job ${SLURM_ARRAY_JOB_ID}"
echo "Running on node: $(hostname)"
```

The only new line compared to a regular `sbatch` script is:

```
#SBATCH --array=1-5
```

This tells Slurm to run this script 5 times, with `SLURM_ARRAY_TASK_ID` set to 1, 2, 3, 4, and 5.

Submit it just like any other job:

```
sbatch hello_array.sh
```

```
Submitted batch job 3720115
```

Notice that you get **one** job ID back. This is the **array job ID**. Slurm then gives each task its own ID in the form `arrayjobid_taskid`, for example `3720115_1`, `3720115_2`, and so on.

### Environment variables for arrays

Slurm sets these variables inside each task:

| Variable | What it holds | Example |
| -------- | ------------- | ------- |
| `SLURM_ARRAY_TASK_ID` | The number of this specific task | `3` |
| `SLURM_ARRAY_JOB_ID` | The job ID shared by the whole array | `3720115` |
| `SLURM_JOB_ID` | The unique job ID of this specific task | `3720118` |
| `SLURM_ARRAY_TASK_COUNT` | The total number of tasks in the array | `5` |
| `SLURM_ARRAY_TASK_MIN` | The lowest task ID in the array | `1` |
| `SLURM_ARRAY_TASK_MAX` | The highest task ID in the array | `5` |

The one you will use the most is `SLURM_ARRAY_TASK_ID`.

### Naming output files

By default every job writes to `slurm-<jobid>.out`. For arrays, each task would produce a file named by its own unique job ID, which is hard to keep track of. It is much more readable to use the special filename patterns:

| Pattern | Replaced with |
| ------- | ------------- |
| `%A` | The array job ID |
| `%a` | The array task ID |
| `%x` | The job name |

So `--output=logs/%x_%A_%a.out` produces files like `logs/hello_array_3720115_3.out`.

The same patterns work with `--error`, for example `--error=logs/%x_%A_%a.err`, if you would like error messages written to a separate file from normal output. Including both `%A` and `%a` is important: it keeps one task from overwriting the output of another.
> Slurm will **not** create the `logs` directory for you. If the directory in your `--output` path does not exist, your job will fail without writing any output. This is why we ran `mkdir logs` before submitting.

After the job finishes, look at the output files:

```
ls logs
```

```
hello_array_3720115_1.out  hello_array_3720115_3.out  hello_array_3720115_5.out
hello_array_3720115_2.out  hello_array_3720115_4.out
```

```
cat logs/hello_array_3720115_3.out
```

```
Hello from array task 3
This task is part of array job 3720115
Running on node: c0712
```

Your tasks may well have run on different nodes at the same time. That is the power of a job array!

## Array index syntax

The `--array` option is flexible. You are not limited to counting from 1 to N.

| Syntax | Task IDs that are created |
| ------ | ------------------------- |
| `--array=1-10` | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| `--array=0-9` | 0, 1, 2, ..., 9 (handy if you like counting from zero) |
| `--array=1,3,5,7` | 1, 3, 5, 7 |
| `--array=1-20:5` | 1, 6, 11, 16 (a step of 5) |
| `--array=1-100%10` | 1 through 100, but only 10 run at a time (see [throttling](#resource-allotment-and-limiting-concurrent-tasks)) |
| `--array=1-5,20,30-32` | 1, 2, 3, 4, 5, 20, 30, 31, 32 |

You can also set the array on the command line instead of in the script. The command line takes priority over the `#SBATCH` line, which is useful for quickly re-running a few tasks:

```
sbatch --array=2,4 hello_array.sh
```
The general form on the command line is:

```
sbatch --array=<indexes> [options] script_file
```
> Task IDs must be non-negative integers. On Explorer, at most 50 of your tasks can run at once (see [the next section](#resource-allotment-and-limiting-concurrent-tasks)), so a very large array works through its tasks in waves. For very large arrays, see [the best practices section](#best-practices-and-common-mistakes) for how to bundle work into fewer tasks.

## Using the task ID: processing a list of files

A job array becomes really powerful when each task uses its `SLURM_ARRAY_TASK_ID` to pick something different to work on. The most common pattern is to keep a text file listing your inputs, with one per line, and let each task read the line that matches its ID.

### Step 1: Make a list of input files

First, let's create some small test data so that we all have something to work with:

```
mkdir -p data
for i in $(seq 1 10); do
    echo "This is sample number ${i}" > data/sample_${i}.txt
done
```

Now create a list of the files (using absolute paths!):

```
ls -d /scratch/<username>/job_arrays/data/*.txt > filelist.txt
```

Count the lines. This tells us how many tasks we need:

```
wc -l filelist.txt
```

```
10 filelist.txt
```

### Step 2: Write the array script

The command `sed -n "5p" filelist.txt` prints only line 5 of a file. We can swap the 5 for `${SLURM_ARRAY_TASK_ID}` so that task 5 gets line 5, task 6 gets line 6, and so on.

Create `process_files.sh`:

```
#!/bin/bash
#SBATCH --job-name=process_files
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=00:10:00
#SBATCH --array=1-10
#SBATCH --output=logs/%x_%A_%a.out

# Get the input file for this task from line N of filelist.txt
INPUT_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" filelist.txt)

# Make an output name based on the input name
BASENAME=$(basename "${INPUT_FILE}" .txt)
OUTPUT_FILE="results/${BASENAME}_result.txt"

mkdir -p results

echo "Task ${SLURM_ARRAY_TASK_ID} is processing: ${INPUT_FILE}"

# Replace this with your real analysis command!
wc -w "${INPUT_FILE}" > "${OUTPUT_FILE}"

echo "Task ${SLURM_ARRAY_TASK_ID} finished. Wrote ${OUTPUT_FILE}"
```

A few things to notice:

- `filelist.txt` is referenced by a relative path, so this works because we submit the job from the directory that contains it. Slurm starts each task in the directory you submitted from. For safety you can use the absolute path instead.
- Each task writes to a **different** output file. Tasks must never write to the same output file at the same time, or they will overwrite or corrupt each other.
- The `wc -w` line is just a stand-in. This is where you would put your real program, for example `python analyze.py ${INPUT_FILE}`.

Submit and check the results:

```
sbatch process_files.sh
```

```
Submitted batch job 3720201
```

```
ls results
```

```
sample_1_result.txt   sample_2_result.txt   sample_3_result.txt   ...
```

> **Tip:** Set the `--array` range to match the number of lines in your file. If your list has 250 files, use `--array=1-250`. You can also let the shell do the counting for you at submission time:
>
> ```
> sbatch --array=1-$(wc -l < filelist.txt) process_files.sh
> ```

### Using a real software example

Here is what the same pattern looks like with a conda environment and a Python script:

```
#!/bin/bash
#SBATCH --job-name=analyze
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=02:00:00
#SBATCH --array=1-50
#SBATCH --output=logs/%x_%A_%a.out

# load required software
module load anaconda3/2024.06
source activate /home/<username>/.conda/envs/myenv

# Get the input file for this task
INPUT_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" filelist.txt)

# Run the analysis on that one file
python analyze.py --input "${INPUT_FILE}" --threads ${SLURM_CPUS_PER_TASK}
```

Notice `${SLURM_CPUS_PER_TASK}`. Using this variable means the number of threads you give your program always matches the number of CPUs you requested, even if you change the `#SBATCH` line later.
> Remember that the resources you request (`--mem`, `--cpus-per-task`, `--time`) apply to **each task**, not to the array as a whole. The script above asks for 8 GB for every one of its 50 tasks.

### Reading the task ID inside your own program

Often the cleanest approach is to hand the task ID to your script as an argument, for example `python analyze.py ${SLURM_ARRAY_TASK_ID}`. You can also read the environment variable directly from inside your program.

In Python:

```
import os
task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
print(f"I am task {task_id}")
```

In R:

```
task_id <- as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))
cat("I am task", task_id, "\n")
```

Once you have the task ID, you can use it to pick an input file, a parameter, or a random seed, exactly as we did in the shell above.

## Using the task ID: parameter sweeps

Job arrays are also great when you don't have a list of files, but want to try different values of a parameter.

### Option 1: Use the task ID directly as the parameter

If your parameter is simply a number, you can use the task ID itself. Here we use it as a random seed:

```
#!/bin/bash
#SBATCH --job-name=seed_sweep
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --time=00:30:00
#SBATCH --array=1-20
#SBATCH --output=logs/%x_%A_%a.out

module load anaconda3/2024.06
source activate /home/<username>/.conda/envs/myenv

python simulate.py --seed ${SLURM_ARRAY_TASK_ID} --output results/sim_seed${SLURM_ARRAY_TASK_ID}.csv
```

### Option 2: Look up the parameter in a table

If your parameters are not simple numbers, make a small parameter file with one set of values per line. For example `params.txt`:

```
0.01 100
0.01 200
0.05 100
0.05 200
0.10 100
0.10 200
```

Then read the line for each task and split it into variables:

```
#!/bin/bash
#SBATCH --job-name=param_sweep
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --time=00:30:00
#SBATCH --array=1-6
#SBATCH --output=logs/%x_%A_%a.out

# Read line N of params.txt into two variables
read LEARNING_RATE EPOCHS < <(sed -n "${SLURM_ARRAY_TASK_ID}p" params.txt)

echo "Task ${SLURM_ARRAY_TASK_ID}: learning rate = ${LEARNING_RATE}, epochs = ${EPOCHS}"

python train.py --lr ${LEARNING_RATE} --epochs ${EPOCHS} --out results/run_${SLURM_ARRAY_TASK_ID}
```

### Option 3: Do some arithmetic with the task ID

Suppose you want to test 3 values of one parameter against 4 values of another, for 12 combinations total. You can use shell arithmetic with the task ID (counting from 0) to work out which combination belongs to each task:

```
#!/bin/bash
#SBATCH --array=0-11
#SBATCH --job-name=grid
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%A_%a.out

ALPHAS=(0.1 0.5 1.0)
BETAS=(10 20 30 40)

# 12 tasks = 3 alphas x 4 betas
ALPHA=${ALPHAS[$(( SLURM_ARRAY_TASK_ID / 4 ))]}
BETA=${BETAS[$(( SLURM_ARRAY_TASK_ID % 4 ))]}

echo "Task ${SLURM_ARRAY_TASK_ID}: alpha=${ALPHA}, beta=${BETA}"
```

Here, integer division (`/ 4`) picks the alpha value and the remainder (`% 4`) picks the beta value. Task 0 gets alpha=0.1, beta=10; task 1 gets alpha=0.1, beta=20; and so on up to task 11, which gets alpha=1.0, beta=40.

## Job arrays with GPUs

Job arrays work with GPUs too, and they are a natural fit for jobs like training the same model with different hyperparameters, or running the same model on many inputs. Each task asks for its own GPU, and the tasks run independently, just like CPU tasks.

There are a few things to know before you start. For more detail, see the [GPU Access](https://rc-docs.northeastern.edu/en/latest/gpus/accessinggpus.html) and [GPU Job Submission](https://rc-docs.northeastern.edu/en/latest/gpus/gpujobsubmission.html) pages of our documentation, and our [Intro to GPUs on the Cluster](https://rc.northeastern.edu/calendar_event/intro-to-gpus-on-the-cluster/) training.

- **Request a GPU with `--gres`.** A GPU is only given to your job if you ask for it, for example `--gres=gpu:1`.
- **Use the `gpu` partition for `sbatch` jobs.** The `gpu-interactive` and `gpu-short` partitions are for interactive GPU work, so test your commands there first with `srun`.
- **Request one GPU per task.** On the `gpu` partition, a request for more than one GPU (for example `--gres=gpu:2`) will cause your request to fail, and you also cannot request all the CPUs on a GPU node, since they are reserved for the other GPUs on that node. One GPU per task is exactly what a job array is built for.
- **The GPU request is per task, like all other resources.** An array with 10 tasks will use 10 GPUs if all of them run at once.
- **You can ask for a specific GPU type**, such as `--gres=gpu:v100-pcie:1`, but this could mean longer waits depending on what is available. See the [GPU Nodes table](https://rc.northeastern.edu/compute/) for the types we have.
> If you use a general request (`--gres=gpu:1`) you get whichever GPU type is available first. That is fine for quick tests, but for repeated runs you should use the same GPU type or at least record which one was used, so that your results and timings are comparable.

### Test on one task first

Before launching an array, test with an interactive session on a GPU node (this can only be run from a login node):

```
srun --partition=gpu-interactive --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --mem=8GB --time=01:00:00 --nodes=1 --pty /bin/bash
```

Once you are on the node, run `nvidia-smi` to confirm you can see a GPU, and run your script once by hand with a single parameter set.

### A GPU hyperparameter sweep

Suppose we want to train a model with 8 different combinations of learning rate and batch size. We make a parameter file called `gpu_params.txt`, with one set of values per line:

```
0.001 16
0.001 32
0.001 64
0.001 128
0.0001 16
0.0001 32
0.0001 64
0.0001 128
```

And then create `gpu_array.sh`:

```
#!/bin/bash
#SBATCH --job-name=gpu_sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --array=1-8%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# load required software
module load anaconda3/2024.06
source activate /home/<username>/.conda/envs/gpu_env

# Read line N of gpu_params.txt into two variables
read LEARNING_RATE BATCH_SIZE < <(sed -n "${SLURM_ARRAY_TASK_ID}p" gpu_params.txt)

# Record which GPU this task received (useful for reproducibility!)
echo "Task ${SLURM_ARRAY_TASK_ID} running on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Learning rate: ${LEARNING_RATE}, batch size: ${BATCH_SIZE}"

mkdir -p results

python train.py \
    --lr ${LEARNING_RATE} \
    --batch-size ${BATCH_SIZE} \
    --out results/run_${SLURM_ARRAY_TASK_ID}
```

Let's look at what is different from our CPU arrays:

- `--partition=gpu` and `--gres=gpu:1` request one GPU for every task.
- `--array=1-8%4` limits the array to 4 tasks (and so 4 GPUs) at a time. GPUs are in high demand, so a smaller throttle is a considerate choice. Remember that Explorer's limit of 50 running tasks per account still applies as an upper bound.
- `--cpus-per-task=4` gives each task a few CPU cores to feed data to the GPU (for example, to load and preprocess batches). Match this to what your program can actually use, such as the number of data loader workers.
- The `nvidia-smi` line writes the GPU model into each task's log, so you can tell afterwards which type of GPU each run used.
- The conda environment `gpu_env` needs a GPU-enabled build of your framework. See our [GPU Job Submission](https://rc-docs.northeastern.edu/en/latest/gpus/gpujobsubmission.html) page for how to set up CUDA and deep learning environments.

Submit it like any other array:

```
sbatch gpu_array.sh
```

```
Submitted batch job 3720612
```

And watch it with `squeue`:

```
squeue -u <username>
```

```
  JOBID       PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
3720612_[5-8%4]  gpu  gpu_swee  s.caplin PD       0:00      1 (None)
3720612_1        gpu  gpu_swee  s.caplin  R       2:10      1 d1011
3720612_2        gpu  gpu_swee  s.caplin  R       2:10      1 d1011
3720612_3        gpu  gpu_swee  s.caplin  R       2:10      1 d1023
3720612_4        gpu  gpu_swee  s.caplin  R       2:10      1 c2165
```

### Making sure your tasks really use the GPU

A GPU job that quietly falls back to the CPU is a common problem, and it wastes a lot of waiting time. Before you scale up, check one task's log for the `nvidia-smi` output, and confirm your framework sees the GPU. For example, in Python with PyTorch:

```
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))
```

If you also want to know how well a task used its GPU, run `nvidia-smi` in an interactive session while your program runs. Then right-size your request for the full array in the same way we did for CPU and memory.
> Don't request a GPU for steps that don't need one. If your workflow has a CPU-only preprocessing step and a GPU training step, run them as separate jobs (you can link them with a [job dependency](#running-a-final-step-after-the-array-finishes)) so that the GPU isn't sitting idle while the CPU work runs.

## Resource allotment and limiting concurrent tasks

### Resources apply to each task

All resource requests in your `#SBATCH` lines (`--mem`, `--cpus-per-task`, `--nodes`, `--time`, and so on) apply to **each individual task**, not to the array as a whole. A request of `--mem=128G` does not divide 128 GB among your tasks. Every task receives 128 GB.

For example, consider this array of 600 tasks, run 20 at a time:

```
#SBATCH --partition=short
#SBATCH --array=1-600%20
#SBATCH --mem=128G
#SBATCH --nodes=2
```

At peak, Slurm may allocate up to 20 x 128 GB = 2,560 GB of memory (and 40 nodes) to this one array, subject to availability and limits. Requests that are larger than you need make every task wait longer to start, and take resources away from other users.

If your program is not multi-threaded, use a lean header like this one instead:

```
#SBATCH --partition=short
#SBATCH --array=1-600%50
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
```
> Test your program on one task first so you know how much RAM it really needs, and then set `--mem-per-cpu` (or `--mem`) accordingly.

### Explorer allows 50 running tasks at once

On Explorer, the maximum number of jobs allowed to run at once is **50 per user account**. If you submit a larger array, the remaining tasks simply wait as pending and start as earlier tasks finish. Nothing is lost, but it is good to know when you estimate how long a large array will take.

### Throttle your array with `%`

If you submit an array of 500 tasks, Slurm may try to run many of them at the same time, depending on how busy the cluster is. Sometimes you want to limit this. Maybe your tasks all read from the same file, or they put heavy load on the file system, or you simply want to be considerate of other users.

Add a `%` followed by a number to the end of the array range to set the maximum number of tasks that can run simultaneously:

```
#SBATCH --array=1-500%20
```

This runs all 500 tasks, but never more than 20 at a time. As each task finishes, the next one starts. The same syntax works on the command line:

```
sbatch --array=1-200%5 script.sh
```

You can also change the limit on an array that is already running, using `scontrol`:

```
scontrol update JobId=3720201 ArrayTaskThrottle=10
```
> The throttle does not change the total amount of work done. It only controls how fast you work through it. A throttle is a good habit for very large arrays.

### Why requesting less gets you through the queue faster

Explorer uses a fair-share scheduling policy. Your job priority depends on how many resources your group has used in the past, how large your jobs are (smaller jobs typically have higher priority), and how long the job has been waiting. Two tips from our documentation on [the queuing system](https://rc-docs.northeastern.edu/en/latest/runningjobs/understandingqueuing.html) apply especially well to job arrays:

1. **Request only the resources you need.** Overestimating can result in longer queue times.
2. **Break up large jobs.** Large jobs tend to wait longer than small ones. An array of many small tasks is a natural way to do this.

## Monitoring and canceling arrays

### Checking the partition

Before you submit, you can check the status and time limit of a partition with `sinfo`:

```
sinfo -p short
```

The `TIMELIMIT` column shows the longest time a task in that partition can request with `--time`. Remember that every task in your array has its own time limit.

### Checking the queue

When you run `squeue` on an array, pending tasks are grouped together on a single line to keep the output short:

```
squeue -u <username>
```

```
  JOBID       PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
3720350_[6-20%5]  short  param_sw  s.caplin PD       0:00      1 (None)
3720350_1         short  param_sw  s.caplin  R       0:42      1 c0712
3720350_2         short  param_sw  s.caplin  R       0:42      1 c0713
3720350_3         short  param_sw  s.caplin  R       0:42      1 c0713
3720350_4         short  param_sw  s.caplin  R       0:42      1 c0715
3720350_5         short  param_sw  s.caplin  R       0:42      1 c0715
```

Here tasks 1-5 are running (`R`), and tasks 6-20 are pending (`PD`). The `%5` in the brackets reminds us that a throttle of 5 is set.

If you would like every task to be shown on its own line, add the `-r` (array) flag:

```
squeue -u <username> -r
```

### Canceling jobs

You can cancel the whole array, or only part of it:

```
scancel 3720350            # cancel the entire array
scancel 3720350_4          # cancel only task 4
scancel 3720350_[10-15]    # cancel tasks 10 through 15
scancel -u <username>      # cancel ALL of your jobs (use with care!)
```

## Checking resource use and rerunning failed tasks

### Check efficiency with seff

As in our introductory session, `seff` shows how well your job used the resources you requested. For an array, run it on an individual task using the `arrayjobid_taskid` format:

```
seff 3720350_1
```

Check a few tasks, and use what you learn to tune the resources for the rest. If you notice every task is only using 10% of the memory you requested, reduce `--mem` for the next run. Since you are requesting resources for many tasks, over-requesting adds up quickly and makes your whole array wait longer in the queue.

### Find out which tasks failed

With hundreds of tasks, it is likely that a few may fail (for example by running out of time or memory). The `sacct` command lists the state of every task:

```
sacct -j 3720350 -X --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

```
JobID           JobName      State ExitCode    Elapsed     MaxRSS
------------ ---------- ---------- -------- ---------- ----------
3720350_1    param_swe+  COMPLETED      0:0   00:03:11
3720350_2    param_swe+  COMPLETED      0:0   00:03:09
3720350_3    param_swe+ OUT_OF_ME+    0:125   00:01:42
3720350_4    param_swe+  COMPLETED      0:0   00:03:15
3720350_5    param_swe+    TIMEOUT      0:0   00:30:12
```

The `-X` flag tells `sacct` to show one line per task rather than extra lines for each step inside the task.

To show only the tasks that did not complete successfully:

```
sacct -j 3720350 -X --state=FAILED,TIMEOUT,OUT_OF_MEMORY,CANCELLED --format=JobID,State,ExitCode
```

### Rerun only the failed tasks

You don't need to rerun the whole array. Just resubmit the failed task IDs, and increase the resources if that was the cause of the failure:

```
sbatch --array=3,5 --mem=8GB --time=01:00:00 param_sweep.sh
```

Command-line options override the `#SBATCH` lines in the script, so you don't even have to edit the file.

## Running a final step after the array finishes

Sometimes you need a final step that combines the results of all the array tasks, for example merging all the output files into one table. You can tell Slurm to wait until the entire array has finished successfully using a **job dependency**.

First, submit the array and save its job ID. The `--parsable` flag makes `sbatch` print only the job ID:

```
ARRAY_ID=$(sbatch --parsable process_files.sh)
echo "Array job ID: ${ARRAY_ID}"
```

Then submit the combining job, telling Slurm it depends on the array:

```
sbatch --dependency=afterok:${ARRAY_ID} combine_results.sh
```

The `afterok` option means the second job will only start once **every** task in the array has completed with no errors. If any task fails, the dependent job will never run and will stay in the queue with the reason `DependencyNeverSatisfied`; you can cancel it with `scancel`.

Here is a simple `combine_results.sh`:

```
#!/bin/bash
#SBATCH --job-name=combine
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out

cat results/*_result.txt > all_results.txt
echo "Combined $(wc -l < all_results.txt) result lines into all_results.txt"
```

If you would rather run the combining step even when some tasks fail, use `afterany` instead of `afterok`.

## Best practices and common mistakes

**Do:**

- **Test first.** Run your script on a single task (`sbatch --array=1 script.sh`) or interactively with `srun` before launching hundreds of tasks. Use what you learn to set your memory request.
- **Start small, then scale.** Check `seff` on a few tasks and adjust your resource requests before launching the full array.
- **Write to unique output files.** Include the task ID or input name in every output filename.
- **Use absolute paths** in your scripts, as we recommended in the intro training.
- **Create your `logs` directory before submitting.**
- **Use a throttle** (`%N`) for large arrays, keeping in mind the limit of 50 running tasks per user account. Use a smaller throttle for GPU arrays.
- **Record the GPU type** each task used (`nvidia-smi`) so GPU results are reproducible.
- **Keep a record** of your `filelist.txt` or `params.txt` along with your results so your work is reproducible.

**Don't:**

- **Don't make tasks too short.** Every task has scheduling overhead, and with only 50 tasks running at once, thousands of tiny tasks will take a long time to cycle through. If each of your tasks only takes a few seconds, it is much more efficient to bundle many pieces of work into each task. For example, with 10,000 small files you might use 100 tasks that each process 100 files:

  ```
  #SBATCH --array=0-99

  CHUNK_SIZE=100
  START=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE + 1 ))
  END=$(( START + CHUNK_SIZE - 1 ))

  sed -n "${START},${END}p" filelist.txt | while read -r FILE; do
      python analyze.py --input "${FILE}"
  done
  ```

- **Don't have tasks write to the same file** at the same time.
- **Don't request more than one GPU per task on the `gpu` partition.** That request will fail.
- **Don't forget that resources are per task.** Requesting `--mem=100GB` in a 200-task array means requesting 20 TB across the array (although not all of it at once if you use a throttle).
- **Don't use job arrays for dependent steps.** If task N needs the result of task N-1, use job dependencies or a different workflow.
- **Don't put your `module load` commands in your `.bashrc`.** Load modules inside your sbatch script so each task is reproducible.

| Symptom | Likely cause |
| ------- | ------------ |
| No output files appear | The directory in `--output` does not exist |
| Every task processes the same file | The script is not using `${SLURM_ARRAY_TASK_ID}` (check for typos or single quotes) |
| `sed` returns an empty line for some tasks | The `--array` range is larger than the number of lines in your list file |
| Task ends with `OUT_OF_MEMORY` | `--mem` too small for that input; rerun just that task with more memory |
| Task ends with `TIMEOUT` | `--time` too small; rerun with a longer time (or ask about the `long` partition) |
| Array sits in the queue for a long time | Each task asks for a lot of resources; reduce per-task requests |
| GPU task fails to start, or the request is rejected | More than one GPU was requested on the `gpu` partition (only `--gres=gpu:1` per job works there), or the requested GPU type does not exist |
| GPU task runs but is very slow | The program is running on the CPU; check the log for `nvidia-smi` output and confirm your framework sees the GPU |
| Fewer tasks are running than you expected | You have reached the limit of 50 running jobs per user account, or your `%N` throttle is set low; remaining tasks wait as pending |

## Practice exercise

Try this on your own or with a neighbor.

**The scenario:** You have a folder of 20 text files named `book_1.txt` through `book_20.txt`, and you want to count the number of lines in each one.

1. Create the test data:

   ```
   mkdir -p exercise/data
   cd exercise
   for i in $(seq 1 20); do
       seq 1 $(( i * 10 )) > data/book_${i}.txt
   done
   ```

2. Make a file called `filelist.txt` with the absolute path to each file in `data/`.
3. Write an sbatch array script that:
   - uses `--array` with a throttle of 5 tasks at once,
   - names its log files `logs/%x_%A_%a.out`,
   - reads its input filename from `filelist.txt` using `SLURM_ARRAY_TASK_ID`,
   - writes the line count to `results/book_<N>_lines.txt`.
4. Submit your array and watch it with `squeue -u <username>`.
5. Use `seff` on one of your finished tasks. Did you ask for too many resources?
6. **Bonus:** Submit a second job that depends on your array and combines all the results into one file called `all_line_counts.txt`.
7. **GPU bonus:** Copy your script and change it to request one GPU on the `gpu` partition. Have each task print the GPU model with `nvidia-smi --query-gpu=name --format=csv,noheader` instead of counting lines, and limit the array to 2 tasks at a time. Look at your log files to see which GPU each task received.

<details>
<summary>Click to see a sample solution</summary>

```
#!/bin/bash
#SBATCH --job-name=count_lines
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --mem=1GB
#SBATCH --time=00:05:00
#SBATCH --array=1-20%5
#SBATCH --output=logs/%x_%A_%a.out

INPUT_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" filelist.txt)
mkdir -p results

wc -l < "${INPUT_FILE}" > results/book_${SLURM_ARRAY_TASK_ID}_lines.txt
echo "Task ${SLURM_ARRAY_TASK_ID} counted lines in ${INPUT_FILE}"
```

</details>

## Quick reference

| I want to... | Command |
| ------------ | ------- |
| Submit an array | `sbatch --array=1-100 script.sh` |
| Limit to 10 at a time | `sbatch --array=1-100%10 script.sh` |
| Check a partition's time limit | `sinfo -p short` |
| Give each task one GPU | `#SBATCH --partition=gpu` and `#SBATCH --gres=gpu:1` |
| Rerun specific tasks | `sbatch --array=4,17,32 script.sh` |
| Get the task number in a script | `${SLURM_ARRAY_TASK_ID}` |
| Name output per task | `#SBATCH --output=logs/%x_%A_%a.out` |
| See my queue, one line per task | `squeue -u <username> -r` |
| Cancel whole array | `scancel <arrayjobid>` |
| Cancel one task | `scancel <arrayjobid>_<taskid>` |
| Check one task's efficiency | `seff <arrayjobid>_<taskid>` |
| List all task states | `sacct -j <arrayjobid> -X` |
| Run a job after an array finishes | `sbatch --dependency=afterok:<arrayjobid> next.sh` |

### Further reading

In our Explorer documentation:

- [Slurm Jobs Array](https://rc-docs.northeastern.edu/en/latest/runningjobs/slurmarray.html)
- [Understanding the Queuing System](https://rc-docs.northeastern.edu/en/latest/runningjobs/understandingqueuing.html)
- [Monitoring and Managing Jobs](https://rc-docs.northeastern.edu/en/latest/runningjobs/slurmmonitoringandmanaging.html)
- [Interactive and Batch Mode](https://rc-docs.northeastern.edu/en/latest/runningjobs/interactiveandbatch.html)
- [GPU Access](https://rc-docs.northeastern.edu/en/latest/gpus/accessinggpus.html)
- [GPU Job Submission](https://rc-docs.northeastern.edu/en/latest/gpus/gpujobsubmission.html)
- [Job Efficiency (Cluster Usage best practices)](https://rc-docs.northeastern.edu/en/latest/best-practices/clusterusage.html#job-eff)
- [Courses CheatSheet](https://rc-docs.northeastern.edu/en/latest/classroom/cheatsheet.html), which lists many commonly used Slurm commands

For all of the available options, see the official [Slurm sbatch documentation](https://slurm.schedmd.com/sbatch.html) and the [Slurm job array guide](https://slurm.schedmd.com/job_array.html).

## How to get help

Email the Research Computing team at <rchelp@northeastern.edu>.

Come to [office hours](https://rc.northeastern.edu/getting-help/) hosted on Zoom.

Or [book a consultation](https://rc.northeastern.edu/getting-help/) with an RC team member.

Review our [Documentation](https://rc-docs.northeastern.edu/en/latest/index.html).

Thank you!

---

*For questions or support, contact the Research Computing team at <rchelp@northeastern.edu>*
