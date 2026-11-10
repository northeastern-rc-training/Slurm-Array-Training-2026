# Research Computing Training

[![drawing](ac-grid-5-red.svg)](images/ac-grid-5-red.svg)

## Presentor

Your Name Here

Title

[Research Computing](https://rc.northeastern.edu/research-computing-team/)

## Running Many Jobs at Once with Slurm Job Arrays

Welcome to the [Research Computing Training Series](https://rc.northeastern.edu/training/)! In this training you will learn how to use Slurm job arrays on the Explorer cluster to run many similar jobs from one `sbatch` script, and how the [partition limits](https://rc.northeastern.edu/partitions/) shape what you can submit.

This training assumes you can log in to Explorer, move around the file system, and submit a basic `sbatch` script. If not, start with our [Intro to Explorer and the OOD](https://rc.northeastern.edu/training/) session.

> Since November 6, 2026, logging in to Explorer over SSH requires SSH keys on your local machine. See the [SSH Instructions for Explorer](https://rc.northeastern.edu/2026/10/23/ssh-instructions-for-explorer/). This does not affect the Open OnDemand website.

Today this presentation will cover:

1. [What is a job array?](#what-is-a-job-array)
2. [Your first job array](#your-first-job-array)
3. [Partition limits and job arrays](#partition-limits-and-job-arrays)
4. [Using the task ID: a list of files](#using-the-task-id-a-list-of-files)
5. [Using the task ID: a simple parameter](#using-the-task-id-a-simple-parameter)
6. [Job arrays with GPUs](#job-arrays-with-gpus)
7. [Monitoring, canceling, and rerunning](#monitoring-canceling-and-rerunning)
8. [Running a final step after the array](#running-a-final-step-after-the-array)
9. [Best practices and common mistakes](#best-practices-and-common-mistakes)
10. [Practice exercise](#practice-exercise)
11. [How to get help](#how-to-get-help)

## What is a job array?

Imagine you have 200 samples that each need the same analysis. You could write 200 separate scripts, or put a loop in one long job that runs the samples one after another. A **job array** is a better option: you write **one** script and tell Slurm to run it 200 times, each time with a different task number.

Every **task** in an array is its own job. It is scheduled separately, gets its own resources, and can run on a different node at the same time as the others. The only difference between tasks is an environment variable, `SLURM_ARRAY_TASK_ID`, which holds that task's number. Your script uses that number to decide what to work on.

Good uses for job arrays:

- Running the same analysis on many input files
- Testing many values of a parameter
- Running replicates with different random seeds

> Job arrays are for tasks that are **independent** of each other. If task 2 needs the results of task 1, use a [job dependency](#running-a-final-step-after-the-array) instead.

## Your first job array

Log in to Explorer and make a working directory:

```
mkdir -p /scratch/<username>/job_arrays
cd /scratch/<username>/job_arrays
mkdir logs
```

Create a file called `hello_array.sh`:

```
#!/bin/bash
#SBATCH --job-name=hello_array
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1GB
#SBATCH --time=00:05:00
#SBATCH --array=1-5
#SBATCH --output=logs/%x_%A_%a.out

echo "Hello from task ${SLURM_ARRAY_TASK_ID} of array job ${SLURM_ARRAY_JOB_ID}"
echo "Running on node: $(hostname)"
```

The only new line compared to a regular script is `#SBATCH --array=1-5`. It runs the script 5 times, with `SLURM_ARRAY_TASK_ID` set to 1, 2, 3, 4, and 5. Submit it as usual:

```
sbatch hello_array.sh
```

```
Submitted batch job 3720115
```

You get **one** job ID, the **array job ID**. Each task is then named `arrayjobid_taskid`, such as `3720115_1` and `3720115_2`.

| Variable / pattern | Meaning | Example |
| ------------------ | ------- | ------- |
| `SLURM_ARRAY_TASK_ID` | The number of this task | `3` |
| `SLURM_ARRAY_JOB_ID` | The job ID shared by the whole array | `3720115` |
| `%A` in a filename | The array job ID | `3720115` |
| `%a` in a filename | The task ID | `3` |
| `%x` in a filename | The job name | `hello_array` |

Using both `%A` and `%a` in `--output` gives every task its own file, such as `logs/hello_array_3720115_3.out`, so tasks never overwrite each other.

> Slurm does **not** create the `logs` directory for you. If it does not exist, your job will fail without writing any output.

When the tasks finish, look at one of the logs:

```
cat logs/hello_array_3720115_3.out
```

```
Hello from task 3 of array job 3720115
Running on node: c0712
```

### Choosing task numbers

| Syntax | Task IDs |
| ------ | -------- |
| `--array=1-10` | 1, 2, 3, ..., 10 |
| `--array=0-9` | 0, 1, 2, ..., 9 |
| `--array=1,3,5,7` | 1, 3, 5, 7 |
| `--array=1-20:5` | 1, 6, 11, 16 |
| `--array=1-100%10` | 1 to 100, but only 10 run at a time |

You can also set the array on the command line, which overrides the script. This is handy for rerunning a few tasks:

```
sbatch --array=2,4 hello_array.sh
```

## Partition limits and job arrays

Every partition has limits, and they apply to job arrays in a way that is easy to miss: **each array task counts as one job** toward the running and submitted job limits. The limits below come from the [Partitions](https://rc.northeastern.edu/partitions/) page.

**CPU partitions**

| Partition | Approval needed? | Time limit (default / max) | Running jobs | Submitted jobs |
| --------- | ---------------- | -------------------------- | ------------ | -------------- |
| `short` | No | 4 hours / 48 hours | 50 | 1000 |
| `long` | Yes | 1 day / 5 days | 25 | 1000 |
| `sharing` | No | 30 minutes / 60 minutes | 2 | 4 |

**GPU partitions** (each has a limit of 1 GPU per job, except `multigpu`)

| Partition | Approval needed? | Time limit (default / max) | Running jobs | Submitted jobs |
| --------- | ---------------- | -------------------------- | ------------ | -------------- |
| `gpu-short` | No | 1 hour / 2 hours | 2 | 4 |
| `gpu-interactive` | No | 1 hour / 2 hours | 2 | 4 |
| `gpu` | No | 4 hours / 8 hours | 4 | 8 |
| `multigpu` (up to 8 GPUs) | Yes | 12 hours / 24 hours | 4 | 8 |

What this means for your arrays:

1. **The submitted limit is the largest array you can submit.** On `short`, that is 1000 tasks (`--array=1-1000`). On `gpu`, it is 8 tasks. If you go over, you will see the message `job violates accounting/QOS policy`.
2. **The running limit is how many tasks run at once**, no matter how big your array is. On `short`, at most 50 of your tasks run at once and the rest wait as pending. This is a maximum, and the actual number depends on what resources are free on the cluster.
3. **Each task's `--time` must be at or below the partition's maximum.** If you leave out `--time`, you get the default, which is only 4 hours on `short`. Always set it yourself.
4. **Resources are per task.** `--mem=4GB` means every task gets 4 GB. With 50 tasks running on `short`, that is up to 200 GB at once. Only ask for what one task needs.

If your array is too big for the submitted limit, bundle several pieces of work into each task. For example, 5,000 files fit into 100 tasks of 50 files each:

```
#SBATCH --array=0-99

CHUNK=50
START=$(( SLURM_ARRAY_TASK_ID * CHUNK + 1 ))
END=$(( START + CHUNK - 1 ))

sed -n "${START},${END}p" filelist.txt | while read -r FILE; do
    echo "Processing ${FILE}"
done
```

### Slowing an array down with `%`

The `%` throttle (`--array=1-500%20`) sets your own cap on simultaneous tasks. It can only be **lower** than the partition's running limit, never higher. Use it when tasks all read the same file or put heavy load on the file system, or just to be considerate of other users.

> Requesting more resources than you need does not make your jobs faster. It makes them wait longer. Explorer uses fair-share scheduling, so smaller requests start sooner.

## Using the task ID: a list of files

The most common pattern is a text file listing your inputs, one per line. Each task reads the line that matches its task ID.

First, make some test files and a list of them (use absolute paths):

```
mkdir -p data
for i in 1 2 3 4 5; do echo "This is sample number ${i}" > data/sample_${i}.txt; done
ls -d /scratch/<username>/job_arrays/data/*.txt > filelist.txt
wc -l filelist.txt
```

```
5 filelist.txt
```

The command `sed -n "3p" filelist.txt` prints only line 3. Replacing the 3 with the task ID gives each task its own file. Create `process_files.sh`:

```
#!/bin/bash
#SBATCH --job-name=process_files
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1GB
#SBATCH --time=00:05:00
#SBATCH --array=1-5
#SBATCH --output=logs/%x_%A_%a.out

INPUT_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" filelist.txt)

mkdir -p results
echo "Task ${SLURM_ARRAY_TASK_ID} is processing ${INPUT_FILE}"

# Replace this line with your real analysis command
wc -w "${INPUT_FILE}" > results/result_${SLURM_ARRAY_TASK_ID}.txt
```

Submit it, then check the `results` folder:

```
sbatch process_files.sh
ls results
```

```
result_1.txt  result_2.txt  result_3.txt  result_4.txt  result_5.txt
```

Each task writes to a **different** output file. Tasks must never write to the same file at the same time.

> Make the `--array` range match the number of lines in your list. You can let the shell count for you: `sbatch --array=1-$(wc -l < filelist.txt) process_files.sh`

## Using the task ID: a simple parameter

If all you need is a different number for each run, such as a random seed, use the task ID directly:

```
#SBATCH --array=1-10

echo "Running with seed ${SLURM_ARRAY_TASK_ID}"
python simulate.py --seed ${SLURM_ARRAY_TASK_ID}
```

You can also read the task ID inside your own program.

In Python:

```
import os
task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
```

In R:

```
task_id <- as.numeric(Sys.getenv("SLURM_ARRAY_TASK_ID"))
```

## Job arrays with GPUs

GPU arrays work the same way, and each task asks for its own GPU with `--gres`. The limits in the [GPU table above](#partition-limits-and-job-arrays) matter even more here:

- Use the `gpu` partition for batch jobs. It allows **1 GPU per job**, **4 running** jobs, and **8 submitted** jobs, so an array there can have at most 8 tasks.
- `gpu-short` has a higher priority and shorter waits, but it is limited to 2 hours, 2 running, and 4 submitted jobs. It is a good choice for quick tests.
- Need more than 1 GPU per job? That requires the `multigpu` partition, which needs a [Partition Access Request](https://bit.ly/NURC-PartitionAccess).
- Asking for a specific GPU type (for example `--gres=gpu:v100-pcie:1`) can mean a longer wait.

Here is a simple GPU array. Each task just reports which GPU it received. Create `gpu_array.sh`:

```
#!/bin/bash
#SBATCH --job-name=gpu_array
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4GB
#SBATCH --time=00:10:00
#SBATCH --array=1-4
#SBATCH --output=logs/%x_%A_%a.out

echo "Task ${SLURM_ARRAY_TASK_ID} is running on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
```

```
sbatch gpu_array.sh
```

Before you scale up a real GPU job, test one task in an interactive session on `gpu-interactive`:

```
srun --partition=gpu-interactive --nodes=1 --ntasks=1 --gres=gpu:1 --mem=4GB --time=00:30:00 --pty /bin/bash
```

Then run `nvidia-smi` to confirm you can see a GPU. For setting up CUDA and deep learning environments, see [GPU Job Submission](https://rc-docs.northeastern.edu/en/latest/gpus/gpujobsubmission.html).

## Monitoring, canceling, and rerunning

**Check the queue.** Pending tasks are grouped on one line:

```
squeue -u <username>
```

```
  JOBID       PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
3720350_[6-20%5]  short  param_sw  s.caplin PD       0:00      1 (None)
3720350_1         short  param_sw  s.caplin  R       0:42      1 c0712
3720350_2         short  param_sw  s.caplin  R       0:42      1 c0713
```

Add `-r` to see every task on its own line.

**Cancel jobs.** You can cancel all of an array or just part of it:

```
scancel 3720350          # the whole array
scancel 3720350_4        # only task 4
scancel 3720350_[10-15]  # tasks 10 through 15
```

**Check efficiency.** Run `seff` on one task using the `arrayjobid_taskid` form, and adjust the resources for the rest:

```
seff 3720350_1
```

**Find and rerun failed tasks.** List the state of every task:

```
sacct -j 3720350 -X --format=JobID,State,ExitCode,Elapsed
```

```
JobID             State ExitCode    Elapsed
------------ ---------- -------- ----------
3720350_1     COMPLETED      0:0   00:03:11
3720350_2     COMPLETED      0:0   00:03:09
3720350_3 OUT_OF_MEMORY    0:125   00:01:42
3720350_4       TIMEOUT      0:0   00:30:12
```

Then rerun only the tasks that failed, giving them more memory or time if that was the cause. Command-line options override the `#SBATCH` lines:

```
sbatch --array=3,4 --mem=8GB --time=01:00:00 process_files.sh
```

## Running a final step after the array

To combine results after all tasks finish, use a job dependency. The `--parsable` flag makes `sbatch` print only the job ID:

```
ARRAY_ID=$(sbatch --parsable process_files.sh)
sbatch --dependency=afterok:${ARRAY_ID} combine_results.sh
```

The second job starts only after **every** task in the array completes without errors. Here is a simple `combine_results.sh`:

```
#!/bin/bash
#SBATCH --job-name=combine
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --mem=1GB
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x_%j.out

cat results/result_*.txt > all_results.txt
```

If any task fails, the dependent job never starts. Cancel it with `scancel`, or use `afterany` instead of `afterok` to run it regardless.

## Best practices and common mistakes

- **Test first.** Run `sbatch --array=1 script.sh` before launching the full array.
- **Set `--time` yourself**, and keep it within the partition's maximum.
- **Keep the array within the submitted limit** of your partition, and bundle work into fewer tasks if needed.
- **Don't make tasks too short.** Tasks that last only seconds waste time on scheduling. Bundle them.
- **Write to unique output files.** Include the task ID in every output filename.
- **Create your `logs` directory first.**
- **Use absolute paths** in your scripts.
- **Request only what one task needs.** Resources are per task.
- **Request 1 GPU per task** on the `gpu` partition.

| Symptom | Likely cause |
| ------- | ------------ |
| No output files appear | The directory in `--output` does not exist |
| Every task processes the same file | The script is not using `${SLURM_ARRAY_TASK_ID}` |
| A task reads an empty line | The `--array` range is larger than your list file |
| `job violates accounting/QOS policy` | The array is larger than the partition's submitted job limit |
| Only some tasks run at once | You have reached the partition's running job limit, or your `%` throttle |
| Task ends with `TIMEOUT` | `--time` was too short; rerun with more time (up to the partition max) |
| Task ends with `OUT_OF_MEMORY` | `--mem` was too small; rerun that task with more |

## Practice exercise

You have 5 text files, and you want to count the lines in each one.

1. Create the data:

   ```
   mkdir -p exercise/data && cd exercise
   for i in 1 2 3 4 5; do seq 1 $(( i * 10 )) > data/book_${i}.txt; done
   ```

2. Make `filelist.txt` containing the absolute path of each file in `data/`.
3. Write an array script (`--array=1-5`, on `short`, with `--time=00:05:00`) that reads its file from `filelist.txt` and writes the line count to `results/book_<N>_lines.txt`.
4. Submit it and watch it with `squeue -u <username>`.
5. Run `seff` on one finished task. Did you ask for too much memory?
6. **Bonus:** Change `--array=1-5` to `--array=1-5%2` and watch how many tasks run at once.
7. **GPU bonus:** Copy `gpu_array.sh` and change the array to `--array=1-2`. Look in the logs to see which GPU each task received.

<details>
<summary>Click to see a sample solution</summary>

```
#!/bin/bash
#SBATCH --job-name=count_lines
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --mem=1GB
#SBATCH --time=00:05:00
#SBATCH --array=1-5
#SBATCH --output=logs/%x_%A_%a.out

INPUT_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" filelist.txt)
mkdir -p results
wc -l < "${INPUT_FILE}" > results/book_${SLURM_ARRAY_TASK_ID}_lines.txt
```

</details>

## Quick reference

| I want to... | Command |
| ------------ | ------- |
| Submit an array | `sbatch --array=1-100 script.sh` |
| Limit to 10 at a time | `sbatch --array=1-100%10 script.sh` |
| Rerun specific tasks | `sbatch --array=4,17,32 script.sh` |
| Get the task number in a script | `${SLURM_ARRAY_TASK_ID}` |
| Name output per task | `#SBATCH --output=logs/%x_%A_%a.out` |
| Check a partition's limits | `scontrol show partition short` |
| Give each task one GPU | `#SBATCH --partition=gpu` and `#SBATCH --gres=gpu:1` |
| See my queue, one line per task | `squeue -u <username> -r` |
| Cancel the array / one task | `scancel <arrayjobid>` / `scancel <arrayjobid>_<taskid>` |
| Check one task's efficiency | `seff <arrayjobid>_<taskid>` |
| List all task states | `sacct -j <arrayjobid> -X` |
| Run a job after an array | `sbatch --dependency=afterok:<arrayjobid> next.sh` |

Further reading:

- [Partitions](https://rc.northeastern.edu/partitions/)
- [Slurm Jobs Array](https://rc-docs.northeastern.edu/en/latest/runningjobs/slurmarray.html)
- [Understanding the Queuing System](https://rc-docs.northeastern.edu/en/latest/runningjobs/understandingqueuing.html)
- [Monitoring and Managing Jobs](https://rc-docs.northeastern.edu/en/latest/runningjobs/slurmmonitoringandmanaging.html)
- [GPU Access](https://rc-docs.northeastern.edu/en/latest/gpus/accessinggpus.html)
- [Slurm job array guide](https://slurm.schedmd.com/job_array.html)

## How to get help

Email the Research Computing team at <rchelp@northeastern.edu>.

Come to [office hours](https://rc.northeastern.edu/getting-help/) hosted on Zoom.

Or [book a consultation](https://rc.northeastern.edu/getting-help/) with an RC team member.

Review our [Documentation](https://rc-docs.northeastern.edu/en/latest/index.html).

Thank you!

---

*For questions or support, contact the Research Computing team at <rchelp@northeastern.edu>*
