# 573-SADTech

## Team members
  1. Abhinav Patil (abhinavp)
  1. Didi O'Connell (danieloc)
  1. Sam Briggs (briggs3)
  1. Tara Wueger (taraw28)

## D2 Run Instructions
  1. Download data/ and outputs/ directory from [Google Drive](https://drive.google.com/drive/folders/1f6dj4FIjZ8c6yNKZ6FTk7oA3ejqObalY). The structure of the data/ directory should look like:
  ```
    data/
      mal/
        audio/
          MAL_MSA_01.mp3
          ...
          MAL_MSA_70.mp3
        text/
          MAL_MSA_01.txt
          ...
          MAL_MSA_70.txt
        video/
          MAL_MSA_01.mp4
          ...
          MAL_MSA_70.mp4
        all.csv
        dev.csv
        train.csv
      tam/
        audio/
          TAM_MSA_01.mp3
          ...
          TAM_MSA_64.mp3
        text/
          TAM_MSA_01.txt
          ...
          TAM_MSA_64.txt
        video/
          TAM_MSA_01.mp4
          ...
          TAM_MSA_64.mp4
        all.csv
        dev.csv
        train.csv
  ```
  The outputs/ directory should look like:
  ```
    outputs/
      D2/
        mal/
        tam/
  ```
  Note: You might need to merge the outputs/ folder downloaded from Google Drive with the outputs folder in the git repository.
  
  2. For condor: Run `condor_submit D2.sh` from root directory
  3. For local: Run  `./src/d2_run.sh` from root directory

## Shared Task
TODO

## Development & Contribution Guidelines

### Requirements

You will only need to make sure you have a recent version of Anaconda. All other requirements are listed in `environment.yml`/`gpu_environment.yml` and installed/managed by conda.

### Local Development Setup

1. After installing conda, it is highly recommended (but optional) that you run the following before proceeding (this will set `channel_priority` to `strict` as the default for all environments; if you know what this means, and don't want this, it is recommended you at least set this value to `strict`  before installing the environment in the following steps; you can then, later, reverse this config setting globally, and keep it just for the environment in question- see step 4. **If none of this means anything to you, just run the code below.**)
    ```sh
    conda config --set channel_priority strict
    ```
1. Create a fresh conda environment using `environment.yml` (if you haven't done so for this project previously):
    ```sh
    conda env create -f environment.yml
    ```
    By default this will create a conda env whose name is indicated on the first line of the `environment.yml` file (presently, `SADTech`). You can change this by adding the `-n` flag followed by the desired name of your environment.
1. After the environment is created, whenever you want to work on this project, first activate the environment:
    ```sh
    conda activate SADTech
    ```
1. Run the following code.
    ```sh
    conda config --env --set channel_priority strict
    ```
    This will set strict channel priority for _just_ the current environment, so if you intend to reverse step 1 as mentioned there, you can now do so, changing the global setting by running a similar command as above but without the `--env` flag and with a different value than `strict` (whatever value you want). **If you don't know what any of this means, disregard; just make sure you ran the code above.**
1. When you are done, you can exit the environment with `conda deactivate`.
1. If you pull code from the repo and the `environment.yml` file has changed, update your environment by running the following (after activating the environment):
    ```sh
    conda env update -f environment.yml --prune
    ```
1. Remember: once you've run steps 1, 2, and 4, you won't need to repeat them again. Just activate as in step 3 and deactivate when done as in step 4.

### Hyak Development Setup
TODO

### Patas Development Setup

TODO: Once Patas is working, create environments for shared use in a shared directory so we don't all have to recreate
environments each time. The following instructions assume that has already been done.

TODO: The following instructions mention a CPU/GPU `environment.yml` file distinction, but we still need to create the GPU environment, probably on Hyak for the time being.

On Patas, we have already created two environments for use with this project. One is for use with GPU nodes, and the other with CPU nodes (including the head node that you would normally ssh into) (TODO: actually, no we have not yet).

Instructions on setting up Conda on Patas can be found [here](https://www.shane.st/teaching/575/spr22/patas-gpu.pdf). n.b.: you will have to go to the Anaconda website and find the link to the most recent version, as the link in this PDF is out of date.

#### Head Node Use (w/o Condor)

After installing conda as above, you may wish to test small changes while working on your own account on the head node. To do so, you will want to first activate the CPU environment like so:

```sh
conda activate TODO:/path/to/SADTech/env
```

As always, please abide by general Patas etiquete and avoid running jobs on the head node that require non-trivial amounts of CPU or memory usage.

#### Condor: CPU or GPU Nodes

There are two ways to tell Condor to use the environment when running a job. The first works for CPU or GPU nodes, while the second works only for CPU nodes.

##### Method A

1. In your Condor submit file, add a line saying `getenv = False` (or edit if `getenv` is already there)
1. Add these two lines near/at the top of the shell script (executable) that you are submitting to Condor, adjusting the first line if your condor installation is elsewhere:

For CPU nodes:
```sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate TODO:/path/to/SADTech/environment.yml
```

For GPU nodes:
```sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate TODO:/path/to/SADTech/gpu_environment.yml
```

Note that you will also have to edit your Condor submit file to request GPU nodes; for instructions regarding how to do that, see the document linked to near the top of this README that also contain the instructions for installing conda on Patas.

##### Method B
n.b.: This only works for CPU nodes.

1. While logged into your Patas account on the Patas node, run `conda activate TODO:/path/to/SADTech/environment.yml` (unless you are already working within this environment)
1. Add `getenv = True` to your Condor submit file
1. Call `condor_submit` with the submit file as per usual.

### Contribution Guidelines

For any non-trivial changes, please work on your own branch rather than on `main` and submit a PR when you are ready to merge your changes.

If you need any new packages, install them with `conda install PACKAGE_NAME`. Then, before committing, run:

```sh
conda env export --from-history | grep -vE "^(prefix):" > environment.yml
```

Replace `environment.yml` with `gpu_environment.yml` as appropriate; also, if you have changed the environment name on your own setup to something other than `SADTech`, please manually edit the resulting YAML file before committing to use the standard name.

This makes sure the `prefix:` line automatically created by Conda's `export` command are not included, since this can vary by platform/machine.

Then make sure the updated `(gpu_)environment.yml` file is included with your commit. Note: if you did not install the command with `conda install`, the above command will not work properly, due to the `--from-history` flag. However, using this flag is necessary to ensure the `requirements.yml` file is platform-agnostic. Therefore, please only install packages via `conda install` (or by manually adding requirements to the YAML files).

Please manually edit the YAML file to include appropriate version number strings if at all possible. If you installed
without specifying an explicit version string, it won't be included with the `--from-history` flag.


## Directory Structure

[WIP]

