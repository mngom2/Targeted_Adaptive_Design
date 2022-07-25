# Guide of Login to ThetaGPU, Load Environments, Run Jobs, and Install GPytorch

We can think about ThetaGPU as a related but seperate system from Theta. We need to login to Theta login node first, then ThetaGPU service node (which is
basically only for login purposes, always build packages and programs on compute node), and finally submit your job as a bash script or start an
interactive session on the compute node.

### 1. Login
Login to Theta
```
ssh alcfusername@theta.alcf.anl.gov
```
  and type in the mobile token password
  
ssh onto the thetagpu service node
```
ssh thetagpusnX
``` 
  where X is the Service Node Number, e.g. 1

Start an Interactive ssh Session on a Compute Node
```
qsub -A projectname -q single-gpu -n 1 -t 60 -I
```
  where:
* -A for project name
* -q for single-gpu or full-node
* -n number of resources (n gpu or n node)
* -t number of minutes you want to have
* -I indicates an interactive session. One can also remove -I and specify a executable bash script for it to run directly on the compute node

### 2. Once on a Compute Node, Load Modules

Load the Conda Environment (Module) with Pytorch, since our GPytorch has Pytorch dependency
```
module load conda/pytorch
conda activate
```
* Notice that we can check the available modules with "module avail" and check the loaded modules with "module list"

Create a virtual environment with python and activate it (For me, I create a folder under my project folder, i.e. /grand/projectname/tiany/python_venv as my path_to_myenv)
```
python -m venv --system-site-packages path_to_myenv
source path_to_myenv/bin/activate
```
Now the bash prompt should show that we're in the environment we just created, and we're good to use pip install
```
pip install gpytorch
```
If GPytorch finishes download with pytorch, numpy, etc. already satisfied, we're good. If it downloads them as well, there is a problem (and consider exit the compute node, come back again, rm -r the virtual environment folder you just created and do it again). We want to use ThetaGPU's packages (Pytorch, etc.) if it already exists. Not our own. --system-site-packages should tell it to link to the existing system packages and not downloading them in the virtual environment.

## After the First Time
After the first time, to run the files, simply activate the python_venv with 
```
module load conda/2022-07-01
source path_to_myenv/bin/activate
``` 
once on a compute node

## Using Jupyter Notebook to Run GPytorch on ThetaGPU
Now we know how to run jobs from the terminal. But sometimes we may want to run things with Jupyter Notebook. Here is the guide:
1. Go to [Jupyter Hub of ALCF](https://jupyter.alcf.anl.gov/), click Login ThetaGPU
2. Queue up for a "single-gpu". single-gpu should always work. If not, try again or run qstat from a terminal on service node of ThetaGPU to see the current jobs and queue. full-node might not work on ThetaGPU
3. Once on a compute node, click "New" and open a **terminal**
4. Run 
``` 
module load conda/2022-07-01
source <path_to_previously_created_python_venv>/bin/activate
python -m ipykernel install --user --name python_venv
```
Note: depending on the system and environment, you might need to install the "ipykernel" package first. The python_venv that I just created has the ipykernel module.

Go back to your .ipynb file, change kernel to python_venv from the dropdown menu, and we'll be good to run GPytorch!
