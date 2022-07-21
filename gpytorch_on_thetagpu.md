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

