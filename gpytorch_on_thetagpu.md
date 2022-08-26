# Guide of Login to ThetaGPU, Load Environments, Run Jobs, and Install GPytorch

We can think about ThetaGPU as a related but seperate system from Theta. We need to login to Theta login node first, then ThetaGPU service node (which is
basically only for login purposes, always build packages and programs on compute node), and finally submit your job as a bash script or start an
interactive session on the compute node.

#### 1. Login and queue a job
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
* -q for single-gpu or full-node. If using full-node, it would be better to add `-M <youremail>` as well, so that you will receive an email when your job is starting.
* -n number of resources (n gpu or n node)
* -t number of minutes you want to have
* -I indicates an interactive session. One can also remove -I and specify a executable bash script for it to run directly on the compute node

If queueing for an interactive session, once it is running, we can use `qstat -u <yourusername>` **on a service node** (use `ssh thetagpusn1` to go to a thetagpu service node if on a compute node or thetalogin node) to see our job id and allocated node. 

Then, on the service node, do
```
ssh thetagpuXX
```
where XX is the allocated node number, to ssh directly onto the allocated compute node and start the interactive session.

#### 2. Once on a Compute Node, Load Modules

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

### After the First Time
After the first time, to run the files, simply activate the python_venv with 
```
module load conda/2022-07-01
source path_to_myenv/bin/activate
``` 
once on a compute node

## Using Jupyter Notebook to Run GPytorch on ThetaGPU
Now we know how to run jobs from the terminal. But sometimes we may want to run things with Jupyter Notebook. Here is the guide:

### Approach 1 - Use Jupyter Hub
1. Go to [Jupyter Hub of ALCF](https://jupyter.alcf.anl.gov/), click Login ThetaGPU
2. Queue up for a "single-gpu". single-gpu should always work. If not, try again or run qstat from a terminal on service node of ThetaGPU to see the current jobs and queue. full-node might not work on ThetaGPU

**For the first time only, one needs to set up the environment and Kernel by follow these extra steps**

4. Once Jupyter Notebook is launched on a compute node, click "New" and open a **terminal**
5. Run 
``` 
module load conda/2022-07-01
source <path_to_previously_created_python_venv>/bin/activate
python -m ipykernel install --user --name python_venv
```
Note: depending on the system and environment, you might need to install the "ipykernel" package first. The python_venv that I just created has the ipykernel module.

Go back to your .ipynb file, change kernel to python_venv from the dropdown menu, and we'll be good to run GPytorch!

### Approach 2 - Use ssh tunnel (recommended when running with a full node and/or having several hours of Wall Time)
To use ssh tunnel, we first need to be in an interactive session on a compute node. See Part 1, "Log in and queue a job" for more details on this.

After on a compute node, follow these steps:
1. On the compute node terminal, do
```
module load conda/2022-07-01
conda activate
jupyter notebook
```
You should see a line like `http://localhost:XXXX/`, where XXXX is the port number that jupyter notebook is launched on the compute node, usually the default 8888. If it is not 8888, replace 8888 in the following with your port number.

2. Then, on a **new, local terminal**, do
```
export PORT_NUM=8889
ssh -L $PORT_NUM:localhost:8888 <yourusername@theta.alcf.anl.gov>
ssh -L 8888:localhost:8888 thetagpuXX
```
where XX is your allocated compute node number.

3. Finally, navigate to localhost:8889 in your browser, and you should see a jupyter notebook, and it's on ThetaGPU compute node!
Notice that for the first time doing this, one might need to input some password or weird key. Just follow the direction on that page.

(Essentially, the above steps, using ssh, sets the local port 8889 to listen to the allocated compute node port 8888 which we initiated a jupyter notebook.)

**For the first time only, one needs to set up the environment and Kernel by follow these extra steps**

Click "New" and open a **terminal**, and run
``` 
module load conda/2022-07-01
source <path_to_previously_created_python_venv>/bin/activate
python -m ipykernel install --user --name python_venv
```
Note: depending on the system and environment, you might need to install the "ipykernel" package first. The python_venv that I just created has the ipykernel module.

Go back to your .ipynb file, change kernel to python_venv from the dropdown menu, and we'll be good to run GPytorch!
