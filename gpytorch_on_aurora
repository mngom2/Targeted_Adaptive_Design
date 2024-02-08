# Guide of Login to Aurora Load Environments, Run Jobs, and Install GPytorch

#### 1. Login and queue a job
Login to Aurora
```
ssh <username>@bastion.alcf.anl.gov
```
Then, after entering your passcode
ssh <username>@login.aurora.alcf.anl.gov

(We are supposion you already set the environment variables that provide access to the proxy host. )

Start a sessiom, for example Interactive ssh Session on a Compute Node
```
qsub -I -q EarlyAppAccess -l select=1,walltime=60:00 -A Aurora_deployment
```



#### 2. Once on a Compute Node, Load Modules

```
module use /soft/modulefiles
module load frameworks
python3 -m venv --system-site-packages env_gpytorch
source env_gpytorch/bin/activate
python3 -m pip install gpytorch
```

Create a 'activation_env.sh' file that contains
```
module use /soft/modulefiles
module load frameworks
source env_gpytorch/bin/activate
``` 
and do `source activation_env.sh` to activate your environment for subsequent runs.

## Running on GPUs
To run on GPUs, one needs to add
```
import intel_extension_for_pytorch as ipex
 ```
and 
set the device as follows:

```
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.xpu.is_available():
    device = torch.device('xpu')
else: 
    device = torch.device('cpu')
```
(One might need to install an earlier version of GPytorch for multiple GPUs running.)
