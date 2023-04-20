# distributed_storage_trace_generator
This is the repo for the distributed trace generator

# Trace Data
Data source are FIU home, webmail and Nexus 5 traces found in SNIA Repository found in this link http://iotta.snia.org/traces/block-io
Traces are not present in the git because of the space consideration. 

# Scripts
The python scripts ar eto generate the plots of the best distribution that a particular metric follows after distribution fitting. 
There are two kind of script: A) Distribution fitting and plot generation for FIU and nexus trace namely "script_nexus.py", "distribution_script", "distribution_script". B) Code to migrate the plots generated using pkl so that the generated plots can be interactible using matplotlib across system rather than viewing static image plot. 

# Plots Directory
The directory *_plots contain the plots in .pkl format for all the traces used in the study

# Overleaf link

The paper can be viewed at https://www.overleaf.com/read/ggjddrtpnvjj
