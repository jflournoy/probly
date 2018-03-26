library(future)
library(future.batchtools)

plan(batchtools_slurm,
     template = system.file('batchtools', 'batchtools.slurm.tmpl', package = 'probly'),
     resources = list(ncpus = 4, walltime = 60*24-1, memory = '1G'))

fit_size_f <- future({
    data_dir <- '/home/flournoy/otherhome/data/splt/probly/'
    source(system.file('r_aux', 'simulate_data_from_splt.R', package = 'probly'))
    source(system.file('r_aux', 'estimate_rl_from_sim.R', package = 'probly'))
})

print(value(fit_size_f))
