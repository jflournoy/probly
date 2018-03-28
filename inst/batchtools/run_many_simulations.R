library(future)
library(future.batchtools)
library(probly)

plan(batchtools_slurm,
     template = system.file('batchtools', 'batchtools.slurm.tmpl', package = 'probly'),
     resources = list(ncpus = 1, walltime = 60*24-1, memory = '1G'))

data_dir <- '/home/flournoy/otherhome/data/splt/probly/'
if(!file.exists(data_dir)){
    stop('Data directory "', data_dir, '" does not exist')
} else {
    message("Data goes here: ", data_dir)
}

data(splt)
splt_no_na <- splt[!is.na(splt$pressed_r), ]

iter <- 100
results <- list()

for(i in 1:iter){
    results[[i]] %<-% probly::make_task_structure_from_data(splt_no_na)
}

saveRDS(results, file.path(data_dir, 'test_rez.RDS'))

