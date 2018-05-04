library(probly)
library(lme4)
library(brms)
library(future)
library(future.batchtools)
library(listenv)

if(grepl('(^n\\d|talapas-ln1)', system('hostname', intern = T))){
    simiter <- 100 #number of sims we have already generated.
    niter <- 2000
    nchains <- 6
    niterperchain <- ceiling(niter/nchains)
    warmup <- 1000
    data_dir <- '/gpfs/projects/dsnlab/flournoy/data/splt/probly/logistics'
}

if(!dir.exists(data_dir)) dir.create(data_dir)

#Formulae
formulae <- list(
    null_id_only_m_form = press_opt ~ 1 + (1 | id) + (1 | stim_image),

    null_m_form = press_opt ~ 1 + (1 | sample/id) + (1 | stim_image),

    time_m_form = press_opt ~ 1 + trial_index_c0_s +
        (1 + trial_index_c0_s | sample:id) +
        (1 | sample) +
        (1 + trial_index_c0_s | stim_image),

    condition_m_form = press_opt ~ 1 + condition +
        (1 + condition | sample:id) +
        (1 | sample) +
        (1 + condition | stim_image),

    timeXcondition_m_form = press_opt ~ 1 + condition*trial_index_c0_s +
        (1 + condition*trial_index_c0_s | sample:id) +
        (1 | sample) +
        (1 + condition*trial_index_c0_s | stim_image),

    timeXcondition_age_m_form = press_opt ~ 1 + condition*trial_index_c0_s*age_std*gender +
        (1 + condition*trial_index_c0_s | sample:id) +
        (1 | sample) +
        (1 + condition*trial_index_c0_s | stim_image),

    timeXcondition_dev_m_form = press_opt ~ 1 + condition*trial_index_c0_s*pds_std*gender +
        (1 + condition*trial_index_c0_s | sample:id) +
        (1 | sample) +
        (1 + condition*trial_index_c0_s | stim_image)
)

##brms models
plan(batchtools_slurm,
     template = system.file('batchtools', 'batchtools.slurm.tmpl', package = 'probly'),
     resources = list(ncpus = nchains, walltime = 60*24*1, memory = '1G',
                      partitions = 'short,fat,long,longfat'))

brms_results <- listenv()

for(i in seq_along(formulae)){
    message('Deploying brms model: ', names(formulae)[[i]])
    brms_results[[i]] <- future::future(
        probly::CachedFit(
            {
                afit <- brms::brm(formulae[[i]],
                                  family = 'bernoulli',
                                  data = splt,
                                  cores = nchains, chains = nchains,
                                  iter = warmup + niterperchain, warmup = warmup,
                                  control = list(adapt_delta = 0.99))
                afit <- add_ic(afit, ic = c("loo", "waic"))
                afit
            },
            rds_filename = file.path(data_dir,
                                     paste0(sub('_m_form$', '', names(formulae)[[i]]), '_bm.rds')),
            save_only = TRUE))
}

##lme4 models
plan(batchtools_slurm,
     template = system.file('batchtools', 'batchtools.slurm.tmpl', package = 'probly'),
     resources = list(ncpus = 1, walltime = 60*24*1, memory = '3G',
                      partitions = 'short,fat,long,longfat'))

lme4_results <- listenv()

for(i in seq_along(formulae)){
    message('Deploying lme4 model: ', names(formulae)[[i]])
    lme4_results[[i]] <- future::future(
        probly::CachedFit(
            lme4::glmer(formulae[[i]],
                        family = 'binomial',
                        data = splt),
            rds_filename = file.path(data_dir,
                                     paste0(sub('_m_form$', '', names(formulae)[[i]]), '_lm.rds')),
            save_only = TRUE))
}

print(lme4_results)
print(brms_results)

