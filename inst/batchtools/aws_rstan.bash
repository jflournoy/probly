########################################################
# Set up R and Stan on an Amazon EC2 instance
# Using Ubuntu 64-bit
# Christopher Gandrud
# 16 December 2014
# Partially from http://blog.yhathq.com/posts/r-in-the-cloud-part-1.html
# See yhat for EC2 instance set up
########################################################

# In your terminal navigate to key pair
# ssh -i YOUR_KEYPAIR.pem ubuntu@PUBLIC_DNS

sudo /bin/dd if=/dev/zero of=/var/swap.1 bs=1M count=3072
sudo /sbin/mkswap /var/swap.1
sudo chmod 600 /var/swap.1
sudo /sbin/swapon /var/swap.1

# Get all programs up to date
sudo apt-get update

# Install latest g++ and clang
sudo apt-get install gcc
sudo apt-get install clang

# Install prerequisites for RCurl
sudo apt-get install libcurl4-openssl-dev

# Check that you have the latest R installed
## see also: http://askubuntu.com/a/352438
sudo apt-get update
apt-cache showpkg r-base

sudo apt-get install libssl-dev
sudo apt-get install libxml2-dev

# If R is not the latest version add latest PACKAGE_VERSION returned
# from the pervious line to:
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
sudo add-apt-repository 'deb [arch=amd64,i386] https://cran.rstudio.com/bin/linux/ubuntu xenial/'
sudo apt-get update

sudo apt-get install r-base

# Install rJava
sudo apt-get install r-cran-rjava

# Install Stan and rstan
# First find the number of cores
lscpu

# Load r, set cores, and install
R
Sys.setenv(MAKEFLAGS = "-j4")
install.packages("rstan", type = "source")

# Install other packages I often use
## Dependencies include reshape2, dplyr
pkgs <- c('repmis', 'devtools', 'future', 'future.batchtools', 'listenv', 'tidyverse')
install.packages(pkgs)
devtools::install_github('jflournoy/probly')
devtools::install_github('jflournoy/qualtrics')
devtools::install_github('jflournoy/milav')

q()

######## Running Scripts and Downloading Results onto Local ####
# Run a script so that it runs even if the terminal hands up
# sudo nohup Rscript R_SCRIPT_PATH

# Download a file from Amazon EC2 to your local machine
# (current working directory)
# scp -i YOUR_KEYPAIR.pem ubuntu@PUBLIC_DNS:EC2_FILE_PATH .
