# H2 benchmarking data

Data and analysis used in arXiv:2305.xxxx. 

## Project Organization
------------

    ├── README.md   
    ├── requirements.txt          
    ├── /component_benchmarks    <- component benchmarking data and analysis
    ├── /ghz                     <- ghz state prep system-level benchmark data and analysis
    ├── /holoquads               <- holoquads algorithmic benchmark data and analysis
    ├── /mirror_benchmarking     <- mirror benchamrking system-level benchmark data and analysis
    ├── /qaoa                    <- QAOA algorithmic benchmark data and analysis
    ├── /parameterized_rzz       <- parameterized Rzz gate component benchmarking data and analysis
    ├── /quantum_volume          <- quantum volume system-level benchmark data and analysis
    ├── /repetition_code         <- repetition code system-level benchmark data and analysis
    ├── /rcs                     <- random cicuit sampling system-level benchmark data and analysis
    └── /tfim                    <- transverse field Ising model algorithmic benchmark data and analysis

--------

## Getting Started

No installation required. Each test has a directory that contains a notebook for loading data and creating the plots in arXiv:2305.xxxx. Data is saved in json files that contain circuits that were run and list of observed outputs. Requirements are listed in the requirements.txt file but may not be needed for every notebook.

<div align="center"> &copy; 2023 by Quantinuum. All Rights Reserved. </div>