Model trained on triaxial_compression.hdf5 dataset.

- Contact_params (7): 
  - young's modulus  $10^E$ 
  - poisson's ratio $\nu$
  - rolling stiffness $k_r$
  - rolling friction $\eta$
  - sliding friction $\mu$
  - confinement pressure
  - experiment type: drained/undrained

- inputs sequences(200, 3):
  - strain in x $\varepsilon_x$
  - strain in y $\varepsilon_y$
  - strain in z $\varepsilon_z$ 

- outputs sequences (200, 4):
  - void ratio $e$
  - mean stress $p$
  - deviatoric stress $q$
  - average contact normal force $f_0$
  - fabric anisotropy $a_c$
  - mechanical anisotropy $a_n$
  - mechanical anisotropy due to tangential forces $a_t$
